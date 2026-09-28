"""Git worktrees, branches, checkpoints, merge and discard.

The only module that knows git exists. Worktrees are the required
isolation boundary: seven agents can work on one repo at once because
none of them shares a checkout, and none of them can reach your base branch
without a deliberate `buddy merge`.

Every git call goes through `asyncio.create_subprocess_exec`.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from buddy import leaks
from buddy.config import Config
from buddy.models import RunOutcome, TaskRun, TaskSpec, utcnow
from buddy.processes import release

#: Identity for the WIP commits Buddy makes on the harness's behalf.
#: Explicit, so a repo with no configured user still checkpoints, and so the
#: safety commit is never mistaken for the agent's own work.
WIP_AUTHOR_NAME = "Buddy"
WIP_AUTHOR_EMAIL = "buddy@localhost"

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


class WorkspaceError(Exception):
    pass


class TamperedWorktree(WorkspaceError):
    """A worktree whose git metadata no longer leads back to its repository.

    A worktree's `.git` is a file *inside* the worktree, naming the directory
    git keeps for it, which in turn names the repository's own `.git`. An
    agent can rewrite either, and git on the host then reads whatever config
    it points at - `core.fsmonitor` is a command, `core.hooksPath` a place
    to run hooks from. Buddy runs git in every worktree when an attempt ends,
    so without this check a sandboxed agent's checkpoint ran the agent's own
    code on your machine. Measured, before the fix.
    """

    def __init__(self, worktree: Path, why: str) -> None:
        super().__init__(
            f"{worktree}: {why}. Buddy ran no git there; whatever the agent left "
            "uncommitted was not checkpointed."
        )
        self.worktree = worktree


#: Every git Buddy runs gets this. None of it is interactive, and a git that
#: prompts for credentials does so on the terminal - which is Buddy's own
#: session - while the scheduler waits on an answer nobody knows it wants.
GIT_ENV = {"GIT_TERMINAL_PROMPT": "0"}


class MergeConflict(WorkspaceError):
    """A merge that git could not complete.

    Never resolved automatically: Buddy reports it, and you either fix
    it yourself or ask for a task that rebases and resolves.
    """

    def __init__(self, task_id: str, branch: str, into: str, detail: str) -> None:
        super().__init__(
            f"merging {branch} into {into} conflicts. Nothing was changed in your checkout. "
            f"Resolve it yourself, or `buddy resolve {task_id}` to have an agent do it.\n{detail}"
        )
        self.task_id = task_id
        self.branch = branch
        self.into = into
        self.detail = detail


#: Worktrees need 2.20.
MIN_GIT_VERSION = (2, 20)


async def git_version() -> tuple[int, ...]:
    """The installed git's version, or a `WorkspaceError` saying why not.

    Setup needs this and only this module knows git, so it lives here
    rather than as a second `git --version` somewhere else.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, PermissionError) as exc:
        raise WorkspaceError(f"git is not on PATH: {exc}") from exc
    try:
        out, err = await proc.communicate()
    finally:
        release(proc)
    text = out.decode(errors="replace").strip()
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if proc.returncode != 0 or not match:
        raise WorkspaceError(f"could not read git's version: {text or err.decode().strip()}")
    return tuple(int(part) for part in match.groups() if part is not None)


def slugify(title: str, *, max_length: int = 40) -> str:
    """A branch-safe slug for `buddy/<task_id>-<slug>`.

    The task id already makes the branch unique, so this only has to be
    readable in `git branch` and legal as a ref.
    """
    slug = _SLUG_STRIP.sub("-", title.lower()).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or "task"


def branch_name(task: TaskSpec) -> str:
    """`buddy/t-0142-fix-onboarding`."""
    return f"buddy/{task.id}-{slugify(task.title)}"


@dataclass(frozen=True)
class Checkout:
    """Where a run's code lives.

    For a git project this is a worktree on its own branch. For a non-git
    project it is the project directory itself, with no branch, which is why
    such a project takes a one-slot lock.
    """

    path: Path
    branch: str | None
    base_ref: str
    is_worktree: bool


class SecretsInBranch(WorkspaceError):
    """A branch that would bring a secret into your checkout."""

    def __init__(self, branch: str, findings: list[leaks.Finding], *, overridable: bool) -> None:
        listed = "\n".join(f"  {finding.describe()}" for finding in findings)
        remedy = (
            "If these are lookalikes you have checked, merge with --allow-secret-patterns."
            if overridable
            else "One is a key of yours; remove it from the branch - there is no override."
        )
        super().__init__(
            f"not merging {branch}: it adds what looks like secrets. Nothing was changed.\n"
            f"{listed}\n{remedy} If a real key was ever pushed anywhere, rotate it."
        )
        self.branch = branch
        self.findings = findings
        self.overridable = overridable


class Workspace:
    def __init__(
        self, config: Config, *, known_secrets: Callable[[], dict[str, str]] | None = None
    ) -> None:
        self.config = config
        # Merges into a project are serialized.
        self._merge_locks: dict[str, asyncio.Lock] = {}
        #: Your real keys, for recognising them whatever they look like. Buddy's
        #: own process may read its keychain entries without a prompt.
        self.known_secrets = known_secrets or leaks.known_secrets
        #: What a checkpoint kept out of its commit, by (task id, attempt).
        self.withheld: dict[tuple[str, int], list[str]] = {}

    # -- git plumbing ------------------------------------------------------

    async def _git(
        self, repo: Path, *args: str, check: bool = True, env: dict[str, str] | None = None
    ) -> str:
        code, stdout, stderr = await self._git_status(repo, *args, env=env)
        if check and code != 0:
            raise WorkspaceError(
                f"git {' '.join(args)} in {repo} failed ({code}): "
                f"{stderr.strip() or stdout.strip()}"
            )
        return stdout

    async def _git_status(
        self, repo: Path, *args: str, env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo),
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **GIT_ENV, **(env or {})},
        )
        try:
            out, err = await proc.communicate()
        finally:
            release(proc)
        return (
            proc.returncode or 0,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    # -- git in a task's worktree, where the agent has been -----------------

    def _owner(self, task_id: str, worktree: Path) -> str | None:
        """The project whose Buddy worktree this is, or None.

        Decided from config and the task id, never from anything in the
        directory: a project that is not a git repository runs in place, and
        an agent there can `git init` and configure the result as it likes.
        """
        for name in self.config.projects:
            if self.config.worktree_path(name, task_id) == worktree:
                return name
        return None

    async def _common_dir(self, repo: Path) -> Path:
        """The project's own `.git`, asked of its own checkout."""
        found = Path((await self._git(repo, "rev-parse", "--git-common-dir")).strip())
        return (found if found.is_absolute() else repo / found).resolve()

    async def _pinned(self, project: str, worktree: Path) -> dict[str, str]:
        """The environment that makes git in `worktree` read only what Buddy trusts.

        `GIT_COMMON_DIR` is the project's own `.git`, learned from the
        project's checkout; `GIT_DIR` is the worktree's own directory inside
        it. Neither is taken from the worktree's say-so, so its `.git` file
        and the `commondir` it leads to can say what they like: git on the
        host reads the repository's config and hooks and nothing else.
        """
        common = await self._common_dir(self.config.project(project).path)
        marker = worktree / ".git"
        if marker.is_symlink() or not marker.is_file():
            raise TamperedWorktree(worktree, ".git is no longer the file git wrote")
        try:
            text = marker.read_text()[:4096].strip()
        except OSError as exc:
            raise TamperedWorktree(worktree, f".git cannot be read ({exc})") from exc
        if not text.startswith("gitdir:"):
            raise TamperedWorktree(worktree, ".git no longer names a git directory")
        # Relative when git's `worktree.useRelativePaths` is on.
        admin = (worktree / text.removeprefix("gitdir:").strip()).resolve()
        if admin.parent != common / "worktrees":
            raise TamperedWorktree(worktree, f".git points outside {common}")
        # And back again. `GIT_COMMON_DIR` keeps config and hooks where they
        # belong whatever this file says, but git 2.39 still writes refs
        # where it points - so a rewritten one sent the checkpoint elsewhere.
        try:
            back = (admin / (admin / "commondir").read_text().strip()).resolve()
        except OSError as exc:
            raise TamperedWorktree(worktree, f"its commondir cannot be read ({exc})") from exc
        if back != common:
            raise TamperedWorktree(worktree, f"its commondir points outside {common}")
        # Only read when the repository enables `extensions.worktreeConfig`,
        # and never created by Buddy: one here was put here.
        if (admin / "config.worktree").exists():
            raise TamperedWorktree(worktree, "it has a config.worktree of its own")
        return {
            "GIT_DIR": str(admin),
            "GIT_COMMON_DIR": str(common),
            "GIT_WORK_TREE": str(worktree),
        }

    async def is_git_repo(self, path: Path) -> bool:
        code, out, _ = await self._git_status(path, "rev-parse", "--is-inside-work-tree")
        return code == 0 and out.strip() == "true"

    async def requires_exclusive_slot(self, project: str) -> bool:
        """True for a non-git project: two agents in one directory is the
        exact hazard worktrees exist to prevent."""
        return not await self.is_git_repo(self.config.project(project).path)

    async def branch_exists(self, repo: Path, branch: str) -> bool:
        code, _, _ = await self._git_status(
            repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"
        )
        return code == 0

    async def is_clean(self, path: Path, *, env: dict[str, str] | None = None) -> bool:
        return not (await self._git(path, "status", "--porcelain", env=env)).strip()

    async def current_branch(self, path: Path, *, env: dict[str, str] | None = None) -> str:
        return (await self._git(path, "rev-parse", "--abbrev-ref", "HEAD", env=env)).strip()

    # -- creating a run's checkout ---------------------------------

    async def create(self, task: TaskSpec) -> Checkout:
        """The worktree and branch for a task.

        Idempotent by design, not by accident: a requeued attempt reuses the
        same worktree and branch so its partial work is still there to resume
        from.
        """
        project = self.config.project(task.project)
        repo = project.path
        if not repo.exists():
            raise WorkspaceError(f"project {task.project!r} path does not exist: {repo}")

        if not await self.is_git_repo(repo):
            # Runs in place, under a one-slot lock enforced by the manager.
            return Checkout(path=repo, branch=None, base_ref="", is_worktree=False)

        worktree = self.config.worktree_path(task.project, task.id)
        branch = branch_name(task)

        if worktree.exists():
            # A previous attempt ran here, so the agent has had its hands on
            # the worktree's git metadata: asked through the pinned view only.
            pinned = await self._pinned(task.project, worktree)
            existing = await self._git(worktree, "rev-parse", "HEAD", env=pinned)
            return Checkout(
                path=worktree,
                branch=await self.current_branch(worktree, env=pinned),
                base_ref=existing.strip(),
                is_worktree=True,
            )

        worktree.parent.mkdir(parents=True, exist_ok=True)

        if await self.branch_exists(repo, branch):
            # The worktree is gone but the branch is not: a shutdown removed
            # the working copy and kept the work, or a discard did.
            # `-b` would fail here with "a branch named ... already exists",
            # so the existing branch is checked out instead and the attempt
            # resumes from its last checkpoint rather than from the base.
            await self._git(repo, "worktree", "add", str(worktree), branch)
            head = (await self._git(worktree, "rev-parse", "HEAD")).strip()
            return Checkout(path=worktree, branch=branch, base_ref=head, is_worktree=True)

        # No fetch first. The branch is cut from the *local* base, which is
        # what `buddy merge` merges back into, so a fetch changed nothing a
        # task starts from - and a remote that wanted credentials prompted
        # on Buddy's own terminal while the scheduler waited on it.
        #
        # A conflict fix starts on the branch that conflicted; every
        # other task starts on the project's base.
        start = task.start_from or project.base_branch
        base_ref = (await self._git(repo, "rev-parse", start)).strip()
        await self._git(repo, "worktree", "add", str(worktree), "-b", branch, base_ref)
        return Checkout(path=worktree, branch=branch, base_ref=base_ref, is_worktree=True)

    # -- checkpointing ---------------------------------------------

    async def checkpoint(self, run: TaskRun, outcome: RunOutcome | None = None) -> str | None:
        """Commit anything uncommitted at the end of a run, however it ended.

        In the happy path this is a no-op, because the brief asks the harness
        to commit its own work with real messages. In the unhappy path
        (kill, timeout, crash) it is the reason nothing is lost.

        Only ever in a worktree Buddy made, and only through `_pinned`: this
        runs git on the host in a directory the agent has just had full
        control of. A worktree whose metadata was tampered with is left
        alone and said so, in the run's result, rather than trusted.
        """
        project = self._owner(run.task_id, run.worktree)
        if project is None or not run.worktree.exists():
            return None  # a non-git project runs in place, with no branch to commit to
        try:
            pinned = await self._pinned(project, run.worktree)
        except TamperedWorktree as exc:
            self.withheld.setdefault((run.task_id, run.attempt), []).append(str(exc))
            return None
        if await self.is_clean(run.worktree, env=pinned):
            return None

        label = (outcome or run.outcome or RunOutcome.INTERRUPTED).value
        await self._git(run.worktree, "add", "-A", env=pinned)
        withheld = await self._unstage_secrets(run.worktree, pinned)
        if withheld:
            self.withheld.setdefault((run.task_id, run.attempt), []).extend(withheld)
        staged, _, _ = await self._git_status(
            run.worktree, "diff", "--cached", "--quiet", env=pinned
        )
        if staged == 0:
            return None  # everything left over was a secret; nothing to commit
        await self._git(
            run.worktree,
            "-c",
            f"user.name={WIP_AUTHOR_NAME}",
            "-c",
            f"user.email={WIP_AUTHOR_EMAIL}",
            "commit",
            "--no-verify",
            "-m",
            f"WIP (buddy, {label}, attempt {run.attempt})",
            env=pinned,
        )
        return (await self._git(run.worktree, "rev-parse", "HEAD", env=pinned)).strip()

    async def _unstage_secrets(self, worktree: Path, pinned: dict[str, str]) -> list[str]:
        """Take secrets back out of the index before Buddy commits.

        `git add -A` stages whatever an agent left behind, and Buddy commits
        with `--no-verify`, so no hook of the project's would catch it. A file
        that holds secrets by its nature, or whose contents include your key
        or a provider's key shape, is unstaged: it stays in the worktree,
        yours, and out of git. What was withheld is recorded so it is said.
        """
        names = await self._git(
            worktree, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR", env=pinned
        )
        known = self.known_secrets()
        withheld: list[str] = []
        for path in filter(None, names.split("\0")):
            if leaks.is_sensitive_path(path):
                reason = "holds secrets by its nature"
            else:
                code, content, _ = await self._git_status(worktree, "show", f":{path}", env=pinned)
                findings = leaks.scan_text(content, path, known) if code == 0 else []
                if not findings:
                    continue
                reason = ", ".join(sorted({finding.kind for finding in findings}))
            await self._git(worktree, "reset", "-q", "--", path, env=pinned)
            withheld.append(f"{path}: {reason}")
        return withheld

    async def secrets_in_branch(
        self, task: TaskSpec, *, into: str | None = None, head: str | None = None
    ) -> list[leaks.Finding]:
        """What the task's branch adds that looks like a secret.

        `head` pins the commit examined, so a merge can scan exactly what it
        is about to merge rather than whatever the branch says a moment later.
        """
        project = self.config.project(task.project)
        target = into or project.base_branch
        span = f"{target}...{head or branch_name(task)}"
        findings: list[leaks.Finding] = []
        added = await self._git(project.path, "diff", "--name-only", "-z", "--diff-filter=A", span)
        for path in filter(None, added.split("\0")):
            if leaks.is_sensitive_path(path):
                findings.append(leaks.Finding(path, 0, "a file that holds secrets by its nature"))
        # Prefixes pinned so the parser never depends on your git config.
        # (Measured: `diff.noprefix` does change them; the parser copes
        # either way, and this keeps it from having to.)
        diff = await self._git(
            project.path,
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "-U0",
            span,
        )
        findings.extend(leaks.scan_added_lines(diff, self.known_secrets()))
        return findings

    # -- inspecting -------------------------------------------------

    async def diff_stat(self, task: TaskSpec, *, into: str | None = None) -> str:
        return await self._diff(task, into=into, stat=True)

    async def diff(self, task: TaskSpec, *, into: str | None = None) -> str:
        return await self._diff(task, into=into, stat=False)

    async def _diff(self, task: TaskSpec, *, into: str | None, stat: bool) -> str:
        project = self.config.project(task.project)
        target = into or project.base_branch
        args = ["diff"]
        if stat:
            args.append("--stat")
        # Three dots: what this branch added since it diverged, not the
        # unrelated work that landed on the base meanwhile.
        args.append(f"{target}...{branch_name(task)}")
        return await self._git(project.path, *args)

    # -- merging ----------------------------------------------------

    def _merge_lock(self, project: str) -> asyncio.Lock:
        return self._merge_locks.setdefault(project, asyncio.Lock())

    async def merge(
        self, task: TaskSpec, *, into: str | None = None, allow_secret_patterns: bool = False
    ) -> str:
        """Merge a finished task into your real branch, in your own checkout.

        The one action that touches your working tree, so it refuses rather
        than surprises: the checkout must be on the target branch and clean,
        and a conflict is aborted and reported.
        """
        project = self.config.project(task.project)
        repo = project.path
        target = into or project.base_branch
        branch = branch_name(task)

        async with self._merge_lock(task.project):
            if not await self.is_git_repo(repo):
                raise WorkspaceError(
                    f"project {task.project!r} is not a git repo; nothing to merge"
                )

            on = await self.current_branch(repo)
            if on != target:
                raise WorkspaceError(
                    f"your checkout at {repo} is on {on!r}, not {target!r}. "
                    f"Switch to {target!r} and merge again."
                )
            if (unfinished := await self.merge_in_progress(task.project)) is not None:
                # Checked before cleanliness, because it is the better answer:
                # a half-done merge is also a dirty checkout.
                raise WorkspaceError(f"merge already in progress: {unfinished}")
            if not await self.is_clean(repo):
                raise WorkspaceError(
                    f"your checkout at {repo} has uncommitted changes. "
                    "Commit or stash them, then merge again."
                )
            # One commit, named once: what is scanned below is exactly what is
            # merged, even if something moves the branch in between.
            code, out, _ = await self._git_status(
                repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}^{{commit}}"
            )
            if code != 0:
                raise WorkspaceError(f"there is no branch {branch} to merge")
            head = out.strip()
            # The one gate every agent's commits pass through on their way to
            # your branch. Your own keys are never let through; a provider key
            # shape can be, once you have looked at it.
            findings = await self.secrets_in_branch(task, into=target, head=head)
            yours = [f for f in findings if f.kind.startswith("your ")]
            if yours or (findings and not allow_secret_patterns):
                raise SecretsInBranch(branch, findings, overridable=not yours)

            try:
                code, out, err = await self._git_status(
                    repo,
                    "merge",
                    "--no-ff",
                    head,
                    "-m",
                    f"Merge {branch} ({task.title})",
                )
            except BaseException:
                # Cancelled or failed with git possibly mid-merge. There was
                # no merge in progress a moment ago, so any there now is ours
                # to undo - shielded, so a second Ctrl-C cannot interrupt the
                # cleanup of the first. A SIGKILL runs none of this; that case
                # is `merge_in_progress`, found afterwards.
                await asyncio.shield(self._abort_merge(repo))
                raise
            if code != 0:
                # Leave the user's checkout exactly as it was found.
                await self._abort_merge(repo)
                raise MergeConflict(task.id, branch, target, (err or out).strip())

            await self.remove_worktree(task)
            return (await self._git(repo, "rev-parse", "HEAD")).strip()

    async def _abort_merge(self, repo: Path) -> None:
        code, _, _ = await self._git_status(repo, "rev-parse", "-q", "--verify", "MERGE_HEAD")
        if code == 0:
            await self._git_status(repo, "merge", "--abort")

    async def merge_in_progress(self, project: str) -> str | None:
        """A merge left half-done in the project's own checkout, or None.

        Described rather than undone. Buddy's merges only ever start from a
        clean checkout, so one left behind is almost always a Buddy merge
        that was killed mid-conflict - but "almost" is not good enough to
        abort in your checkout: you may already be resolving it by hand.
        """
        repo = self.config.project(project).path
        if not await self.is_git_repo(repo):
            return None
        code, _, _ = await self._git_status(repo, "rev-parse", "-q", "--verify", "MERGE_HEAD")
        if code != 0:
            return None
        git_dir = Path((await self._git(repo, "rev-parse", "--git-dir")).strip())
        message = (git_dir if git_dir.is_absolute() else repo / git_dir) / "MERGE_MSG"
        try:
            what = message.read_text().strip().splitlines()[0]
        except (OSError, IndexError):
            what = "an unfinished merge"
        return (
            f"{repo} is in the middle of {what!r}. Finish it (resolve, then `git commit`) "
            f"or undo it with `git -C {repo} merge --abort`."
        )

    # -- discarding and cleanup ------------------------------------

    async def discard(self, task: TaskSpec) -> None:
        """Remove the worktree, keep the branch for the grace period."""
        await self.remove_worktree(task)

    async def remove_worktree(self, task: TaskSpec) -> None:
        project = self.config.project(task.project)
        worktree = self.config.worktree_path(task.project, task.id)
        if not worktree.exists():
            return
        code, _, err = await self._git_status(
            project.path, "worktree", "remove", "--force", str(worktree)
        )
        if code != 0:
            raise WorkspaceError(f"could not remove worktree {worktree}: {err.strip()}")

    async def delete_branch(self, task: TaskSpec, *, force: bool = False) -> bool:
        project = self.config.project(task.project)
        code, _, _ = await self._git_status(
            project.path, "branch", "-D" if force else "-d", branch_name(task)
        )
        return code == 0

    async def prune_worktrees(self, project: str) -> None:
        """`git worktree prune`, run weekly."""
        await self._git(self.config.project(project).path, "worktree", "prune")

    async def prune_discarded(
        self, task: TaskSpec, discarded_at: datetime | None, *, now: datetime | None = None
    ) -> bool:
        """Delete a discarded task's branch once its grace is up.

        True only when a branch was actually deleted. Refused, returning
        False, when the grace has not passed, the branch is already gone, the
        project is not git - or the branch has moved since it was discarded.
        Discarding is a decision about the *task*; if anyone checked the
        branch out and kept working, its reflog records that after the
        discard, and the branch stays theirs no matter how long ago that was.
        """
        if not self.is_past_grace(discarded_at, now=now):
            return False
        repo = self.config.project(task.project).path
        if not await self.is_git_repo(repo):
            return False
        ref = f"refs/heads/{branch_name(task)}"
        code, _, _ = await self._git_status(repo, "rev-parse", "-q", "--verify", ref)
        if code != 0:
            return False
        # When the ref last *moved*, from its reflog entry - not the commit's
        # own date, which pointing a branch back at an old commit leaves old.
        code, out, _ = await self._git_status(
            repo, "log", "-g", "-1", "--format=%gd", "--date=unix", ref
        )
        stamp = re.search(r"@\{(\d+)\}", out)
        if code == 0 and stamp and discarded_at is not None:
            moved_at = datetime.fromtimestamp(int(stamp.group(1)), tz=discarded_at.tzinfo or UTC)
            if moved_at > discarded_at:
                return False
        return await self.delete_branch(task, force=True)

    def is_past_grace(self, discarded_at: datetime | None, *, now: datetime | None = None) -> bool:
        """Whether a discarded task's branch may be deleted."""
        if discarded_at is None:
            return False
        grace: timedelta = self.config.buddy.discard_grace
        return (now or utcnow()) - discarded_at >= grace
