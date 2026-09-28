"""Git worktrees, checkpoints, merge and conflict, against a real repo."""

from __future__ import annotations

import asyncio
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from buddy.config import Config
from buddy.models import RunOutcome, TaskRun, TaskSpec, utcnow
from buddy.workspace import (
    MergeConflict,
    SecretsInBranch,
    TamperedWorktree,
    Workspace,
    WorkspaceError,
    branch_name,
    slugify,
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real repo on `main` with one commit."""
    path = tmp_path / "code" / "webapp"
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.name", "Test")
    git(path, "config", "user.email", "test@example.com")
    (path / "README.md").write_text("hello\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")
    return path


@pytest.fixture
def config(tmp_path: Path, repo: Path) -> Config:
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{repo}"\nbase_branch = "main"\n'
        f'[projects.notes]\npath = "{tmp_path / "notes"}"\n'
    )
    (tmp_path / "notes").mkdir()
    return Config.load(home=tmp_path)


@pytest.fixture
def workspace(config: Config) -> Workspace:
    return Workspace(config)


def make_task(**overrides) -> TaskSpec:
    defaults = {
        "id": "t-0142",
        "title": "Fix onboarding flow",
        "brief": "# Task\n...",
        "harness": "claude_code",
        "project": "webapp",
        "priority": 2,
    }
    return TaskSpec(**(defaults | overrides))


def make_run(checkout, task: TaskSpec, attempt: int = 1) -> TaskRun:
    return TaskRun(
        task_id=task.id,
        attempt=attempt,
        slot="Tuesday",
        worktree=checkout.path,
        branch=checkout.branch or "",
        base_ref=checkout.base_ref,
        log_path=Path("/tmp/unused.log"),
    )


# -- naming ---------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Fix onboarding flow", "fix-onboarding-flow"),
        ("Add retry logic (HTTP 429s)", "add-retry-logic-http-429s"),
        ("  ...  ", "task"),
        ("Refactor payments/", "refactor-payments"),
    ],
)
def test_slugify_is_branch_safe(title: str, expected: str):
    assert slugify(title) == expected


def test_slug_is_bounded():
    assert len(slugify("word " * 60)) <= 40


def test_branch_name_carries_the_task_id():
    assert branch_name(make_task()) == "buddy/t-0142-fix-onboarding-flow"


# -- creating worktrees --------------------------------------------


async def test_create_makes_a_worktree_on_its_own_branch(workspace: Workspace, repo: Path):
    task = make_task()
    checkout = await workspace.create(task)

    assert checkout.is_worktree
    assert checkout.path.is_dir()
    assert (checkout.path / "README.md").exists()
    assert checkout.branch == "buddy/t-0142-fix-onboarding-flow"
    assert await workspace.current_branch(checkout.path) == checkout.branch
    # Your checkout and your base branch are untouched.
    assert await workspace.current_branch(repo) == "main"


async def test_seven_tasks_on_one_repo_do_not_collide(workspace: Workspace):
    checkouts = [
        await workspace.create(make_task(id=f"t-000{i}", title=f"task {i}")) for i in range(1, 8)
    ]
    assert len({c.path for c in checkouts}) == 7
    assert len({c.branch for c in checkouts}) == 7
    for checkout in checkouts:
        (checkout.path / "file.txt").write_text(checkout.branch or "")
    assert all((c.path / "file.txt").read_text() == c.branch for c in checkouts)


async def test_create_is_idempotent_so_a_requeued_attempt_resumes(workspace: Workspace):
    """A retry gets the same worktree and branch, with the partial work still
    in it."""
    task = make_task()
    first = await workspace.create(task)
    (first.path / "partial.txt").write_text("half-done\n")

    second = await workspace.create(task)

    assert second.path == first.path
    assert second.branch == first.branch
    assert (second.path / "partial.txt").read_text() == "half-done\n"


async def test_a_non_git_project_runs_in_place_and_takes_the_slot_lock(
    workspace: Workspace, tmp_path: Path
):
    checkout = await workspace.create(make_task(project="notes"))
    assert not checkout.is_worktree
    assert checkout.path == tmp_path / "notes"
    assert checkout.branch is None
    assert await workspace.requires_exclusive_slot("notes") is True
    assert await workspace.requires_exclusive_slot("webapp") is False


async def test_a_missing_project_path_is_reported(config: Config, tmp_path: Path):
    (tmp_path / "config.toml").write_text(f'[projects.gone]\npath = "{tmp_path / "nowhere"}"\n')
    workspace = Workspace(Config.load(home=tmp_path))
    with pytest.raises(WorkspaceError, match="does not exist"):
        await workspace.create(make_task(project="gone"))


# -- checkpointing -------------------------------------------------


async def test_checkpoint_commits_uncommitted_work(workspace: Workspace):
    task = make_task()
    checkout = await workspace.create(task)
    (checkout.path / "new.txt").write_text("agent work\n")

    sha = await workspace.checkpoint(make_run(checkout, task), RunOutcome.PREEMPTED)

    assert sha
    assert await workspace.is_clean(checkout.path)
    message = git(checkout.path, "log", "-1", "--pretty=%s")
    assert message.strip() == "WIP (buddy, preempted, attempt 1)"
    assert "Buddy" in git(checkout.path, "log", "-1", "--pretty=%an")


async def test_checkpoint_is_a_no_op_when_the_harness_committed_its_own_work(
    workspace: Workspace,
):
    """The happy path: the working rules tell the harness to commit as it goes."""
    task = make_task()
    checkout = await workspace.create(task)
    (checkout.path / "done.txt").write_text("finished\n")
    git(checkout.path, "add", "-A")
    git(checkout.path, "-c", "user.name=A", "-c", "user.email=a@b.c", "commit", "-q", "-m", "real")

    assert await workspace.checkpoint(make_run(checkout, task), RunOutcome.DONE) is None
    assert git(checkout.path, "log", "-1", "--pretty=%s").strip() == "real"


async def test_checkpoint_records_the_attempt_and_outcome(workspace: Workspace):
    task = make_task()
    checkout = await workspace.create(task)
    (checkout.path / "x").write_text("1")
    await workspace.checkpoint(make_run(checkout, task, attempt=3), RunOutcome.TIMEOUT)
    assert "attempt 3" in git(checkout.path, "log", "-1", "--pretty=%s")
    assert "timeout" in git(checkout.path, "log", "-1", "--pretty=%s")


async def test_checkpoint_tolerates_a_worktree_that_is_gone(workspace: Workspace, tmp_path: Path):
    task = make_task()
    checkout = await workspace.create(task)
    run = make_run(checkout, task)
    await workspace.remove_worktree(task)
    assert await workspace.checkpoint(run, RunOutcome.INTERRUPTED) is None


# -- diff -----------------------------------------------------------


async def test_diff_shows_only_this_branch_s_work(workspace: Workspace, repo: Path):
    task = make_task()
    checkout = await workspace.create(task)
    (checkout.path / "added.txt").write_text("agent\n")
    git(checkout.path, "add", "-A")
    git(checkout.path, "-c", "user.name=A", "-c", "user.email=a@b.c", "commit", "-q", "-m", "work")

    # Unrelated work lands on main meanwhile; a three-dot diff must ignore it.
    (repo / "unrelated.txt").write_text("yours\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "your own commit")

    stat = await workspace.diff_stat(task)
    assert "added.txt" in stat
    assert "unrelated.txt" not in stat
    assert "agent" in await workspace.diff(task)


# -- merging --------------------------------------------------------


async def _commit_in(workspace: Workspace, task: TaskSpec, name: str, text: str):
    checkout = await workspace.create(task)
    (checkout.path / name).write_text(text)
    git(checkout.path, "add", "-A")
    git(checkout.path, "-c", "user.name=A", "-c", "user.email=a@b.c", "commit", "-q", "-m", "work")
    return checkout


async def test_merge_lands_the_work_and_removes_the_worktree(workspace: Workspace, repo: Path):
    task = make_task()
    checkout = await _commit_in(workspace, task, "feature.txt", "shipped\n")

    sha = await workspace.merge(task)

    assert sha
    assert (repo / "feature.txt").read_text() == "shipped\n"
    assert not checkout.path.exists()
    # --no-ff, so the merge is visible in history.
    assert git(repo, "log", "-1", "--pretty=%P").split().__len__() == 2


async def test_merge_refuses_when_your_checkout_is_dirty(workspace: Workspace, repo: Path):
    task = make_task()
    await _commit_in(workspace, task, "feature.txt", "shipped\n")
    (repo / "scratch.txt").write_text("my own work in progress\n")

    with pytest.raises(WorkspaceError, match="uncommitted changes"):
        await workspace.merge(task)
    assert not (repo / "feature.txt").exists()


async def test_merge_refuses_when_you_are_on_another_branch(workspace: Workspace, repo: Path):
    task = make_task()
    await _commit_in(workspace, task, "feature.txt", "shipped\n")
    git(repo, "checkout", "-q", "-b", "some-experiment")

    with pytest.raises(WorkspaceError, match="is on 'some-experiment'"):
        await workspace.merge(task)


async def test_a_conflict_is_reported_and_leaves_your_checkout_untouched(
    workspace: Workspace, repo: Path
):
    """A conflict is never resolved automatically."""
    task = make_task()
    await _commit_in(workspace, task, "README.md", "agent version\n")
    (repo / "README.md").write_text("your version\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "your change")

    with pytest.raises(MergeConflict) as excinfo:
        await workspace.merge(task)

    assert excinfo.value.branch == branch_name(task)
    assert (repo / "README.md").read_text() == "your version\n"
    assert await workspace.is_clean(repo)


async def _conflicting(workspace: Workspace, repo: Path) -> TaskSpec:
    task = make_task()
    await _commit_in(workspace, task, "README.md", "agent version\n")
    (repo / "README.md").write_text("your version\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "your change")
    return task


async def test_a_merge_cancelled_mid_conflict_still_leaves_your_checkout_clean(
    workspace: Workspace, repo: Path, monkeypatch
):
    """The abort was not in a `finally`. Ctrl-C landing as `git merge`
    returned left your own checkout mid-conflict, MERGE_HEAD and all."""
    import asyncio

    task = await _conflicting(workspace, repo)
    real = workspace._git_status

    async def interrupted(cwd, *args, **kwargs):
        result = await real(cwd, *args, **kwargs)
        if args[:2] == ("merge", "--no-ff"):
            raise asyncio.CancelledError  # the user hit Ctrl-C just then
        return result

    monkeypatch.setattr(workspace, "_git_status", interrupted)
    with pytest.raises(asyncio.CancelledError):
        await workspace.merge(task)

    assert not (repo / ".git" / "MERGE_HEAD").exists()
    assert (repo / "README.md").read_text() == "your version\n"
    assert await workspace.is_clean(repo)


async def test_a_merge_left_in_progress_is_found_and_never_merged_over(
    workspace: Workspace, repo: Path
):
    """A SIGKILL runs no `finally`, so the half-done merge has to be *found*:
    named, with its fix, and refused as the base for another merge. Never
    aborted for you - you may already be resolving it."""
    import asyncio

    task = await _conflicting(workspace, repo)
    # A merge git started and nobody finished: what a SIGKILL leaves.
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(repo),
        "merge",
        "--no-ff",
        branch_name(task),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert await proc.wait() != 0 and (repo / ".git" / "MERGE_HEAD").exists()

    found = await workspace.merge_in_progress("webapp")
    assert found is not None and f"git -C {repo} merge --abort" in found

    other = make_task(id="t-0002", title="other")
    await _commit_in(workspace, other, "other.txt", "x\n")
    with pytest.raises(WorkspaceError, match="merge already in progress"):
        await workspace.merge(other)
    assert (repo / ".git" / "MERGE_HEAD").exists(), "Buddy must not abort a merge it did not start"


async def test_merges_into_one_project_are_serialized(workspace: Workspace, repo: Path):
    first = make_task(id="t-0001", title="first")
    second = make_task(id="t-0002", title="second")
    await _commit_in(workspace, first, "one.txt", "1\n")
    await _commit_in(workspace, second, "two.txt", "2\n")

    await asyncio.gather(workspace.merge(first), workspace.merge(second))

    assert (repo / "one.txt").exists()
    assert (repo / "two.txt").exists()


# -- discard and cleanup -------------------------------------------


async def test_discard_removes_the_worktree_but_keeps_the_branch(workspace: Workspace, repo: Path):
    task = make_task()
    checkout = await _commit_in(workspace, task, "wip.txt", "unfinished\n")

    await workspace.discard(task)

    assert not checkout.path.exists()
    assert branch_name(task) in git(repo, "branch", "--list", branch_name(task))


async def test_branch_deletion_and_pruning(workspace: Workspace, repo: Path):
    task = make_task()
    await _commit_in(workspace, task, "wip.txt", "unfinished\n")
    await workspace.discard(task)

    assert await workspace.delete_branch(task, force=True) is True
    assert git(repo, "branch", "--list", branch_name(task)).strip() == ""
    await workspace.prune_worktrees("webapp")


def test_grace_period_gates_branch_deletion(workspace: Workspace):
    now = utcnow()
    assert workspace.is_past_grace(None) is False
    assert workspace.is_past_grace(now - timedelta(days=3), now=now) is False
    assert workspace.is_past_grace(now - timedelta(days=8), now=now) is True


# -- the discard sweep -----------------------------------------------------


async def test_a_discarded_branch_past_its_grace_is_deleted(workspace: Workspace, repo: Path):
    task = make_task()
    await _commit_in(workspace, task, "idea.txt", "abandoned\n")
    await workspace.discard(task)
    discarded = utcnow()

    early = await workspace.prune_discarded(task, discarded, now=discarded + timedelta(days=3))
    assert early is False
    assert git(repo, "branch", "--list", branch_name(task)).strip()

    late = await workspace.prune_discarded(task, discarded, now=discarded + timedelta(days=8))
    assert late is True
    assert git(repo, "branch", "--list", branch_name(task)).strip() == ""
    # Idempotent: a branch that is already gone is not an error.
    again = await workspace.prune_discarded(task, discarded, now=discarded + timedelta(days=9))
    assert again is False


async def test_a_discarded_branch_someone_kept_working_on_is_never_deleted(
    workspace: Workspace, repo: Path
):
    """Discarding is a decision about the task, not a lock on the branch. If
    anyone moved the branch after the discard, its reflog says so, and seven
    days later it is still theirs."""
    import asyncio

    task = make_task()
    await _commit_in(workspace, task, "idea.txt", "abandoned\n")
    await workspace.discard(task)
    discarded = utcnow()
    await asyncio.sleep(1.1)  # reflog timestamps are whole seconds

    git(repo, "checkout", "-q", branch_name(task))
    (repo / "idea.txt").write_text("rescued it after all\n")
    git(repo, "commit", "-qam", "picking this back up")
    git(repo, "checkout", "-q", "main")

    kept = await workspace.prune_discarded(task, discarded, now=discarded + timedelta(days=30))
    assert kept is False
    assert git(repo, "branch", "--list", branch_name(task)).strip()


async def test_the_sweep_ignores_a_project_that_is_not_git(workspace: Workspace):
    task = make_task(project="notes")
    old = utcnow() - timedelta(days=30)
    assert await workspace.prune_discarded(task, old) is False


# -- secrets stay out of git ---------------------------------------------------


def _fake_key(seed: int) -> str:
    import random
    import string

    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits
    return "sk-" + "proj-" + "".join(rng.choice(alphabet) for _ in range(64))


async def test_the_checkpoint_never_commits_a_secret(config: Config, repo: Path):
    """Buddy's own commit. An agent is handed your key in its environment; if
    it writes it into a `.env` or a config file and stops, `git add -A`
    committed it onto the branch - Buddy putting your key into git itself."""
    mine = "".join(reversed("0f1e2d3c4b5a69788796a5b4c3d2e1f0"))
    workspace = Workspace(config, known_secrets=lambda: {"FISH_API_KEY": mine})
    task = make_task()
    checkout = await workspace.create(task)
    (checkout.path / "feature.py").write_text("print('real work')\n")
    (checkout.path / ".env").write_text("DEBUG=1\n")
    (checkout.path / "settings.py").write_text(f'VOICE = "{mine}"\n')
    (checkout.path / "client.py").write_text(f'OPENAI = "{_fake_key(1)}"\n')

    sha = await workspace.checkpoint(make_run(checkout, task), RunOutcome.INTERRUPTED)

    committed = git(checkout.path, "show", "--name-only", "--format=", sha).split()
    assert committed == ["feature.py"]
    withheld = workspace.withheld[(task.id, 1)]
    assert sorted(w.split(":")[0] for w in withheld) == [".env", "client.py", "settings.py"]
    assert all(mine not in w for w in withheld), "a finding never repeats the secret"
    # Still there, still yours, just not in git.
    assert (checkout.path / ".env").exists() and (checkout.path / "settings.py").exists()


async def test_a_checkpoint_with_nothing_but_secrets_commits_nothing(config: Config, repo: Path):
    workspace = Workspace(config, known_secrets=dict)
    task = make_task()
    checkout = await workspace.create(task)
    (checkout.path / ".env.local").write_text("TOKEN=x\n")
    assert await workspace.checkpoint(make_run(checkout, task)) is None
    assert git(checkout.path, "log", "--oneline", "main..HEAD").strip() == ""


async def test_merge_refuses_a_branch_that_adds_a_secret(config: Config, repo: Path):
    """The agent's own commits are not Buddy's to police - but nothing reaches
    your branch without `buddy merge`, so that is where it is caught."""
    mine = "".join(reversed("0f1e2d3c4b5a69788796a5b4c3d2e1f0"))
    workspace = Workspace(config, known_secrets=lambda: {"FISH_API_KEY": mine})

    lookalike = make_task(id="t-0001", title="lookalike")
    await _commit_in(workspace, lookalike, "client.py", f'KEY = "{_fake_key(2)}"\n')
    with pytest.raises(SecretsInBranch, match="client.py:1: openai key") as refused:
        await workspace.merge(lookalike)
    assert _fake_key(2) not in str(refused.value)
    assert not (repo / "client.py").exists()
    # A lookalike you have checked can be let through...
    await workspace.merge(lookalike, allow_secret_patterns=True)
    assert (repo / "client.py").exists()

    yours = make_task(id="t-0002", title="yours")
    await _commit_in(workspace, yours, "voice.toml", f'token = "{mine}"\n')
    # ...but your own key, never.
    with pytest.raises(SecretsInBranch, match="your FISH_API_KEY"):
        await workspace.merge(yours, allow_secret_patterns=True)

    env = make_task(id="t-0003", title="env")
    await _commit_in(workspace, env, ".env", "DEBUG=1\n")
    with pytest.raises(SecretsInBranch, match=r"\.env"):
        await workspace.merge(env)


async def test_the_merge_scan_reads_paths_whatever_your_git_config_says(config: Config, repo: Path):
    """The scan parses `git diff`'s paths; a user's own diff settings must not
    change what it reports."""
    git(repo, "config", "diff.noprefix", "true")
    git(repo, "config", "diff.mnemonicPrefix", "true")
    workspace = Workspace(config, known_secrets=dict)
    task = make_task(id="t-0001", title="leaky")
    await _commit_in(workspace, task, "client.py", f'KEY = "{_fake_key(4)}"\n')
    findings = await workspace.secrets_in_branch(task)
    assert [f.describe() for f in findings] == ["client.py:1: openai key"]


# -- git on the host, in a directory the agent controlled -------------------


def _plant(where: Path, marker: Path) -> None:
    """What an agent can leave in a git directory it controls: config that
    makes the *next* git to read it run a command."""
    git(where, "config", "core.fsmonitor", f"echo fsmonitor >> {marker}; false")
    hooks = where / "planted-hooks"
    hooks.mkdir()
    (hooks / "post-commit").write_text(f"#!/bin/sh\necho hook >> {marker}\n")
    (hooks / "post-commit").chmod(0o755)
    git(where, "config", "core.hooksPath", str(hooks))


async def test_a_checkpoint_never_runs_what_the_agent_pointed_git_at(
    workspace: Workspace, repo: Path, tmp_path: Path
):
    """Reproduced first: a sandboxed agent can only write its worktree, and the
    worktree's `.git` is a file in it. Pointed at a git directory of the
    agent's own, Buddy's end-of-run checkpoint - run on the host - executed
    the agent's `core.fsmonitor` fifty-odd times and its post-commit hook once.
    """
    task = make_task()
    checkout = await workspace.create(task)
    marker = tmp_path / "ran-on-the-host"
    (checkout.path / ".git").unlink()
    git(checkout.path, "init", "-q", "--separate-git-dir", str(checkout.path / ".own"), ".")
    _plant(checkout.path, marker)
    (checkout.path / "work.txt").write_text("left uncommitted\n")
    run = make_run(checkout, task)

    assert await workspace.checkpoint(run, RunOutcome.DONE) is None

    assert not marker.exists(), marker.read_text()
    [said] = workspace.withheld[(task.id, 1)]
    assert "Buddy ran no git there" in said
    assert git(repo, "log", "-1", "--format=%s", branch_name(task)).strip() == "initial"


async def test_a_checkpoint_refuses_a_rewritten_commondir(
    workspace: Workspace, repo: Path, tmp_path: Path
):
    """The quieter route to the same place: leave `.git` alone and rewrite the
    `commondir` file it leads to, which sits in the shared git directory the
    sandbox mounts read-write. Measured: pinning `GIT_COMMON_DIR` kept its
    config and hooks from running, but git still wrote the checkpoint's ref
    wherever the file pointed. So the file has to lead back, or no git runs."""
    task = make_task()
    checkout = await workspace.create(task)
    marker = tmp_path / "ran-on-the-host"
    elsewhere = tmp_path / "elsewhere"
    git(tmp_path, "init", "-q", "--bare", str(elsewhere))
    _plant(elsewhere, marker)
    admin = Path((checkout.path / ".git").read_text().removeprefix("gitdir:").strip())
    (admin / "commondir").write_text(f"{elsewhere}\n")
    (checkout.path / "work.txt").write_text("left uncommitted\n")

    assert await workspace.checkpoint(make_run(checkout, task), RunOutcome.DONE) is None

    assert not marker.exists(), marker.read_text()
    assert "commondir points outside" in workspace.withheld[(task.id, 1)][0]
    assert not (elsewhere / "refs" / "heads" / "buddy").exists()


async def test_a_retry_will_not_start_in_a_tampered_worktree(workspace: Workspace, tmp_path: Path):
    task = make_task()
    checkout = await workspace.create(task)
    (checkout.path / ".git").write_text(f"gitdir: {tmp_path}\n")

    with pytest.raises(TamperedWorktree):
        await workspace.create(task)


def _relative(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target, start)


async def test_a_relative_gitdir_is_still_this_worktrees(workspace: Workspace, repo: Path):
    """git 2.48's `worktree.useRelativePaths` writes the pointer relative.
    That is git's own choice, not tampering, and still checkpoints."""
    task = make_task()
    checkout = await workspace.create(task)
    admin = Path((checkout.path / ".git").read_text().removeprefix("gitdir:").strip())
    (checkout.path / ".git").write_text(f"gitdir: {_relative(admin, checkout.path)}\n")
    (checkout.path / "work.txt").write_text("left uncommitted\n")

    sha = await workspace.checkpoint(make_run(checkout, task), RunOutcome.DONE)

    assert sha is not None
    assert git(repo, "rev-parse", branch_name(task)).strip() == sha


async def test_starting_a_task_never_contacts_the_remote(
    workspace: Workspace, repo: Path, tmp_path: Path
):
    """A fetch used to run first, and changed nothing - the branch is cut from
    the local base - while a remote that wanted credentials prompted on
    Buddy's own terminal and held the scheduler until someone answered."""
    marker = tmp_path / "contacted"
    upload = tmp_path / "upload-pack"
    upload.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    upload.chmod(0o755)
    git(repo, "remote", "add", "origin", str(tmp_path / "nowhere"))
    git(repo, "config", "remote.origin.uploadpack", str(upload))

    await workspace.create(make_task())

    assert not marker.exists()


async def test_pointing_a_discarded_branch_back_still_counts_as_moving_it(
    workspace: Workspace, repo: Path
):
    """The reflog's own time decides, not the commit's: pointing a branch at
    an older commit after the discard is a move made after the discard."""
    import asyncio

    task = make_task()
    await _commit_in(workspace, task, "idea.txt", "abandoned\n")
    await workspace.discard(task)
    discarded = utcnow()
    await asyncio.sleep(1.1)  # reflog timestamps are whole seconds

    git(repo, "branch", "-f", branch_name(task), "main")

    kept = await workspace.prune_discarded(task, discarded, now=discarded + timedelta(days=30))
    assert kept is False
    assert git(repo, "branch", "--list", branch_name(task)).strip()


async def test_merging_a_task_with_no_branch_says_so(workspace: Workspace):
    with pytest.raises(WorkspaceError, match="no branch"):
        await workspace.merge(make_task())
