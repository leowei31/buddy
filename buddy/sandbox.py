"""The second isolation boundary: running a harness inside a container.

Separate from the manager because it is not the scheduler's business alone.
An adapter's preflight has to ask its questions - is the CLI here, what
version, is it signed in - *where the harness will actually run*, and for a
sandboxed harness that is inside the image, not on this machine.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from pathlib import Path

from buddy.config import HarnessConfig, fill_template

#: A `docker run` that actually works, for `[harness.<name>.sandbox_command]`.
#:
#: "A docker run that mounts only the worktree" is the obvious shape, and it
#: cannot work, because the brief
#: lives in `~/.buddy/tasks/<id>/prompt.md`, outside the worktree, and every
#: harness invocation reads it. A container with only the worktree mounted
#: fails on the first redirect. So the prompt is mounted too, read-only, and
#: it is the only other thing that is.
#:
#: No `-i`. A harness runs in a tmux pane, so its stdin is that pane's
#: terminal, and `docker run -i` forwards it into the container as a *pipe*.
#: A CLI that reads stdin when it is not a terminal - which is the usual way
#: to accept piped input - then waits for an end that a terminal never sends.
#: Measured: identical runs differing only in stdin, OpenCode finished in 7s
#: with stdin closed and hung indefinitely with the pane's terminal, printing
#: nothing at all. Nothing needs the container's stdin: every harness gets its
#: brief from the mounted `prompt.md`, and a redirect in the command line is
#: performed by the shell *inside* the container, reading that file.
#:
#: `--network host` is deliberate rather than lazy: the agent has to reach
#: its own API, and the boundary this draws is the filesystem one.
#:
#: `{env}` forwards the names in `[harness.<name>.env]`, and nothing else.
#: `run.sh` exports them on the host, and `docker run` inherits nothing, so
#: without this a key configured there simply was not in the container - the
#: documented way to give a sandboxed harness its credentials did not work.
#:
#: `{git}` is what makes the worktree a *working* worktree inside the
#: container. A worktree's `.git` is a file reading `gitdir: <repo>/.git/
#: worktrees/<id>`, which is outside the mount, so with only the worktree
#: mounted every git command failed with "not a git repository" - while the
#: working rules Buddy appends to every brief tell the agent to commit as it
#: goes, and a retry's rules tell it to run `git log` first. Measured: a real
#: sandboxed task through the CLI, where the agent's own commit never
#: happened and only Buddy's end-of-run checkpoint survived.
#:
#: `--entrypoint sh` with `-c {command}`, and `{command}` substituted as one
#: shell-quoted argument, because a harness command is a shell line: flags, a
#: redirect, `$(cat ...)`. Passed as bare words to `docker run` there was no
#: shell in the container to read them. The entrypoint is overridden rather
#: than assumed absent: an image whose own entrypoint is a shell swallowed
#: everything after the first word, so `claude -p --output-format ...` ran as
#: bare `claude` - measured against exactly such an image.
DEFAULT_SANDBOX_COMMAND = (
    "docker run --rm"
    " -v {worktree}:{worktree}"
    " -v {prompt_path}:{prompt_path}:ro"
    " -w {worktree}"
    " --network host"
    "{git}"
    "{env}"
    " --entrypoint sh"
    " {image} -c {command}"
)

#: What `setup/assets/Dockerfile.harness` builds.
DEFAULT_SANDBOX_IMAGE = "buddy-harness:latest"

#: Read-only inside the container, and the reason is a sandbox escape rather
#: than tidiness. `.git/config` can name `core.hooksPath`, and `.git/hooks`
#: holds scripts git runs; both are read by *your* git, on the host, the next
#: time you use that repository. Measured: an agent in the sandbox wrote
#: `hooksPath = /tmp/mine` into the host's `.git/config` and it stayed there.
#: Neither is written by committing, so nothing legitimate is lost.
GIT_READ_ONLY = ("config", "hooks")

#: Identity for the agent's own commits. A container has no `~/.gitconfig`,
#: so git refused every commit with "Please tell me who you are" - measured.
#: Taken from the repository itself (`workspace.git_identity`), so a
#: sandboxed commit is attributed exactly as an unsandboxed one, and nothing
#: but these two values crosses.
GIT_IDENTITY = (
    ("user.name", ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME")),
    ("user.email", ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL")),
)


def common_git_dir(checkout: Path) -> Path | None:
    """The `.git` directory `checkout` really uses, or None if it is not a
    checkout at all.

    Read from the files rather than asked of git: this runs while a run is
    being prepared, and a subprocess per attempt to learn a path that is
    written down is not worth it.

    Only ever asked of a checkout the agent cannot write - the project's own,
    never the task's worktree. A worktree's `.git` is a file *inside* the
    mount, so an agent can point it anywhere, and whatever it names here is
    what the next attempt's container is given, read-write. Measured: one
    line in `.git` and the retry mounted `~/.ssh`.
    """
    marker = checkout / ".git"
    if marker.is_dir():
        return marker
    try:
        text = marker.read_text()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text.split(":", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (checkout / gitdir).resolve()
    # A linked worktree's gitdir holds `commondir`, pointing at the `.git`
    # that owns the objects and refs - which is what a commit writes to.
    try:
        common = (gitdir / "commondir").read_text().strip()
    except OSError:
        return gitdir
    return (gitdir / common).resolve() if common else gitdir


def git_support(
    repo: Path | None,
    worktree: Path | None = None,
    identity: Mapping[str, str] | None = None,
) -> str:
    """The `docker run` flags that make git work inside the container.

    `repo` is the project's own checkout, which is where the git directory
    is learned from - never the worktree, which the agent controls. Empty
    when there is none, or it is not a checkout: a project that is not a git
    repository runs in place and has no branch to commit to.

    `identity` is the repository's `user.name` and `user.email`, as
    `workspace.git_identity` found them.
    """
    if repo is None or (common := common_git_dir(repo)) is None:
        return ""
    flags = []
    # Already inside the worktree mount when that is where `.git` lives.
    if worktree is None or not common.is_relative_to(worktree):
        flags.append(f" -v {shlex.quote(str(common))}:{shlex.quote(str(common))}")
    for name in GIT_READ_ONLY:
        path = common / name
        if path.exists():
            quoted = shlex.quote(str(path))
            flags.append(f" -v {quoted}:{quoted}:ro")
    variables = {
        name: value
        for key, names in GIT_IDENTITY
        if (value := (identity or {}).get(key))
        for name in names
    }
    for name, value in sorted(variables.items()):
        flags.append(f" -e {shlex.quote(f'{name}={value}')}")
    return "".join(flags)


def wrap_in_sandbox(
    harness: HarnessConfig,
    command: str,
    worktree: Path,
    *,
    prompt_path: Path | None = None,
    repo: Path | None = None,
    git_identity: Mapping[str, str] | None = None,
) -> str:
    """The recommended second isolation boundary, off by default.

    The worktree and the branch are the *required* boundary and are always
    there; this is the one you turn on once auto-approve is on, so that an
    agent that decides to `rm -rf` something can only reach what was mounted.

    `repo` is the project's own checkout, for the git directory the worktree
    belongs to, and `git_identity` who commits there. None for a probe,
    which has no repository.
    """
    if harness.sandbox != "docker" or not harness.sandbox_command:
        return command
    # `-e NAME` with no value: docker takes it from the environment `run.sh`
    # exported, so the secret is never on a command line.
    env_flags = "".join(f" -e {shlex.quote(name)}" for name in sorted(harness.env))
    return fill_template(
        harness.sandbox_command,
        worktree=shlex.quote(str(worktree)),
        git=git_support(repo, worktree, git_identity),
        env=env_flags,
        # One argument, so the shell inside the container sees the whole line.
        command=shlex.quote(command),
        prompt_path=shlex.quote(str(prompt_path)) if prompt_path is not None else "",
        image=DEFAULT_SANDBOX_IMAGE,
    )


def sandboxed(harness: HarnessConfig) -> bool:
    """Whether runs of this harness go through a container."""
    return harness.sandbox == "docker" and bool(harness.sandbox_command)
