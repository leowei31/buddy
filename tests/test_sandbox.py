"""The container boundary, against a real Docker daemon.

The claim is specific and checkable: with auto-approve on, the worktree and
branch are the *required* boundary and a container is the recommended second
one. The claim worth testing is not that `docker run` appears in a string -
`test_run_script.py` covers that - but that an agent inside the sandbox
genuinely cannot reach the base repository, the rest of `~/.buddy`, or the
home directory, and that what it writes in the worktree survives.

These run only where a Docker daemon is listening. That is a real gap and is
recorded as one: on a machine without Docker the boundary is asserted by
construction and never by observation.
"""

from __future__ import annotations

import asyncio
import os
import pty
import shutil
import subprocess
from pathlib import Path

import pytest

from buddy.config import HarnessConfig
from buddy.harnesses.base import BaseAdapter
from buddy.sandbox import DEFAULT_SANDBOX_COMMAND, wrap_in_sandbox
from buddy.workspace import git_identity

#: Small, ubiquitous, and enough to prove a mount boundary. Any image with a
#: shell will do, so `BUDDY_TEST_IMAGE` can name one already on the machine -
#: which is what an air-gapped or mirror-only Docker needs.
IMAGE = os.environ.get("BUDDY_TEST_IMAGE", "alpine:3")


def _docker(*args: str, timeout: int) -> bool:
    """Run a docker command, and treat "it never answered" as "no".

    A daemon that is installed but wedged makes `docker` hang rather than
    fail, and an unanswered probe is not a reason to error out of collection
    on a machine that was only ever going to skip these tests.
    """
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def docker_is_up() -> bool:
    if not shutil.which("docker"):
        return False
    return _docker("docker", "info", timeout=20)


needs_docker = pytest.mark.skipif(
    not docker_is_up(), reason="no Docker daemon; the sandbox boundary cannot be observed"
)


def image_is_available() -> bool:
    """Already here, or pullable. A daemon that cannot reach a registry is a
    reason to skip with a clear word, not to hang for five minutes."""
    if _docker("docker", "image", "inspect", IMAGE, timeout=30):
        return True
    return _docker("docker", "pull", "-q", IMAGE, timeout=120)


needs_image = pytest.mark.skipif(
    not (docker_is_up() and image_is_available()),
    reason=f"{IMAGE} is neither present nor pullable; set BUDDY_TEST_IMAGE to a local image",
)


def image_has_git() -> bool:
    """Most of this file needs only a shell, so the default image is a tiny
    one. The git tests need git, and `alpine:3` has none - which failed them
    for a reason that had nothing to do with what they check."""
    if not (docker_is_up() and image_is_available()):
        return False
    return _docker(
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "sh",
        IMAGE,
        "-c",
        "command -v git",
        timeout=120,
    )


needs_git_in_image = pytest.mark.skipif(
    not image_has_git(),
    reason=f"{IMAGE} has no git; set BUDDY_TEST_IMAGE to an image that has one",
)


@pytest.fixture(scope="module")
def image() -> str:
    return IMAGE


def sandboxed(command: str, worktree: Path, prompt: Path, image: str) -> str:
    harness = HarnessConfig.from_dict(
        "probe",
        {
            "command": "unused",
            "sandbox": "docker",
            "sandbox_command": DEFAULT_SANDBOX_COMMAND.replace("{image}", image),
        },
    )
    return wrap_in_sandbox(harness, command, worktree, prompt_path=prompt)


def run(command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=120)


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A base repo, a worktree beside it, and a brief outside both."""
    base = tmp_path / "code" / "webapp"
    base.mkdir(parents=True)
    (base / "SECRET.txt").write_text("the base branch nobody may touch\n")

    worktree = tmp_path / "worktrees" / "t-0001"
    worktree.mkdir(parents=True)
    (worktree / "README.md").write_text("work here\n")

    prompt = tmp_path / "tasks" / "t-0001" / "prompt.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("# Goal\nDo the thing.\n")

    # The rest of ~/.buddy, which this task has no business reading: another
    # task's brief and result, the database, the keys directory.
    other = tmp_path / "tasks" / "t-0002"
    other.mkdir(parents=True)
    (other / "prompt.md").write_text("ANOTHER TASKS BRIEF\n")
    (other / "result.json").write_text('{"summary": "ANOTHER TASKS RESULT"}\n')
    (tmp_path / "state.db").write_text("THE CONVERSATION\n")
    (tmp_path / "keys").mkdir()
    (tmp_path / "keys" / "service-account.json").write_text("A SERVICE ACCOUNT KEY\n")
    return base, worktree, prompt


@needs_image
def test_the_agent_can_read_its_brief(workspace, image):
    _, worktree, prompt = workspace
    result = run(sandboxed(f"cat {prompt}", worktree, prompt, image))
    assert result.returncode == 0, result.stderr
    assert "Do the thing." in result.stdout


@needs_image
def test_a_harness_that_takes_its_brief_on_stdin_gets_it(workspace, image):
    """Claude Code and Codex are given the brief by redirect - `... < {prompt_path}`
    - not as an argument. The redirect is performed by the shell *inside* the
    container, reading the mounted file, which is why removing `docker run -i`
    costs these two nothing: they never wanted the host's stdin."""
    _, worktree, prompt = workspace
    result = run(sandboxed(f"cat < {prompt}", worktree, prompt, image))
    assert result.returncode == 0, result.stderr
    assert "Do the thing." in result.stdout


@needs_image
def test_the_agent_can_work_in_its_worktree(workspace, image):
    _, worktree, prompt = workspace
    command = "echo written-inside > new.txt && cat README.md"
    result = run(sandboxed(command, worktree, prompt, image))
    assert result.returncode == 0, result.stderr
    assert "work here" in result.stdout
    # And the write survives, because the worktree is the one thing mounted rw.
    assert (worktree / "new.txt").read_text().strip() == "written-inside"


@needs_image
def test_the_command_runs_as_a_shell_command_inside_the_container(workspace, image):
    """A harness command is a shell line - flags, redirects, `$(cat ...)` -
    and the sandbox has to run it as one.

    It did not: the words went to `docker run` as argv, so nothing
    interpreted them. Against the image Buddy's own Dockerfile builds, whose
    entrypoint is a shell, only the first word survived: `claude -p --output-
    format ...` ran as bare `claude`, with its flags as positional parameters.
    """
    _, worktree, prompt = workspace
    command = "echo one two three > written.txt && echo done"
    result = run(sandboxed(command, worktree, prompt, image))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "done"
    assert (worktree / "written.txt").read_text().strip() == "one two three"


@needs_image
def test_the_agent_cannot_reach_the_base_repository(workspace, image):
    """The whole sandbox claim in one assertion.

    Auto-approve means the agent can run anything it likes. What stops that
    from mattering is that the base branch is not there to be reached.
    """
    base, worktree, prompt = workspace
    result = run(sandboxed(f"cat {base / 'SECRET.txt'}", worktree, prompt, image))
    assert result.returncode != 0
    assert "the base branch nobody may touch" not in result.stdout


@needs_image
def test_the_agent_cannot_reach_the_rest_of_buddys_state(workspace, image):
    """`~/.buddy` holds every other task's worktree, the database, and the
    keys directory. Only this task's brief is mounted, and read-only."""
    _, worktree, prompt = workspace
    buddy_home = prompt.parent.parent.parent

    # Docker creates the directories a file mount needs, so the path to this
    # task's brief exists inside. What matters is that it is all that does.
    reach = run(
        sandboxed(
            f"cat {buddy_home}/state.db {buddy_home}/keys/*.json "
            f"{buddy_home}/tasks/t-0002/prompt.md {buddy_home}/tasks/t-0002/result.json 2>&1",
            worktree,
            prompt,
            image,
        )
    )
    for secret in ("THE CONVERSATION", "A SERVICE ACCOUNT KEY", "ANOTHER TASKS"):
        assert secret not in reach.stdout, reach.stdout

    # This task's own brief is there, and is the only thing that is.
    listing = run(sandboxed(f"ls {buddy_home}/tasks/t-0001", worktree, prompt, image))
    assert listing.stdout.split() == ["prompt.md"]

    home = run(sandboxed("ls /root /home 2>&1 | head -20", worktree, prompt, image))
    assert "SECRET.txt" not in home.stdout


@needs_image
def test_the_brief_is_mounted_read_only(workspace, image):
    """A brief the agent can rewrite is a brief that cannot be trusted as the
    record of what was asked."""
    _, worktree, prompt = workspace
    result = run(sandboxed(f"echo tampered > {prompt}", worktree, prompt, image))
    assert result.returncode != 0
    assert prompt.read_text() == "# Goal\nDo the thing.\n"


@needs_image
def test_a_destructive_agent_reaches_nothing_it_should_not(workspace, image):
    """`rm -rf /` inside the sandbox, which is the scenario it exists for."""
    base, worktree, prompt = workspace
    run(sandboxed("rm -rf / --no-preserve-root 2>/dev/null; true", worktree, prompt, image))

    assert (base / "SECRET.txt").exists(), "the base repository was reachable"
    assert prompt.exists(), "the brief was reachable"
    # The worktree *is* mounted read-write, so its contents are forfeit - that
    # is the bargain, and it is why the worktree is disposable and the branch
    # is what carries the work.


def test_the_boundary_is_documented_where_someone_will_look_for_it():
    """The image is not built by setup, so the Dockerfile has to explain
    itself: Buddy does not install your harness CLIs."""
    dockerfile = Path(__file__).parent.parent / "buddy/setup/assets/Dockerfile.harness"
    text = dockerfile.read_text()
    assert "sandbox_command" in text
    assert "{prompt_path}" in text
    assert "HARNESS_INSTALL" in text, "the CLI is the user's to add"


@needs_image
def test_the_harness_env_reaches_the_container(workspace, image):
    """`[harness.<name>.env]` is how a sandboxed harness gets its key.

    `run.sh` exports it on the host and `docker run` inherits nothing, so
    without `{env}` the container had no key at all, however carefully it was
    configured - and the harness failed inside the sandbox for a reason
    nothing on the host could show.
    """
    _, worktree, prompt = workspace
    harness = HarnessConfig.from_dict(
        "probe",
        {
            "command": "unused",
            "sandbox": "docker",
            "sandbox_command": DEFAULT_SANDBOX_COMMAND.replace("{image}", image),
            "env": {"CODEX_API_KEY": "from-the-keychain", "OTHER": "x"},
        },
    )
    line = wrap_in_sandbox(harness, "echo $CODEX_API_KEY", worktree, prompt_path=prompt)

    assert "from-the-keychain" not in line, "the value belongs in the environment, not the argv"
    result = subprocess.run(
        ["bash", "-c", f"export CODEX_API_KEY=from-the-keychain OTHER=x; {line}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.stdout.strip() == "from-the-keychain", result.stderr


# -- preflight asks where the harness actually runs ---------------------------


class _Probe(BaseAdapter):
    """An adapter for a binary that exists in containers and not on this Mac."""

    name = "probe"
    binary = "sh"

    async def preflight(self):
        return await self.report((), [])

    def parse_result(self, log_text: str, exit_code: int):
        raise NotImplementedError


def _probe_config(image: str, *, sandbox: bool) -> HarnessConfig:
    return HarnessConfig.from_dict(
        "probe",
        {
            "command": "sh -c true",
            **(
                {
                    "sandbox": "docker",
                    "sandbox_command": DEFAULT_SANDBOX_COMMAND.replace("{image}", image),
                }
                if sandbox
                else {}
            ),
        },
    )


@needs_image
async def test_preflight_finds_a_binary_that_lives_only_in_the_image(image):
    """A sandboxed harness is installed *in the image* - that is the whole
    point, and what the documented `Dockerfile.harness` build does. Preflight
    used to run `shutil.which` on the host, so `buddy doctor` called a working
    harness "not on PATH" and `buddy spawn` refused every task given to it:
    the Docker sandbox, exactly as documented, could not run anything."""
    adapter = _Probe(_probe_config(image, sandbox=True))

    assert adapter.sandboxed
    located = await adapter.locate()
    assert located and image in located, located
    assert (await adapter.probe("-c", "echo hi"))[1].strip() == "hi"


@needs_image
async def test_a_binary_missing_from_the_image_is_reported_against_the_image(image):
    """And the message has to send its reader to the image, not to `brew`."""

    class Missing(_Probe):
        binary = "definitely-not-installed"

    report = await Missing(_probe_config(image, sandbox=True)).report((), [])

    assert not report.installed
    assert report.summary() == f"probe: not in {image}"
    assert "PATH" not in report.summary()


async def test_an_unsandboxed_harness_is_still_probed_on_this_machine():
    """The common case must not pay for the fix: no container, no docker."""
    adapter = _Probe(_probe_config("unused", sandbox=False))

    assert not adapter.sandboxed
    assert await adapter.locate() == shutil.which("sh")
    assert (await adapter.probe("-c", "echo hi"))[1].strip() == "hi"


@needs_image
def test_a_harness_that_reads_stdin_does_not_hang_on_the_panes_terminal(workspace, image):
    """The hang this caught, found by running a real task through the CLI.

    A harness runs in a tmux pane, so its stdin is that pane's terminal.
    `docker run -i` forwards that into the container as a *pipe*, which is how
    a CLI decides it is being piped to and should read stdin to its end - an
    end a terminal never sends. OpenCode printed nothing and ran forever;
    the same run with stdin closed finished in seven seconds.

    `cat` stands in for any such harness: with `-i` this test times out.
    """
    _, worktree, prompt = workspace
    harness = HarnessConfig.from_dict(
        "probe",
        {
            "command": "unused",
            "sandbox": "docker",
            "sandbox_command": DEFAULT_SANDBOX_COMMAND.replace("{image}", image),
        },
    )
    line = wrap_in_sandbox(harness, "cat > /dev/null; echo finished", worktree, prompt_path=prompt)

    # A real terminal, exactly as the pane supplies - and one that nobody is
    # ever going to type an end-of-file into.
    primary, secondary = pty.openpty()
    try:
        result = subprocess.run(
            ["bash", "-c", line],
            stdin=secondary,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        os.close(primary)
        os.close(secondary)

    assert result.stdout.strip() == "finished", result.stderr


def test_the_sandbox_command_never_attaches_stdin():
    """Stated as its own rule, because the failure it causes is silent: the
    run simply never produces a line, and only the stall timeout ends it."""
    assert " -i " not in DEFAULT_SANDBOX_COMMAND
    assert not DEFAULT_SANDBOX_COMMAND.startswith("docker run -i")


# -- git works inside the sandbox, and cannot reach back out ------------------


@pytest.fixture
def worktree_task(tmp_path: Path):
    """A real linked worktree, the way `Workspace.create` makes one: its
    `.git` is a *file* pointing at a gitdir outside the worktree."""
    repo = tmp_path / "webapp"
    repo.mkdir()
    run_git = lambda *a, cwd=repo: subprocess.run(  # noqa: E731
        ["git", "-c", "user.email=you@example.com", "-c", "user.name=You", *a],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    run_git("init", "-q", "-b", "main")
    # In the repository's own config, so what the agent commits as does not
    # depend on whoever is running the tests.
    run_git("config", "user.name", "You")
    run_git("config", "user.email", "you@example.com")
    (repo / "README.md").write_text("hello\n")
    run_git("add", "-A")
    run_git("commit", "-qm", "initial")
    worktree = tmp_path / "wt"
    run_git("worktree", "add", "-q", "-b", "buddy/t-0001-x", str(worktree))
    assert (worktree / ".git").is_file(), "the case under test is a linked worktree"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("do the thing\n")
    return repo, worktree, prompt


def _sandboxed(image: str) -> HarnessConfig:
    return HarnessConfig.from_dict(
        "probe",
        {
            "command": "unused",
            "sandbox": "docker",
            "sandbox_command": DEFAULT_SANDBOX_COMMAND.replace("{image}", image),
        },
    )


@needs_image
@needs_git_in_image
def test_the_agent_can_commit_on_its_branch_from_inside_the_sandbox(worktree_task, image):
    """The failure this caught, found by running a real task through the CLI.

    A worktree's `.git` is a file naming a gitdir outside the worktree, so a
    container with only the worktree mounted answered every git command with
    "not a git repository" - while the working rules Buddy appends to every
    brief tell the agent to commit as it goes. The agent's commits vanished
    and only Buddy's end-of-run checkpoint survived.
    """
    repo, worktree, prompt = worktree_task
    line = wrap_in_sandbox(
        _sandboxed(image),
        "git rev-parse --abbrev-ref HEAD && printf x > X.txt"
        " && git add X.txt && git commit -qm 'the agent committed' && echo committed",
        worktree,
        prompt_path=prompt,
        repo=repo,
        git_identity=asyncio.run(git_identity(repo)),
    )

    result = subprocess.run(
        ["bash", "-c", line], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL
    )

    assert "committed" in result.stdout, result.stdout + result.stderr
    assert "buddy/t-0001-x" in result.stdout
    # And the host sees it on the branch, which is what carries the work.
    log = subprocess.run(
        ["git", "log", "--format=%s", "-1", "buddy/t-0001-x"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert log.stdout.strip() == "the agent committed"


@needs_image
@needs_git_in_image
def test_the_agent_commits_under_the_repositorys_own_identity(worktree_task, image):
    """A container has no `~/.gitconfig`, so git refused to commit at all."""
    repo, worktree, prompt = worktree_task
    line = wrap_in_sandbox(
        _sandboxed(image),
        "printf x > X.txt && git add X.txt && git commit -qm identity",
        worktree,
        prompt_path=prompt,
        repo=repo,
        git_identity=asyncio.run(git_identity(repo)),
    )

    subprocess.run(["bash", "-c", line], capture_output=True, timeout=120, stdin=subprocess.DEVNULL)

    who = subprocess.run(
        ["git", "log", "--format=%an <%ae>", "-1", "buddy/t-0001-x"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert who.stdout.strip() == "You <you@example.com>"


@needs_image
@needs_git_in_image
def test_the_agent_cannot_leave_a_hook_that_runs_on_your_machine(worktree_task, image):
    """Committing needs the git directory, and the git directory is also where
    `core.hooksPath` and the hook scripts live - read by *your* git, on the
    host, the next time you use that repository. Measured before the fix: an
    agent in the sandbox wrote `hooksPath = /tmp/mine` into the host's
    `.git/config`, and it was still there afterwards."""
    repo, worktree, prompt = worktree_task
    line = wrap_in_sandbox(
        _sandboxed(image),
        "git config core.hooksPath /tmp/mine; "
        "printf 'evil' > .git/hooks/pre-commit 2>/dev/null; "
        f"printf 'evil' > {repo}/.git/hooks/pre-commit 2>/dev/null; echo tried",
        worktree,
        prompt_path=prompt,
        repo=repo,
        git_identity=asyncio.run(git_identity(repo)),
    )

    subprocess.run(["bash", "-c", line], capture_output=True, timeout=120, stdin=subprocess.DEVNULL)

    assert "hooksPath" not in (repo / ".git" / "config").read_text()
    hook = repo / ".git" / "hooks" / "pre-commit"
    assert not hook.exists() or "evil" not in hook.read_text()


def test_a_directory_that_is_not_a_checkout_asks_for_no_git_mounts(tmp_path):
    """A project that is not a git repository runs in place and has no branch
    to commit to; there is nothing to mount and nothing to protect."""
    from buddy.sandbox import git_support

    assert git_support(tmp_path) == ""


# -- a sandbox_command written before Buddy knew better -----------------------


def _notes(command: str) -> list[str]:
    config = HarnessConfig.from_dict(
        "probe", {"command": "x", "sandbox": "docker", "sandbox_command": command}
    )
    return _Probe(config).sandbox_notes()


def test_doctor_warns_about_a_sandbox_command_that_cannot_commit():
    """Nobody's `config.toml` is rewritten for them, and both of these fail
    silently: no git in the container, or a run that never prints a line."""
    stale = (
        "docker run --rm -i -v {worktree}:{worktree} -w {worktree} --entrypoint sh img -c {command}"
    )
    said = " ".join(_notes(stale))

    assert "{git}" in said and "cannot commit" in said
    assert "-i" in said and "stdin" in said


def test_the_shipped_sandbox_command_draws_no_warnings():
    assert _notes(DEFAULT_SANDBOX_COMMAND.replace("{image}", "img")) == []


def test_an_unsandboxed_harness_is_not_warned_about_a_sandbox():
    config = HarnessConfig.from_dict("probe", {"command": "x"})
    assert _Probe(config).sandbox_notes() == []


def test_an_image_name_containing_an_i_is_not_mistaken_for_the_flag():
    """`-i` is looked for before `--entrypoint`, so an image or a command
    that happens to contain one is not reported."""
    fine = DEFAULT_SANDBOX_COMMAND.replace("{image}", "registry.example/my-image:latest")
    assert _notes(fine) == []
