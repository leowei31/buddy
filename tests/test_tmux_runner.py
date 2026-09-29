"""Integration: real tmux, real pipe-pane, real pane_dead_status.

These are deliberately not mocked. The three tmux mechanisms are the ones the
whole design rests on, so they are proven against the tmux that is installed,
not against a fake that agrees with us.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest

from buddy.models import TaskRun
from buddy.tmux_runner import (
    MIN_TMUX_VERSION,
    PaneStatus,
    TmuxError,
    TmuxRunner,
    kill_process_tree,
    process_tree,
)

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")


async def wait_for(predicate, seconds: float = 10.0, interval: float = 0.05):
    """Poll until the predicate is truthy, then return its value.

    The predicate may be sync or may return an awaitable, so a plain
    `lambda: some_coroutine(...)` works and is actually awaited.
    """
    async with asyncio.timeout(seconds):
        while True:
            value = predicate()
            if inspect.isawaitable(value):
                value = await value
            if value:
                return value
            await asyncio.sleep(interval)


@pytest.fixture
def runner(tmux_socket: str) -> TmuxRunner:
    """A runner on its own tmux server (see conftest for the teardown)."""
    return TmuxRunner(session="buddy-test", socket_name=tmux_socket)


def make_run(tmp_path: Path, agent: str = "scout", attempt: int = 1) -> TaskRun:
    return TaskRun(
        task_id="t-0001",
        attempt=attempt,
        agent=agent,
        worktree=tmp_path / "wt",
        branch="buddy/t-0001-test",
        base_ref="HEAD",
        log_path=tmp_path / "tasks" / "t-0001" / f"attempt-{attempt}.log",
    )


def script(tmp_path: Path, body: str, name: str = "run.sh") -> Path:
    path = tmp_path / name
    path.write_text(f"#!/usr/bin/env bash\nset -uo pipefail\n{body}\n")
    return path


# -- version ---------------------------------------------------------------


async def test_version_is_new_enough_for_pane_dead_status(runner: TmuxRunner):
    assert await runner.check_version() >= MIN_TMUX_VERSION


# -- one window per agent --------------------------------------------------


async def test_the_first_agent_brings_the_session_and_the_last_takes_it(
    runner: TmuxRunner, tmp_path: Path
):
    """There is no fixed set of windows to create up front: each agent's is
    made when it starts, and tmux ends a session with its last window."""
    assert not await runner.session_exists()

    await runner.spawn("scout", make_run(tmp_path, "scout"), script(tmp_path, "sleep 30"))
    await runner.spawn("fixer", make_run(tmp_path, "fixer"), script(tmp_path, "sleep 30", "b.sh"))
    assert sorted(await runner.windows()) == ["fixer", "scout"]

    await runner.close_window("scout")
    assert await runner.windows() == ["fixer"]
    await runner.close_window("fixer")
    assert not await runner.session_exists()


async def test_opening_a_window_leaves_running_work_alone(runner: TmuxRunner, tmp_path: Path):
    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "echo alive; sleep 30"))
    before = await runner.status("scout")

    await runner.open_window("scout")

    after = await runner.status("scout")
    assert after.alive
    assert after.pid == before.pid  # the pane was never respawned
    assert await runner.windows() == ["scout"]


async def test_a_window_made_by_hand_still_reports_its_exit(runner: TmuxRunner, tmp_path: Path):
    """`remain-on-exit` is what keeps a dead pane and its status. A window
    someone made by hand does not have it, so it is set on every spawn."""
    await runner._call("new-session", "-d", "-s", runner.session, "-n", "scout")

    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "exit 7"))

    assert (await wait_for(lambda: _dead(runner, "scout"))).exit_code == 7


async def test_a_name_in_any_script_is_a_window(runner: TmuxRunner, tmp_path: Path):
    await runner.spawn("écrivain-2", make_run(tmp_path, "écrivain-2"), script(tmp_path, "exit 0"))
    assert (await wait_for(lambda: _dead(runner, "écrivain-2"))).exit_code == 0


# -- completion detection ------------------------------------------


@pytest.mark.parametrize("exit_code", [0, 1, 42])
async def test_pane_goes_dead_and_reports_its_exit_code(
    runner: TmuxRunner, tmp_path: Path, exit_code: int
):
    run = make_run(tmp_path)
    await runner.spawn("scout", run, script(tmp_path, f"echo working; exit {exit_code}"))

    status = await wait_for(lambda: _dead(runner, "scout"))
    assert status.exit_code == exit_code


async def test_a_pane_killed_by_a_signal_reports_the_signal_not_a_status(
    runner: TmuxRunner, tmp_path: Path
):
    """tmux gives no exit status for a process a signal ended. The manager
    falls back to the log and then to the signal; this proves the signal
    arrives."""
    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "kill -TERM $$; sleep 5"))

    status = await wait_for(lambda: _dead(runner, "scout"))

    assert status.exit_code is None
    assert status.signal == signal.SIGTERM


async def test_the_task_runs_under_bash_whatever_the_login_shell(
    runner: TmuxRunner, tmp_path: Path
):
    """tmux runs a one-string command through `default-shell -c`. A shell
    that runs startup code and does not exec its last command - as `sh`
    need not - would then be what tmux reports on, not the task."""
    ran = tmp_path / "login-shell-ran"
    shell = tmp_path / "shell"
    shell.write_text(f'#!/bin/sh\ntouch {ran}\n/bin/sh "$@"\n')
    shell.chmod(0o755)
    await runner.open_window("scout")
    await runner._tmux("set-option", "-g", "default-shell", str(shell))

    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "kill -TERM $$; sleep 5"))

    status = await wait_for(lambda: _dead(runner, "scout"))
    assert status.signal == signal.SIGTERM
    assert not ran.exists()


async def _dead(runner: TmuxRunner, agent: str) -> PaneStatus | None:
    status = await runner.status(agent)
    return status if status.dead else None


async def test_a_dead_pane_always_says_how_it_ended(runner: TmuxRunner, tmp_path: Path):
    """Ubuntu's tmux can lose the `SIGCHLD` of a pane whose terminal closed
    a moment before it exited, and then reports it dead with no status
    (see `_remind_to_reap`). A process that closes its terminal first is
    the shape that loses it; on ubuntu:24.04 one in a dozen or so did. A
    tmux that never loses it passes this trivially."""
    body = script(tmp_path, "exec </dev/null >/dev/null 2>&1; exit 3")
    for attempt in range(1, 25):
        await runner.spawn("scout", make_run(tmp_path, attempt=attempt), body)

        status = await wait_for(lambda: _dead(runner, "scout"))

        assert status.exit_code == 3, f"attempt {attempt}: {status}"


class ScriptedTmux(TmuxRunner):
    """Answers `list-panes` from a script, one reply per call, and fails on
    a call it has no reply for."""

    def __init__(self, *replies: str):
        super().__init__(session="buddy-test", socket_name="scripted")
        self.replies = list(replies)

    async def session_exists(self) -> bool:
        return True

    async def _call(self, *args: str) -> tuple[int, str, str]:
        assert args[0] == "list-panes", args
        return 0, self.replies.pop(0), ""


#: Stands in for a tmux server: says when it is ready, then reports the
#: `SIGCHLD` it is sent.
SIGCHLD_CATCHER = """
import signal, sys, time
def caught(*_):
    print("SIGCHLD", flush=True)
    sys.exit(0)
signal.signal(signal.SIGCHLD, caught)
print("ready", flush=True)
time.sleep(30)
"""


@contextlib.asynccontextmanager
async def stand_in_server():
    server = await asyncio.create_subprocess_exec(
        sys.executable, "-c", SIGCHLD_CATCHER, stdout=asyncio.subprocess.PIPE
    )
    try:
        assert server.stdout is not None
        assert await server.stdout.readline() == b"ready\n"
        yield server
    finally:
        with contextlib.suppress(ProcessLookupError):
            server.kill()
        await server.wait()


def pane_line(server_pid: int, dead: str = "1", status: str = "", signal_name: str = "") -> str:
    return f"scout\t{dead}\t{status}\t4242\t1\t1\t{signal_name}\t{server_pid}\n"


async def test_a_dead_pane_with_no_status_reminds_tmux_to_reap():
    """The one test here against a stand-in: a lost `SIGCHLD` cannot be
    staged on demand, so this proves the reminder is sent, to the server
    that reported the pane, and that the second look is what is returned."""
    async with stand_in_server() as server:
        runner = ScriptedTmux(pane_line(server.pid), pane_line(server.pid, status="0"))

        status = await runner.status("scout")

        assert status.exit_code == 0
        assert server.stdout is not None
        assert await asyncio.wait_for(server.stdout.readline(), 5) == b"SIGCHLD\n"


@pytest.mark.parametrize(
    "fields",
    [{"status": "3"}, {"signal_name": "term"}, {"dead": "0"}],
    ids=["exited", "signalled", "alive"],
)
async def test_a_pane_that_needs_no_reaping_is_read_once(fields: dict[str, str]):
    """One reply is scripted, so a second look would fail the test."""
    async with stand_in_server() as server:
        runner = ScriptedTmux(pane_line(server.pid, **fields))

        await runner.status("scout")

        assert server.returncode is None


async def test_a_running_pane_reports_alive_with_no_exit_code(runner: TmuxRunner, tmp_path: Path):
    await runner.spawn("fixer", make_run(tmp_path, "fixer"), script(tmp_path, "sleep 30"))
    status = await runner.status("fixer")
    assert status.alive
    assert status.exit_code is None
    assert status.pid


async def test_status_of_a_missing_window_says_so(runner: TmuxRunner, tmp_path: Path):
    """The restart case: tmux was killed underneath a run."""
    assert await runner.status("scout") == PaneStatus(exists=False)
    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "sleep 30"))
    await runner.kill_session()
    assert not (await runner.status("scout")).exists


async def test_a_name_is_never_prefix_matched_onto_another_window(
    runner: TmuxRunner, tmp_path: Path
):
    """Without the `=` prefix, tmux resolves an unknown window name to any
    window it prefixes, so `scout` would silently act on `scout-2`."""
    await runner.spawn("scout-2", make_run(tmp_path, "scout-2"), script(tmp_path, "sleep 30"))
    assert not await runner.window_exists("scout")
    assert not (await runner.status("scout")).exists


async def test_a_missing_window_never_reports_another_agents_pane(
    runner: TmuxRunner, tmp_path: Path
):
    """Regression: `display -p -t <session>:<window>` answers for the
    session's *current* window when the target does not resolve. Reporting a
    live pane for an agent whose window is gone would defeat reconciliation."""
    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "sleep 30"))
    await runner.spawn("fixer", make_run(tmp_path, "fixer"), script(tmp_path, "sleep 30", "b.sh"))
    assert (await runner.status("scout")).alive

    assert not (await runner.status("sco")).exists
    assert not (await runner.status("Nonexistent")).exists

    await runner._tmux("kill-window", "-t", runner.target("scout"))
    assert not (await runner.status("scout")).exists


async def test_panes_reports_every_agent_in_one_call(runner: TmuxRunner, tmp_path: Path):
    for index, name in enumerate(("a", "b", "c")):
        await runner.spawn(
            name, make_run(tmp_path, name), script(tmp_path, "sleep 30", f"{index}.sh")
        )
    panes = await runner.panes()
    assert set(panes) == {"a", "b", "c"}
    assert all(status.alive for status in panes.values())


async def test_a_neighbouring_window_is_left_alone(runner: TmuxRunner, tmp_path: Path):
    await runner._call("new-session", "-d", "-s", runner.session, "-n", "sco")

    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "exit 3"))
    dead = await wait_for(lambda: _dead(runner, "scout"))

    assert dead.exit_code == 3
    decoy = await runner.status("sco")
    assert decoy.alive and not decoy.piped


# -- logging -------------------------------------------------------


async def test_pipe_pane_writes_an_append_only_log(runner: TmuxRunner, tmp_path: Path):
    run = make_run(tmp_path)
    await runner.spawn("scout", run, script(tmp_path, "sleep 0.25; echo __BUDDY_MARKER__; exit 0"))
    await wait_for(lambda: run.log_path.exists() and "__BUDDY_MARKER__" in run.log_path.read_text())
    assert (await runner.status("scout")).piped


async def test_the_pipe_is_open_immediately_after_respawn(runner: TmuxRunner, tmp_path: Path):
    """Verify, and reopen if respawn dropped it."""
    run = make_run(tmp_path)
    await runner.spawn("scout", run, script(tmp_path, "sleep 0.25; echo hello; sleep 5"))
    assert await runner.is_piped("scout")


async def test_a_second_attempt_writes_a_separate_log(runner: TmuxRunner, tmp_path: Path):
    """A retry reuses its agent's window when one is still there."""
    first = make_run(tmp_path, attempt=1)
    await runner.spawn("scout", first, script(tmp_path, "sleep 0.25; echo FIRST; exit 0"))
    await wait_for(lambda: "FIRST" in _read(first.log_path))

    second = make_run(tmp_path, attempt=2)
    await runner.spawn("scout", second, script(tmp_path, "sleep 0.25; echo SECOND; exit 0"))
    await wait_for(lambda: "SECOND" in _read(second.log_path))

    assert "SECOND" not in first.log_path.read_text()
    assert await runner.windows() == ["scout"]


async def test_a_capped_log_rotates_mid_burst_losing_nothing(tmux_socket, tmp_path: Path):
    """Logs grew without bound, and the obvious fix - rename the log,
    then point `pipe-pane` at a new one - lost two to four lines at every
    switch, measured here. The writer now rotates itself. 30,000 numbered
    lines, as fast as bash prints them, must all arrive once and in order."""
    import re

    runner = TmuxRunner(session="buddy-test", socket_name=tmux_socket, max_log_bytes=250_000)
    run = make_run(tmp_path)
    body = "sleep 0.25; for i in $(seq 1 30000); do echo line-$i; done; echo __ALL_DONE__; sleep 5"
    await runner.spawn("scout", run, script(tmp_path, body))

    await wait_for(lambda: "__ALL_DONE__" in _read(run.log_path), seconds=30)
    rotated = run.log_path.with_name(run.log_path.name + ".1")
    assert rotated.exists(), "the cap was crossed, so there is a previous file"
    assert rotated.stat().st_size <= 250_000
    numbers = [int(n) for n in re.findall(r"line-(\d+)", _read(rotated) + _read(run.log_path))]
    assert numbers == list(range(1, 30001))


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


async def test_capture_shows_the_screen(runner: TmuxRunner, tmp_path: Path):
    await runner.spawn(
        "scout", make_run(tmp_path), script(tmp_path, "echo ON_SCREEN_NOW; sleep 30")
    )
    screen = await wait_for(lambda: _captured(runner, "scout", "ON_SCREEN_NOW"))
    assert "ON_SCREEN_NOW" in screen


@pytest.mark.parametrize("agent", ["scout", None], ids=["one agent", "the session"])
async def test_attaching_is_read_only_and_lands_where_asked(
    runner: TmuxRunner, tmp_path: Path, agent: str | None
):
    """What `buddy watch` and `buddy attach` exec, run from a real terminal:
    tmux itself reports the client read-only, and on the agent's window."""
    await runner.spawn("fixer", make_run(tmp_path, "fixer"), script(tmp_path, "sleep 30", "b.sh"))
    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "sleep 30"))
    await runner._tmux("select-window", "-t", runner.target("fixer"))
    terminal, tty = os.openpty()
    env = {key: value for key, value in os.environ.items() if key != "TMUX"}
    client = await asyncio.create_subprocess_exec(
        *runner.attach_command(agent),
        stdin=tty,
        stdout=tty,
        stderr=tty,
        env=env | {"TERM": "xterm"},
    )
    os.close(tty)
    try:
        seen = await wait_for(lambda: _clients(runner))
        assert seen == [("1", "scout" if agent else "fixer")]
    finally:
        await runner._call("detach-client", "-s", runner.session)
        with contextlib.suppress(ProcessLookupError):
            client.kill()
        await client.wait()
        os.close(terminal)


async def _clients(runner: TmuxRunner) -> list[tuple[str, str]]:
    """Each attached client: read-only or not, and the window it is on."""
    out = await runner._tmux("list-clients", "-F", "#{client_readonly}\t#{window_name}")
    rows = (line.split("\t", 1) for line in out.splitlines() if line)
    return [(read_only, window) for read_only, window in rows]


async def _captured(runner: TmuxRunner, agent: str, needle: str) -> str | None:
    screen = await runner.capture(agent)
    return screen if needle in screen else None


# -- kill ----------------------------------------------------------


async def test_kill_takes_the_whole_process_tree(runner: TmuxRunner, tmp_path: Path):
    """A dev server or watcher the harness started must not outlive the task."""
    pids_file = tmp_path / "pids"
    body = (
        "sleep 300 & child=$!\n"
        "( sleep 300 ) & grandchild=$!\n"
        f'echo "$child $grandchild $$" > {pids_file}\n'
        "wait"
    )
    await runner.spawn("builder", make_run(tmp_path, "builder"), script(tmp_path, body))
    await wait_for(lambda: pids_file.exists() and len(pids_file.read_text().split()) == 3)
    child, grandchild, shell = (int(p) for p in pids_file.read_text().split())

    await runner.kill("builder")

    for pid in (child, grandchild, shell):
        await wait_for(lambda pid=pid: _gone(pid), seconds=15)


async def _gone(pid: int) -> bool:
    """Not running: absent, or a zombie - exited, with only its parent's
    bookkeeping left, which in a container whose init reaps nothing is
    where an orphan stays."""
    state = _proc_state(pid)
    if state is None:
        proc = await asyncio.create_subprocess_exec(
            "ps", "-o", "stat=", "-p", str(pid), stdout=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        state = out.decode().strip()
    return not state or state.startswith("Z")


def _proc_state(pid: int) -> str | None:
    """The state letter from `/proc`, "" for a pid that is gone, None where
    there is no `/proc` to ask."""
    if not Path("/proc").is_dir():
        return None
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2][:1]
    except OSError:
        return ""


async def test_kill_removes_the_agents_window(runner: TmuxRunner, tmp_path: Path):
    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "sleep 300"))
    await runner.spawn("fixer", make_run(tmp_path, "fixer"), script(tmp_path, "sleep 300", "b.sh"))

    await runner.kill("scout")

    assert not (await runner.status("scout")).exists
    assert (await runner.status("fixer")).alive, "a neighbour is untouched"


async def test_kill_is_safe_on_an_already_dead_pane(runner: TmuxRunner, tmp_path: Path):
    await runner.spawn("scout", make_run(tmp_path), script(tmp_path, "exit 0"))
    await wait_for(lambda: _dead(runner, "scout"))
    await runner.kill("scout")
    assert not (await runner.status("scout")).exists


async def test_closing_a_window_that_is_not_there_is_harmless(runner: TmuxRunner):
    await runner.close_window("nobody")


# -- process-tree helpers --------------------------------------------------


async def test_process_tree_finds_descendants():
    proc = await asyncio.create_subprocess_exec("bash", "-c", "sleep 30 & sleep 30")
    try:
        tree = await wait_for(lambda: _tree_of(proc.pid, 2))
        assert proc.pid in tree
        assert len(tree) >= 2
    finally:
        await kill_process_tree(proc.pid, grace=1.0)
        await proc.wait()


async def _tree_of(pid: int, at_least: int) -> list[int] | None:
    tree = await process_tree(pid)
    return tree if len(tree) >= at_least else None


async def test_kill_process_tree_never_signals_buddys_own_group():
    """`_process_group` returns None for our own pgid, so a bad pane_pid can
    never take Buddy down with the task."""
    from buddy.tmux_runner import _process_group

    assert _process_group(os.getpid()) is None


async def test_tmux_errors_are_raised_not_swallowed(runner: TmuxRunner):
    with pytest.raises(TmuxError, match="failed"):
        await runner._tmux("display", "-p", "-t", "=nosuchsession:=nope", "#{pane_pid}")


# -- finding a process's children -----------------------------------


def test_children_are_read_from_proc_where_there_is_one(tmp_path: Path):
    """A minimal Linux image often has no `pgrep`, and `buddy kill` died on
    that with a FileNotFoundError - found running the suite on Debian.
    `/proc` needs no command at all."""
    from buddy.tmux_runner import children_from_proc

    table = [(10, 1, "init"), (11, 10, "bash"), (12, 10, "a name (odd) "), (13, 11, "x")]
    for pid, ppid, comm in table:
        entry = tmp_path / str(pid)
        entry.mkdir()
        (entry / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0 -1 0 0 0\n")
    (tmp_path / "self").mkdir()  # not a pid, and must not trip the scan

    assert children_from_proc(10, tmp_path) == [11, 12]
    assert children_from_proc(11, tmp_path) == [13]
    assert children_from_proc(13, tmp_path) == []


def test_a_zombie_is_not_running(tmp_path: Path):
    """An exited process its parent has not reaped yet is a zombie, and
    waiting for one to die cost `kill` its whole grace period."""
    from buddy.tmux_runner import running_from_proc

    for pid, state in [(20, "S"), (21, "R"), (22, "Z"), (23, "X")]:
        entry = tmp_path / str(pid)
        entry.mkdir()
        (entry / "stat").write_text(f"{pid} (a name (odd) ) {state} 1 0 0 0 -1 0 0 0\n")

    assert running_from_proc(20, tmp_path)
    assert running_from_proc(21, tmp_path)
    assert not running_from_proc(22, tmp_path)
    assert not running_from_proc(23, tmp_path)
    assert not running_from_proc(24, tmp_path)  # gone altogether


@pytest.mark.skipif(shutil.which("ps") is None, reason="no ps either; /proc is the only way here")
async def test_children_are_found_without_pgrep(monkeypatch):
    """The `ps` fallback, run for real against a process that has a child.

    Where neither `pgrep` nor `ps` exists - a minimal Linux image - `/proc`
    is the path that runs, and it is covered above.
    """
    import shutil as shutil_module

    from buddy import tmux_runner

    real_which = shutil_module.which

    def without_pgrep(name, *args, **kwargs):
        return None if name == "pgrep" else real_which(name, *args, **kwargs)

    monkeypatch.setattr(tmux_runner.shutil, "which", without_pgrep)
    monkeypatch.setattr(tmux_runner, "HAS_PROC", False)

    parent = await asyncio.create_subprocess_exec(
        "bash", "-c", "sleep 30 & wait", stdout=asyncio.subprocess.DEVNULL
    )
    try:
        tree = await wait_for(lambda: _tree_with_child(parent.pid))
        assert parent.pid in tree and len(tree) >= 2
    finally:
        for pid in await tmux_runner.process_tree(parent.pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        await parent.wait()


async def _tree_with_child(pid: int):
    from buddy import tmux_runner

    tree = await tmux_runner.process_tree(pid)
    return tree if len(tree) >= 2 else None
