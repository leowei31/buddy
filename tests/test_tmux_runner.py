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
    assert status.dead
    assert status.exit_code == exit_code


async def _dead(runner: TmuxRunner, agent: str) -> PaneStatus | None:
    status = await runner.status(agent)
    return status if status.dead else None


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
        await wait_for(lambda pid=pid: not _pid_alive(pid), seconds=15)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
