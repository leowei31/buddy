"""Kill tmux mid-run, restart, assert correct outcomes.

Real tmux and real git: the whole claim of reconciliation is that the log
file and the worktree survive things the process does not, so faking
either of them would test nothing.

The harness is a shell script rather than a real coding agent, because what is
under test is Buddy's recovery, not an agent's work.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from buddy.config import Config, HarnessConfig
from buddy.harnesses.base import ResultSummary
from buddy.logs import DONE_SENTINEL, read_sentinels
from buddy.manager import AgentManager
from buddy.models import (
    RunOutcome,
    TaskRequeued,
    TaskSpec,
    TaskState,
)
from buddy.state import Store
from buddy.tmux_runner import TmuxRunner
from buddy.workspace import Workspace
from tests.helpers import wait_for

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")


class ScriptAdapter:
    """A stand-in harness: whatever shell command the config names.

    Not a mock of tmux or git, which are real here; just a harness that costs
    nothing to run and whose behaviour a test can choose.
    """

    name = "script"

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.waiting_patterns = list(config.waiting_patterns)

    def invocation(self, run, prompt_path: Path, *, model: str | None = None) -> str:
        return self.config.command.format(
            prompt_path=prompt_path, worktree=run.worktree, model=model or "", model_flag=""
        )

    def parse_result(self, log_text: str, exit_code: int) -> ResultSummary:
        return ResultSummary(ok=exit_code == 0, summary="script finished", exit_code=exit_code)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "code" / "webapp"
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.name", "Test")
    git(path, "config", "user.email", "test@example.com")
    (path / "README.md").write_text("hello\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")
    return path


def write_config(home: Path, repo: Path, command: str, *, max_concurrent: int = 0) -> Config:
    (home / "config.toml").write_text(
        f"[buddy]\nmax_concurrent = {max_concurrent}\n"
        f'[projects.webapp]\npath = "{repo}"\nbase_branch = "main"\n'
        f"[harness.script]\ncommand = {command!r}\n"
    )
    return Config.load(home=home)


class Buddy:
    """One Buddy process's worth of state, so a test can throw it away and
    start another against the same tmux session and database."""

    def __init__(self, config: Config, socket: str) -> None:
        self.config = config
        self.store = Store(config.paths.db)
        self.runner = TmuxRunner(session="buddy-reconcile", socket_name=socket)
        self.workspace = Workspace(config)
        self.manager = AgentManager(
            config,
            self.store,
            self.runner,
            self.workspace,
            adapter_for=lambda name: ScriptAdapter(config.harness(name)),
        )

    def close(self) -> None:
        self.store.close()


@pytest.fixture
def socket(tmux_socket: str) -> str:
    """Its own tmux server; conftest kills it and unlinks the socket."""
    return tmux_socket


def make_task(store: Store, **overrides) -> TaskSpec:
    defaults = {
        "id": store.next_task_id(),
        "title": "a task",
        "brief": "# Task\ndo it",
        "harness": "script",
        "project": "webapp",
        "priority": 3,
    }
    return TaskSpec(**(defaults | overrides))


async def running_pane(buddy: Buddy, agent: str):
    status = await buddy.runner.status(agent)
    return status if status.alive else None


# -- scenario 1: the pane ended while Buddy was away ----------------------


async def test_a_run_that_finished_while_buddy_was_down_is_finalized(
    tmp_path: Path, repo: Path, socket: str
):
    """Pane dead with an exit status - we just missed the moment."""
    config = write_config(tmp_path, repo, "bash -c 'echo working; exit 0'")
    first = Buddy(config, socket)
    task = make_task(first.store)
    await first.manager.submit(task)
    pane = await wait_for(lambda: _dead(first, task.agent))
    first.close()

    second = Buddy(config, socket)
    events = await second.manager.reconcile()

    assert second.store.get_task_state(task.id) is TaskState.DONE, evidence(config, task, pane)
    assert second.store.get_run(task.id, 1).outcome is RunOutcome.DONE
    assert config.paths.result_file(task.id).exists()
    assert second.store.get_agent(task.agent) is None, "the agent is retired"
    assert not await second.runner.window_exists(task.agent), "and its window is gone"
    assert events
    second.close()


def evidence(config: Config, task: TaskSpec, pane=None) -> str:
    """What a failure here needs to be diagnosed from an annotation alone:
    the pane as tmux last reported it, and the attempt's log."""
    log = config.paths.log_file(task.id, 1)
    text = log.read_text() if log.exists() else "(no log)"
    return f"pane={pane!r}\nlog:\n{text[-1500:]}"


async def _dead(buddy: Buddy, agent: str):
    status = await buddy.runner.status(agent)
    return status if status.dead else None


async def test_a_failed_run_is_finalized_as_an_error(tmp_path: Path, repo: Path, socket: str):
    config = write_config(tmp_path, repo, "bash -c 'exit 3'")
    first = Buddy(config, socket)
    task = make_task(first.store)
    await first.manager.submit(task)
    pane = await wait_for(lambda: _dead(first, task.agent))
    first.close()

    second = Buddy(config, socket)
    await second.manager.reconcile()

    assert second.store.get_task_state(task.id) is TaskState.ERROR
    assert second.store.get_run(task.id, 1).exit_code == 3, evidence(config, task, pane)
    second.close()


# -- scenario 2: tmux died underneath a running task (the gate) -----------


async def test_killing_tmux_mid_run_loses_no_work(tmp_path: Path, repo: Path, socket: str):
    """The headline case: machine rebooted, tmux killed, Buddy restarted.

    Nothing is lost because the worktree and the log outlive the pane.
    """
    config = write_config(
        tmp_path,
        repo,
        "bash -c 'echo starting; echo half-finished > work.txt; sleep 300'",
    )
    first = Buddy(config, socket)
    task = make_task(first.store)
    await first.manager.submit(task)
    worktree = config.worktree_path("webapp", task.id)
    await wait_for(lambda: (worktree / "work.txt").exists())
    assert await running_pane(first, task.agent)
    first.close()

    # tmux dies with the task mid-flight.
    await TmuxRunner(socket_name=socket)._call("kill-server")

    second = Buddy(config, socket)
    events = await second.manager.reconcile()

    # The attempt is recorded as interrupted, not silently lost.
    assert second.store.get_run(task.id, 1).outcome is RunOutcome.INTERRUPTED
    requeued = [event for event in events if isinstance(event, TaskRequeued)]
    assert requeued and requeued[0].reason is RunOutcome.INTERRUPTED

    # The partial work was checkpointed onto the task's own branch.
    assert (worktree / "work.txt").read_text() == "half-finished\n"
    assert "WIP (buddy, interrupted, attempt 1)" in git(worktree, "log", "-1", "--pretty=%s")

    # And it is queued to resume, at its original priority, one attempt later.
    assert second.store.get_task_state(task.id) is TaskState.QUEUED
    assert second.store.get_task(task.id).attempt == 2
    assert second.store.get_task(task.id).priority == 3
    assert second.store.get_task(task.id).agent == task.agent, "it keeps its name"
    assert second.store.get_agent(task.agent) is None
    second.close()


async def test_the_resumed_attempt_reuses_the_same_worktree_and_is_told_so(
    tmp_path: Path, repo: Path, socket: str
):
    """A retry resumes its partial work, reached through the restart path."""
    config = write_config(tmp_path, repo, "bash -c 'echo partial > work.txt; sleep 300'")
    first = Buddy(config, socket)
    task = make_task(first.store)
    await first.manager.submit(task)
    worktree = config.worktree_path("webapp", task.id)
    await wait_for(lambda: (worktree / "work.txt").exists())
    first.close()
    await TmuxRunner(socket_name=socket)._call("kill-server")

    second = Buddy(config, socket)
    await second.manager.reconcile()
    await second.manager.schedule()

    run = second.store.get_run(task.id, 2)
    assert run is not None
    assert run.worktree == worktree
    assert run.branch == second.store.get_run(task.id, 1).branch
    prompt = config.paths.prompt_file(task.id).read_text()
    assert "partial work from a previous attempt" in prompt
    assert (worktree / "work.txt").exists()
    second.close()


# -- scenario 3: the task is still running across a restart ---------------


async def test_a_still_running_task_is_resumed_not_disturbed(
    tmp_path: Path, repo: Path, socket: str
):
    config = write_config(tmp_path, repo, "bash -c 'echo alive; sleep 300'")
    first = Buddy(config, socket)
    task = make_task(first.store)
    await first.manager.submit(task)
    before = await first.runner.status(task.agent)
    first.close()

    second = Buddy(config, socket)
    await second.manager.reconcile()

    after = await second.runner.status(task.agent)
    assert after.alive
    assert after.pid == before.pid  # the same process, never respawned
    assert after.piped  # the log pipe was verified
    assert second.store.get_task_state(task.id) is TaskState.RUNNING
    assert second.store.get_agent(task.agent) is not None
    second.close()


# -- scenario 4: window gone but the log says how it ended ----------------


async def test_a_missing_window_is_finalized_from_the_logs_sentinel(
    tmp_path: Path, repo: Path, socket: str
):
    """A `__BUDDY_DONE__` line means finalize with that code."""
    config = write_config(tmp_path, repo, "bash -c 'exit 5'")
    first = Buddy(config, socket)
    task = make_task(first.store)
    await first.manager.submit(task)
    log = config.paths.log_file(task.id, 1)
    await wait_for(lambda: read_sentinels(log, task.id).finished)
    assert DONE_SENTINEL in log.read_text()

    # The window disappears without Buddy ever seeing the dead pane.
    await first.runner._tmux("kill-window", "-t", first.runner.target(task.agent))
    first.close()

    second = Buddy(config, socket)
    await second.manager.reconcile()

    assert second.store.get_run(task.id, 1).exit_code == 5
    assert second.store.get_run(task.id, 1).outcome is RunOutcome.ERROR
    assert second.store.get_task_state(task.id) is TaskState.ERROR
    second.close()


async def test_a_stale_sentinel_from_a_previous_task_is_not_believed(
    tmp_path: Path, repo: Path, socket: str
):
    """The id on the sentinel is what makes it trustworthy."""
    config = write_config(tmp_path, repo, "bash -c 'echo running; sleep 300'")
    buddy = Buddy(config, socket)
    task = make_task(buddy.store)
    await buddy.manager.submit(task)
    log = config.paths.log_file(task.id, 1)
    await wait_for(lambda: log.exists() and "running" in log.read_text())

    # Another task's completion line lands in this log.
    with log.open("a") as handle:
        handle.write(f"\n{DONE_SENTINEL} t-9999 0 2026-01-01T00:00:00Z\n")
    await buddy.runner._call("kill-server")

    await buddy.manager.reconcile()

    # Interrupted, not "finished with exit 0".
    assert buddy.store.get_run(task.id, 1).outcome is RunOutcome.INTERRUPTED
    buddy.close()


# -- the queue survives a restart ---------------------------


async def test_the_queue_is_rebuilt_from_the_database(tmp_path: Path, repo: Path, socket: str):
    config = write_config(tmp_path, repo, "bash -c 'sleep 300'", max_concurrent=2)
    first = Buddy(config, socket)
    # Built and submitted one at a time: `next_task_id` only sees persisted
    # tasks, so building them all first would give them the same id.
    running = []
    for _ in range(2):
        running.append(make_task(first.store))
        await first.manager.submit(running[-1])
    queued = make_task(first.store, priority=1, title="waiting its turn")
    await first.manager.submit(queued)
    assert first.store.get_task_state(queued.id) is TaskState.QUEUED
    first.close()

    second = Buddy(config, socket)
    await second.manager.reconcile()

    assert [task.id for task in second.manager.ready_tasks()] == [queued.id]
    # And it starts, in a window of its own, as soon as there is room.
    await second.manager.kill_agent(running[0].agent)
    assert second.store.get_task_state(queued.id) is TaskState.RUNNING
    assert await running_pane(second, "waiting-its-turn")
    second.close()


async def test_reconciling_a_clean_slate_does_nothing(tmp_path: Path, repo: Path, socket: str):
    config = write_config(tmp_path, repo, "bash -c 'true'")
    buddy = Buddy(config, socket)
    assert await buddy.manager.reconcile() == []
    assert buddy.manager.agents() == []
    assert not await buddy.runner.session_exists(), "nothing to run, so no windows"
    buddy.close()


async def test_reconcile_is_idempotent(tmp_path: Path, repo: Path, socket: str):
    config = write_config(tmp_path, repo, "bash -c 'exit 0'")
    buddy = Buddy(config, socket)
    task = make_task(buddy.store)
    await buddy.manager.submit(task)
    await wait_for(lambda: _dead(buddy, task.agent))

    await buddy.manager.reconcile()
    state = buddy.store.get_task_state(task.id)
    await buddy.manager.reconcile()

    assert buddy.store.get_task_state(task.id) is state
    assert len(buddy.store.runs_for(task.id)) == 1
    buddy.close()
