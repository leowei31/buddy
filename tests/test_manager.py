"""Scheduling, deps, preemption/requeue, health - fake runner + workspace.

The fakes exist so these tests can drive time and pane state directly. tmux
and git are proven for real in `test_tmux_runner.py` and `test_workspace.py`;
what is under test here is the manager's decisions.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from buddy.config import Config
from buddy.harnesses.claude_code import ClaudeCodeAdapter
from buddy.manager import AgentManager
from buddy.models import (
    AgentHealthChanged,
    AgentNameError,
    AgentStatus,
    PreemptionProposal,
    RunOutcome,
    TaskBlocked,
    TaskFinished,
    TaskRequeued,
    TaskSpec,
    TaskStarted,
    TaskState,
    utcnow,
)
from buddy.sandbox import DEFAULT_SANDBOX_COMMAND
from buddy.state import Store
from buddy.tmux_runner import PaneStatus
from buddy.workspace import Checkout, WorkspaceError

# -- fakes -----------------------------------------------------------------


class FakeRunner:
    """Stands in for tmux: pane state is set by the test, not by a process.

    One window per agent, as in tmux: made when it spawns, gone when it is
    killed or closed.
    """

    def __init__(self) -> None:
        self.panes_state: dict[str, PaneStatus] = {}
        self.spawned: list[tuple[str, str, int]] = []
        self.killed: list[str] = []
        self.closed: list[str] = []
        self.pipes_opened: list[str] = []

    async def panes(self) -> dict[str, PaneStatus]:
        return dict(self.panes_state)

    async def status(self, agent: str) -> PaneStatus:
        return self.panes_state.get(agent, PaneStatus(exists=False))

    async def spawn(self, agent: str, run, script: Path) -> None:
        self.spawned.append((agent, run.task_id, run.attempt))
        self.panes_state[agent] = PaneStatus(exists=True, dead=False, pid=1234, piped=True)
        run.log_path.parent.mkdir(parents=True, exist_ok=True)
        run.log_path.touch()

    async def kill(self, agent: str) -> None:
        self.killed.append(agent)
        self.panes_state.pop(agent, None)

    async def close_window(self, agent: str) -> None:
        if self.panes_state.pop(agent, None) is not None:
            self.closed.append(agent)

    async def close_pipe(self, agent: str) -> bool:
        return True

    async def open_pipe(self, agent: str, log_path: Path) -> bool:
        self.pipes_opened.append(agent)
        return True

    def finish(self, agent: str, exit_code: int = 0) -> None:
        """The pane exits: the completion signal."""
        self.panes_state[agent] = PaneStatus(
            exists=True, dead=True, exit_code=exit_code, pid=1234, piped=True
        )

    def lose_window(self, agent: str) -> None:
        self.panes_state.pop(agent, None)


class FakeWorkspace:
    """Stands in for git: worktrees are directories, checkpoints are recorded."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.checkpoints: list[tuple[str, int, RunOutcome]] = []
        self.non_git: set[str] = set()
        self.pruned: list[str] = []
        self.prune_explodes: set[str] = set()
        self.worktree_prunes: list[str] = []

    async def requires_exclusive_run(self, project: str) -> bool:
        return project in self.non_git

    async def create(self, task: TaskSpec) -> Checkout:
        path = self.root / task.project / task.id
        path.mkdir(parents=True, exist_ok=True)
        if task.project in self.non_git:
            return Checkout(path=path, branch=None, base_ref="", is_worktree=False)
        return Checkout(
            path=path,
            branch=f"buddy/{task.id}-slug",
            base_ref="basecommit",
            is_worktree=True,
        )

    async def checkpoint(self, run, outcome=None) -> str | None:
        self.checkpoints.append((run.task_id, run.attempt, outcome))
        return f"wip{run.attempt}"

    async def prune_discarded(self, task, discarded_at, *, now=None) -> bool:
        self.pruned.append(task.id)
        if task.id in self.prune_explodes:
            raise RuntimeError("index.lock exists")
        return True

    async def prune_worktrees(self, project: str) -> None:
        self.worktree_prunes.append(project)


class Clock:
    """Time the tests advance by hand, so a ten-minute stall costs nothing."""

    def __init__(self) -> None:
        self.now = utcnow()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


# -- fixtures --------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path / "webapp"}"\n'
        f'[projects.notes]\npath = "{tmp_path / "notes"}"\n'
        "[harness.claude_code]\n"
        'command = "claude -p < {prompt_path}"\n'
        "waiting_patterns = ['\\(y/n\\)']\n"
    )
    return Config.load(home=tmp_path)


@pytest.fixture
def store(config: Config) -> Store:
    with Store(config.paths.db) as s:
        yield s


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def workspace(tmp_path: Path) -> FakeWorkspace:
    return FakeWorkspace(tmp_path / "worktrees")


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def manager(config, store, runner, workspace, clock) -> AgentManager:
    return AgentManager(
        config,
        store,
        runner,
        workspace,
        adapter_for=lambda name: ClaudeCodeAdapter(config.harness(name)),
        now=clock,
    )


def limited_manager(tmp_path, store, runner, workspace, clock, limit: int) -> AgentManager:
    """A manager under `max_concurrent = limit`."""
    (tmp_path / "config.toml").write_text(
        (tmp_path / "config.toml").read_text() + f"\n[buddy]\nmax_concurrent = {limit}\n"
    )
    limited = Config.load(home=tmp_path)
    return AgentManager(
        limited,
        store,
        runner,
        workspace,
        adapter_for=lambda n: ClaudeCodeAdapter(limited.harness(n)),
        now=clock,
    )


@pytest.fixture
def capped(tmp_path, store, runner, workspace, clock) -> AgentManager:
    """Seven at a time. Preemption only exists where there is a limit: with
    none, anything ready simply starts."""
    return limited_manager(tmp_path, store, runner, workspace, clock, 7)


#: Ids handed out this test. `next_task_id` only sees *persisted* tasks, and
#: these tests build several before submitting any.
_issued: set[str] = set()


@pytest.fixture(autouse=True)
def _fresh_ids():
    _issued.clear()
    yield


def _unique_id(store: Store) -> str:
    number = int(store.next_task_id()[2:])
    while f"t-{number:04d}" in _issued:
        number += 1
    task_id = f"t-{number:04d}"
    _issued.add(task_id)
    return task_id


def make_task(store: Store, **overrides) -> TaskSpec:
    defaults = {
        "id": _unique_id(store),
        "title": "a task",
        "brief": "# Task\ndo it",
        "harness": "claude_code",
        "project": "webapp",
        "priority": 3,
    }
    return TaskSpec(**(defaults | overrides))


def kinds(events, kind) -> list:
    return [event for event in events if isinstance(event, kind)]


# -- scheduling -----------------------------------------------------


async def test_submitting_a_task_starts_it_under_its_agents_name(manager, store, runner):
    task = make_task(store, agent="scout")
    events = await manager.submit(task)

    started = kinds(events, TaskStarted)
    assert len(started) == 1
    assert started[0].agent == "scout"
    assert runner.spawned == [("scout", task.id, 1)]
    assert store.get_task_state(task.id) is TaskState.RUNNING
    assert store.get_agent("scout").task_id == task.id


async def test_a_task_whose_harness_cannot_be_built_fails_alone(
    config, store, runner, workspace, clock
):
    """A harness nobody configured must not stop the queue behind it.

    Reproduced from the session: the brain names a harness freely, and the
    CLI's adapter lookup raised out of `schedule()`, which sits outside the
    per-agent isolation - so every tick failed and nothing else ever started.
    """
    from buddy.harnesses.base import HarnessError

    def adapter_for(name: str):
        if name != "claude_code":
            raise HarnessError(f"no [harness.{name}] block in config.toml")
        return ClaudeCodeAdapter(config.harness(name))

    manager = AgentManager(config, store, runner, workspace, adapter_for=adapter_for, now=clock)
    doomed = make_task(store, harness="codex", priority=1)
    fine = make_task(store, priority=3)
    store.create_task(doomed, TaskState.QUEUED)
    events = await manager.submit(fine)

    assert [e.task_id for e in kinds(events, TaskStarted)] == [fine.id]
    failed = [e for e in kinds(events, TaskFinished) if e.task_id == doomed.id]
    assert len(failed) == 1 and failed[0].outcome is RunOutcome.ERROR
    assert "harness.codex" in failed[0].summary
    assert store.get_task_state(doomed.id) is TaskState.ERROR
    # Refused before anything was created for it.
    assert not (workspace.root / doomed.project / doomed.id).exists()
    # And it stays refused: the next tick does not try again.
    assert not kinds(await manager.tick(), TaskFinished)


async def test_a_run_that_cannot_be_prepared_fails_alone(tmp_path, store, runner, workspace, clock):
    """A `${keychain:NAME}` nobody stored raised from `prepare_run` - after the
    worktree existed and outside any isolation - so the scheduler retried it
    on every tick and started nothing else."""
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path / "webapp"}"\n'
        '[harness.claude_code]\ncommand = "claude -p < {prompt_path}"\n'
        "[harness.claude_code.env]\n"
        'ANTHROPIC_API_KEY = "${keychain:BUDDY_TEST_SECRET_NOBODY_STORED}"\n'
        '[harness.opencode]\ncommand = "opencode run -- \\"$(cat {prompt_path})\\""\n'
    )
    config = Config.load(home=tmp_path)
    from buddy.harnesses.opencode import OpenCodeAdapter

    factories = {"claude_code": ClaudeCodeAdapter, "opencode": OpenCodeAdapter}
    manager = AgentManager(
        config,
        store,
        runner,
        workspace,
        adapter_for=lambda name: factories[name](config.harness(name)),
        now=clock,
    )
    doomed = make_task(store, priority=1)
    store.create_task(doomed, TaskState.QUEUED)
    fine = make_task(store, harness="opencode", priority=3)
    events = await manager.submit(fine)

    assert [e.task_id for e in kinds(events, TaskStarted)] == [fine.id]
    failed = [e for e in kinds(events, TaskFinished) if e.task_id == doomed.id]
    assert failed and "BUDDY_TEST_SECRET_NOBODY_STORED" in failed[0].summary
    assert store.get_task_state(doomed.id) is TaskState.ERROR


async def test_an_agent_nobody_named_is_named_from_its_title(manager, store, runner):
    for _ in range(3):
        await manager.submit(make_task(store, title="Add rate limiting to the API"))
    assert [agent for agent, _, _ in runner.spawned] == [
        "add-rate-limiting",
        "add-rate-limiting-2",
        "add-rate-limiting-3",
    ]


async def test_a_name_is_checked_before_anything_is_saved(manager, store, runner):
    await manager.submit(make_task(store, agent="scout"))

    taken = make_task(store, agent="Scout")
    with pytest.raises(AgentNameError, match="already"):
        await manager.submit(taken)
    with pytest.raises(AgentNameError, match="letters, digits"):
        await manager.submit(make_task(store, agent="-rf"))

    assert store.get_task(taken.id) is None
    assert len(runner.spawned) == 1


async def test_a_spoken_name_with_spaces_becomes_a_usable_one(manager, store, runner):
    task = make_task(store, agent="code reviewer")
    await manager.submit(task)
    assert task.agent == "code-reviewer"
    assert runner.spawned[0][0] == "code-reviewer"


async def test_priority_order_then_fifo(manager, store, runner):
    """Lowest priority number first; equal priorities keep their order."""
    store.create_task(make_task(store, priority=5, title="whenever"))
    store.create_task(make_task(store, priority=1, title="urgent"))
    store.create_task(make_task(store, priority=5, title="also whenever"))
    store.create_task(make_task(store, priority=3, title="middle"))

    await manager.schedule()

    titles = [store.get_task(task_id).title for _, task_id, _ in runner.spawned]
    assert titles == ["urgent", "middle", "whenever", "also whenever"]


async def test_there_is_no_limit_unless_one_is_set(manager, store, runner):
    for _ in range(12):
        await manager.submit(make_task(store))
    assert len(runner.spawned) == 12
    assert store.tasks_in_state(TaskState.QUEUED) == []
    assert manager.max_concurrent is None


async def test_max_concurrent_is_honoured(store, runner, workspace, clock, tmp_path):
    manager = limited_manager(tmp_path, store, runner, workspace, clock, 2)
    for _ in range(4):
        await manager.submit(make_task(store))
    assert len(runner.spawned) == 2
    assert len(store.tasks_in_state(TaskState.QUEUED)) == 2


async def test_a_finished_agent_is_retired_and_its_name_is_free(manager, store, runner):
    first = make_task(store, agent="scout")
    await manager.submit(first)
    runner.finish("scout", 0)
    await manager.tick()

    assert runner.closed == ["scout"], "its window goes when its attempt ends"
    assert manager.agents() == []
    again = make_task(store, agent="scout")
    await manager.submit(again)
    assert store.get_agent("scout").task_id == again.id


# -- dependencies ---------------------------------------------------


async def test_a_dependent_task_waits_for_its_dependency(manager, store, runner):
    first = make_task(store)
    second = make_task(store, depends_on=[first.id])
    await manager.submit(first)
    await manager.submit(second)

    assert len(runner.spawned) == 1
    assert store.get_task_state(second.id) is TaskState.QUEUED

    runner.finish(first.agent, 0)
    await manager.tick()

    assert len(runner.spawned) == 2
    assert runner.spawned[1][1] == second.id


async def test_merge_required_makes_the_dependent_wait_for_the_merge(manager, store, runner):
    """DONE is not enough when the dependency declared merge_required."""
    first = make_task(store, merge_required=True)
    second = make_task(store, depends_on=[first.id])
    await manager.submit(first)
    await manager.submit(second)
    runner.finish(first.agent, 0)
    await manager.tick()

    assert len(runner.spawned) == 1
    assert "to be merged" in manager.dependency_block(store.get_task(second.id))

    store.set_task_state(first.id, TaskState.MERGED)
    await manager.schedule()
    assert len(runner.spawned) == 2


async def test_a_failed_dependency_is_reported_once(manager, store, runner):
    first = make_task(store)
    second = make_task(store, depends_on=[first.id])
    await manager.submit(first)
    await manager.submit(second)
    runner.finish(first.agent, 1)

    events = await manager.tick()
    blocked = kinds(events, TaskBlocked)
    assert len(blocked) == 1
    assert first.id in blocked[0].reason
    assert kinds(await manager.tick(), TaskBlocked) == []


async def test_a_dependency_on_an_unknown_task_blocks(manager, store):
    task = make_task(store, depends_on=["t-9999"])
    await manager.submit(task)
    assert "does not exist" in manager.dependency_block(task)


# -- non-git projects run one agent at a time -----------------------


async def test_two_tasks_in_a_non_git_project_do_not_run_at_once(manager, store, runner, workspace):
    workspace.non_git.add("notes")
    first = make_task(store, project="notes")
    second = make_task(store, project="notes")
    await manager.submit(first)
    await manager.submit(second)

    assert len(runner.spawned) == 1
    runner.finish(first.agent, 0)
    await manager.tick()
    assert len(runner.spawned) == 2


async def test_a_locked_project_does_not_block_other_work(manager, store, runner, workspace):
    workspace.non_git.add("notes")
    await manager.submit(make_task(store, project="notes"))
    await manager.submit(make_task(store, project="notes"))
    await manager.submit(make_task(store, project="webapp"))

    projects = [store.get_task(task_id).project for _, task_id, _ in runner.spawned]
    assert projects == ["notes", "webapp"]


# -- finishing ------------------------------------------------------


async def test_a_dead_pane_is_finalized_with_its_exit_code(manager, store, runner, config):
    task = make_task(store)
    await manager.submit(task)
    runner.finish(task.agent, 0)

    events = await manager.tick()

    finished = kinds(events, TaskFinished)[0]
    assert finished.outcome is RunOutcome.DONE
    assert finished.exit_code == 0
    assert store.get_task_state(task.id) is TaskState.DONE
    assert store.get_run(task.id, 1).outcome is RunOutcome.DONE
    assert config.paths.result_file(task.id).exists()


async def test_a_nonzero_exit_is_an_error(manager, store, runner):
    task = make_task(store)
    await manager.submit(task)
    runner.finish(task.agent, 3)
    events = await manager.tick()
    assert kinds(events, TaskFinished)[0].outcome is RunOutcome.ERROR
    assert store.get_task_state(task.id) is TaskState.ERROR


async def test_finishing_checkpoints_the_worktree(manager, store, runner, workspace):
    task = make_task(store)
    await manager.submit(task)
    runner.finish(task.agent, 0)
    await manager.tick()
    assert workspace.checkpoints == [(task.id, 1, RunOutcome.DONE)]


# -- health ---------------------------------------------------------


def write_log(store: Store, config: Config, task_id: str, text: str, attempt: int = 1) -> None:
    path = config.paths.log_file(task_id, attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


async def test_a_prompt_that_slipped_past_auto_approve_becomes_waiting_input(
    manager, store, runner, config, clock
):
    task = make_task(store)
    await manager.submit(task)
    write_log(store, config, task.id, "Overwrite the config? (y/n)\n")
    clock.advance(seconds=5)

    events = await manager.tick()

    health = kinds(events, AgentHealthChanged)[0]
    assert health.status is AgentStatus.WAITING_INPUT
    assert "attach" in health.detail
    assert runner.killed == []  # Buddy never answers for the harness


async def test_silence_becomes_stalled_but_nothing_is_killed(manager, store, runner, config, clock):
    task = make_task(store, stall_timeout=timedelta(minutes=10))
    await manager.submit(task)
    write_log(store, config, task.id, "compiling...\n")
    await manager.tick()

    clock.advance(minutes=11)
    events = await manager.tick()

    assert kinds(events, AgentHealthChanged)[0].status is AgentStatus.STALLED
    assert runner.killed == []
    assert store.get_task_state(task.id) is TaskState.RUNNING


async def test_a_harness_retrying_forever_is_stalled_not_running(
    tmp_path, store, runner, workspace, clock
):
    """Codex prints "Reconnecting... waiting for network" every few seconds
    when its API is unreachable, and never exits (measured, 0.154.0).

    The log grows the whole time, so before `is_progress` the agent said
    RUNNING until `max_runtime` - two hours of an agent doing nothing.
    """
    from buddy.harnesses.codex import CodexAdapter

    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path / "webapp"}"\n'
        '[harness.codex]\ncommand = "codex exec --json - < {prompt_path}"\n'
    )
    config = Config.load(home=tmp_path)
    manager = AgentManager(
        config,
        store,
        runner,
        workspace,
        adapter_for=lambda name: CodexAdapter(config.harness(name)),
        now=clock,
    )
    task = make_task(store, harness="codex", stall_timeout=timedelta(minutes=10))
    await manager.submit(task)
    fixture = Path(__file__).parent / "fixtures" / "harness_logs" / "codex_unreachable.jsonl"
    log = fixture.read_text()
    write_log(store, config, task.id, log)
    await manager.tick()

    retry = '{"type":"error","message":"Reconnecting... waiting for network (Connection failed)"}\n'
    events = []
    for _ in range(12):
        clock.advance(minutes=1)
        log += retry
        write_log(store, config, task.id, log)
        events.extend(await manager.tick())

    # Health is reported on change, so exactly one transition, to STALLED.
    assert [e.status for e in kinds(events, AgentHealthChanged)] == [AgentStatus.STALLED]
    assert "retrying" in store.get_agent(task.agent).last_output

    # A real step of work is progress again.
    log += '{"type":"item.started","item":{"type":"command_execution","command":"ls"}}\n'
    write_log(store, config, task.id, log)
    assert kinds(await manager.tick(), AgentHealthChanged)[0].status is AgentStatus.RUNNING


async def test_output_clears_a_stall(manager, store, runner, config, clock):
    task = make_task(store, stall_timeout=timedelta(minutes=10))
    await manager.submit(task)
    write_log(store, config, task.id, "start\n")
    await manager.tick()
    clock.advance(minutes=11)
    await manager.tick()

    write_log(store, config, task.id, "start\nmore output, it was just slow\n")
    events = await manager.tick()
    assert kinds(events, AgentHealthChanged)[0].status is AgentStatus.RUNNING


async def test_health_is_reported_on_change_not_every_tick(manager, store, runner, config, clock):
    task = make_task(store)
    await manager.submit(task)
    write_log(store, config, task.id, "(y/n)\n")
    assert kinds(await manager.tick(), AgentHealthChanged)
    assert not kinds(await manager.tick(), AgentHealthChanged)


async def test_a_timeout_is_killed_checkpointed_and_requeued_once(
    manager, store, runner, workspace, clock
):
    task = make_task(store, max_runtime=timedelta(hours=2))
    await manager.submit(task)
    clock.advance(hours=3)

    events = await manager.tick()

    requeued = kinds(events, TaskRequeued)[0]
    assert requeued.reason is RunOutcome.TIMEOUT
    assert requeued.next_attempt == 2
    assert task.agent in runner.killed
    assert workspace.checkpoints == [(task.id, 1, RunOutcome.TIMEOUT)]
    # Requeued, and picked straight back up under the same name.
    assert runner.spawned[-1] == (task.agent, task.id, 2)
    assert store.get_task(task.id).attempt == 2


async def test_a_second_timeout_is_an_error(manager, store, runner, clock):
    task = make_task(store, max_runtime=timedelta(hours=2))
    await manager.submit(task)
    clock.advance(hours=3)
    await manager.tick()
    clock.advance(hours=3)

    events = await manager.tick()

    finished = kinds(events, TaskFinished)[0]
    assert finished.outcome is RunOutcome.TIMEOUT
    assert "not retried again" in finished.summary
    assert store.get_task_state(task.id) is TaskState.ERROR


async def test_a_retry_resumes_in_the_same_worktree_with_a_resume_note(
    manager, store, runner, config, clock
):
    """A retry picks up the partial work rather than starting over."""
    task = make_task(store, max_runtime=timedelta(hours=1))
    await manager.submit(task)
    clock.advance(hours=2)
    await manager.tick()

    prompt = config.paths.prompt_file(task.id).read_text()
    assert "partial work from a previous attempt" in prompt
    assert "git diff basecommit" in prompt


# -- preemption -----------------------------------------------------


async def fill(manager, store, priority: int = 3, count: int = 7) -> list[TaskSpec]:
    tasks = [make_task(store, priority=priority) for _ in range(count)]
    for task in tasks:
        await manager.submit(task)
    return tasks


async def test_an_urgent_task_proposes_a_preemption_but_never_takes_one(capped, store, runner):
    running = await fill(capped, store, priority=4)
    urgent = make_task(store, priority=1, title="urgent")

    events = await capped.submit(urgent)

    proposals = kinds(events, PreemptionProposal)
    assert len(proposals) == 1
    assert proposals[0].incoming_task_id == urgent.id
    assert proposals[0].victim_agent in {task.agent for task in running}
    # Proposed only: nothing was killed.
    assert runner.killed == []
    assert store.get_task_state(urgent.id) is TaskState.QUEUED


async def test_with_no_limit_an_urgent_task_simply_starts(manager, store, runner):
    await fill(manager, store, priority=4)
    urgent = make_task(store, priority=1, title="urgent")

    events = await manager.submit(urgent)

    assert kinds(events, PreemptionProposal) == []
    assert [e.task_id for e in kinds(events, TaskStarted)] == [urgent.id]


async def test_no_preemption_is_proposed_for_a_task_that_still_could_not_start(
    config, store, runner, workspace, clock, tmp_path
):
    """`notes` is not a git repo, so it runs one agent at a time.
    Killing the webapp task makes room the notes task still cannot use -
    and the scheduler would put the killed task straight back."""
    workspace.non_git.add("notes")
    manager = limited_manager(tmp_path, store, runner, workspace, clock, 2)
    await manager.submit(make_task(store, project="notes", priority=1, title="notes A1"))
    await manager.submit(make_task(store, project="webapp", priority=5, title="webapp B1"))

    events = await manager.submit(make_task(store, project="notes", priority=2, title="notes A2"))

    assert kinds(events, PreemptionProposal) == []


async def test_the_victim_is_the_task_actually_in_the_way(
    config, store, runner, workspace, clock, tmp_path
):
    """Same shape, but the notes task that holds the lock is the lower priority
    of the two running. Preempting *it* is the only preemption that helps."""
    workspace.non_git.add("notes")
    manager = limited_manager(tmp_path, store, runner, workspace, clock, 2)
    holder = make_task(store, project="notes", priority=4, title="notes holder")
    await manager.submit(holder)
    await manager.submit(make_task(store, project="webapp", priority=5, title="webapp B1"))
    urgent = make_task(store, project="notes", priority=1, title="notes urgent")

    [proposal] = kinds(await manager.submit(urgent), PreemptionProposal)
    assert proposal.victim_task_id == holder.id

    events = await manager.accept_preemption(proposal.proposal_id)
    started = kinds(events, TaskStarted)
    assert [e.task_id for e in started] == [urgent.id]
    assert runner.killed == [proposal.victim_agent]


async def test_a_non_git_projects_lock_is_preempted_even_without_a_limit(
    manager, store, runner, workspace
):
    """No cap, so room is never the problem - but a non-git project still
    runs one agent at a time, and that is worth a question."""
    workspace.non_git.add("notes")
    holder = make_task(store, project="notes", priority=4, title="notes holder")
    await manager.submit(holder)

    urgent = make_task(store, project="notes", priority=1, title="notes urgent")
    [proposal] = kinds(await manager.submit(urgent), PreemptionProposal)

    assert proposal.victim_agent == holder.agent


async def test_accepting_a_stale_preemption_says_so_and_stops_nothing(
    capped, store, runner, config
):
    """The victim finished between the proposal and the yes; the
    acceptance used to pass silently and report success."""
    from buddy.manager import StalePreemption

    await fill(capped, store, priority=4)
    [proposal] = kinds(await capped.submit(make_task(store, priority=1)), PreemptionProposal)

    runner.finish(proposal.victim_agent, 0)
    await capped.tick()  # the victim finishes; the urgent task takes its place

    with pytest.raises(StalePreemption, match="no longer"):
        await capped.accept_preemption(proposal.proposal_id)
    assert runner.killed == []


async def test_no_proposal_when_the_newcomer_does_not_outrank_anything(capped, store):
    await fill(capped, store, priority=1)
    events = await capped.submit(make_task(store, priority=2))
    assert kinds(events, PreemptionProposal) == []


async def test_the_lowest_priority_running_task_is_the_victim(capped, store):
    running = [make_task(store, priority=priority) for priority in (1, 2, 5, 3, 1, 1, 2)]
    for task in running:
        await capped.submit(task)
    events = await capped.submit(make_task(store, priority=1, title="urgent"))
    proposal = kinds(events, PreemptionProposal)[0]
    assert proposal.victim_agent == running[2].agent  # the priority-5 one


async def test_accepting_a_preemption_loses_nothing(capped, store, runner, workspace):
    victims = await fill(capped, store, priority=4)
    urgent = make_task(store, priority=1, title="urgent")
    events = await capped.submit(urgent)
    proposal = kinds(events, PreemptionProposal)[0]
    victim_id = proposal.victim_task_id

    events = await capped.accept_preemption(proposal.proposal_id)

    assert proposal.victim_agent in runner.killed
    assert (victim_id, 1, RunOutcome.PREEMPTED) in workspace.checkpoints
    assert store.get_run(victim_id, 1).outcome is RunOutcome.PREEMPTED
    # Requeued at its original priority, one attempt later, same name.
    requeued = store.get_task(victim_id)
    assert requeued.priority == 4
    assert requeued.attempt == 2
    assert requeued.agent == proposal.victim_agent
    assert store.get_task_state(victim_id) is TaskState.QUEUED
    # The incoming task took the room it made.
    assert kinds(events, TaskStarted)[0].task_id == urgent.id
    assert victims  # the rest are untouched


async def test_declining_a_preemption_leaves_everything_running(capped, store, runner):
    await fill(capped, store, priority=4)
    events = await capped.submit(make_task(store, priority=1))
    proposal = kinds(events, PreemptionProposal)[0]

    capped.decline_preemption(proposal.proposal_id)

    assert runner.killed == []
    assert capped.pending_preemptions == {}


async def test_the_same_task_is_not_proposed_twice(capped, store):
    await fill(capped, store, priority=4)
    await capped.submit(make_task(store, priority=1))
    assert kinds(await capped.tick(), PreemptionProposal) == []


async def test_an_unknown_proposal_is_an_error(manager):
    with pytest.raises(KeyError):
        await manager.accept_preemption("p-nope")


# -- the brain's tools ---------------------------------------------


async def test_list_agents_names_every_running_agent(manager, store, runner, config):
    task = make_task(store, priority=2, title="Fix onboarding", agent="onboarder")
    await manager.submit(task)
    await manager.submit(make_task(store, priority=1, title="urgent", agent="first"))
    write_log(store, config, task.id, "line one\nline two\nline three\n")
    await manager.tick()

    rows = manager.list_agents()
    assert [row["agent"] for row in rows] == ["first", "onboarder"]  # most urgent first
    row = rows[1]
    assert row["task_id"] == task.id
    assert row["title"] == "Fix onboarding"
    assert row["branch"] == f"buddy/{task.id}-slug"
    assert row["priority"] == 2
    assert row["tail"].count("\n") <= 1  # two lines


async def test_get_output_accepts_an_agents_name_or_a_task_id(manager, store, runner, config):
    task = make_task(store, agent="scout")
    await manager.submit(task)
    write_log(store, config, task.id, "\x1b[32mcoloured\x1b[0m output\n")

    assert "coloured output" in manager.get_output("Scout")
    assert "coloured output" in manager.get_output(task.id)
    assert "\x1b" not in manager.get_output("scout")  # ANSI stripped
    assert manager.get_output("nobody") == ""


async def test_reprioritize_reorders_the_queue(capped, store, runner):
    await fill(capped, store)
    low = make_task(store, priority=5, title="low")
    high = make_task(store, priority=4, title="high")
    await capped.submit(low)
    await capped.submit(high)

    capped.reprioritize(low.id, 1)

    assert [task.id for task in capped.ready_tasks()] == [low.id, high.id]


async def test_reprioritizing_a_running_task_only_changes_its_rank(manager, store, runner):
    task = make_task(store)
    await manager.submit(task)
    manager.reprioritize(task.id, 1)
    assert store.get_task(task.id).priority == 1
    assert manager.agent(task.agent).priority == 1
    assert runner.killed == []


async def test_queue_position_reports_how_many_outrank_it(capped, store):
    await fill(capped, store, priority=2)
    first = make_task(store, priority=3)
    second = make_task(store, priority=4)
    await capped.submit(first)
    await capped.submit(second)
    assert capped.queue_position(first.id) == 0
    assert capped.queue_position(second.id) == 1


async def test_kill_marks_the_task_killed_and_does_not_requeue(manager, store, runner, workspace):
    """Kill means kill; pausing is a different thing."""
    task = make_task(store, agent="scout")
    await manager.submit(task)

    events = await manager.kill_agent("SCOUT")

    assert "scout" in runner.killed
    assert (task.id, 1, RunOutcome.KILLED) in workspace.checkpoints
    assert store.get_task_state(task.id) is TaskState.KILLED
    finished = kinds(events, TaskFinished)[0]
    assert finished.outcome is RunOutcome.KILLED and finished.agent == "scout"
    assert store.get_task(task.id).attempt == 1
    assert manager.agents() == []


async def test_killing_makes_room_for_the_queue(capped, store, runner):
    running = await fill(capped, store)
    waiting = make_task(store)
    await capped.submit(waiting)

    await capped.kill_agent(running[0].agent)

    assert runner.spawned[-1][1] == waiting.id


async def test_killing_an_agent_that_is_not_running_says_so(manager):
    with pytest.raises(KeyError, match="no running agent is called nobody"):
        await manager.kill_agent("nobody")


# -- crash and error paths, from the code review ---------------------------


async def test_one_agent_failing_does_not_stop_the_others(manager, store, runner, workspace):
    """A disk-full worktree, a stray index.lock, a repo hook that
    `--no-verify` does not suppress: any of them made one agent's checkpoint
    raise, and the exception left `tick` entirely - so every other agent
    stopped being finalized, stalled-checked and scheduled too."""
    first, second = make_task(store, title="one"), make_task(store, title="two")
    await manager.submit(first)
    await manager.submit(second)
    runner.finish(first.agent, 0)
    runner.finish(second.agent, 0)

    failed: list[str] = []
    real = workspace.checkpoint

    async def one_bad_worktree(run, outcome=None):
        if run.task_id == first.id:
            failed.append(run.task_id)
            raise OSError("No space left on device")
        return await real(run, outcome)

    workspace.checkpoint = one_bad_worktree
    events = await manager.tick()

    assert failed == [first.id]
    finished = [e.task_id for e in events if isinstance(e, TaskFinished)]
    assert second.id in finished, "the healthy agent was never finalized"
    # And the failure is visible rather than silent.
    assert any(isinstance(e, AgentHealthChanged) and "tick failed" in e.detail for e in events)


def fresh_manager(config, store, runner, workspace, clock) -> AgentManager:
    """Another Buddy process, starting up over the same state."""
    return AgentManager(
        config,
        store,
        runner,
        workspace,
        adapter_for=lambda name: ClaudeCodeAdapter(config.harness(name)),
        now=clock,
    )


async def test_a_crash_after_a_preemption_is_recorded_requeues_the_task(
    manager, config, store, runner, workspace, clock
):
    """The attempt's outcome was written and the process died before its
    agent was retired. The agent must not be held forever: the pane is gone
    or dead, so nothing finalizes it, and `_orphaned_runs` cannot see it
    because the outcome is already there."""
    task = make_task(store, title="interrupted by a crash")
    await manager.submit(task)

    # The first half of stopping it, and then nothing.
    run = store.get_run(task.id, 1)
    await runner.kill(task.agent)
    run.ended_at = clock()
    run.outcome = RunOutcome.PREEMPTED
    store.save_run(run)  # ... and the process dies here
    assert store.get_agent(task.agent) is not None, "precondition: still claimed"

    events = await fresh_manager(config, store, runner, workspace, clock).reconcile()

    assert store.get_agent(task.agent) is None, "the agent is still held"
    assert store.get_task_state(task.id) is TaskState.QUEUED
    assert any(isinstance(e, TaskRequeued) for e in events)


async def test_a_killed_run_interrupted_mid_write_stays_killed(
    manager, config, store, runner, workspace, clock
):
    """Kill means kill: recovering the bookkeeping must not quietly
    turn one back into a queued task."""
    task = make_task(store, title="killed on purpose")
    await manager.submit(task)

    run = store.get_run(task.id, 1)
    await runner.kill(task.agent)
    run.ended_at = clock()
    run.outcome = RunOutcome.KILLED
    store.save_run(run)

    await fresh_manager(config, store, runner, workspace, clock).reconcile()
    assert store.get_task_state(task.id) is TaskState.KILLED
    assert store.get_agent(task.agent) is None


async def test_a_run_finalized_before_a_crash_keeps_its_verdict(
    manager, config, store, runner, workspace, clock
):
    """Finalizing writes the outcome and the task's state, then retires the
    agent. A crash in between must not turn a finished task into an error
    or back into a queued one."""
    task = make_task(store, title="finished just before the crash")
    await manager.submit(task)
    run = store.get_run(task.id, 1)
    run.outcome = RunOutcome.DONE
    run.exit_code = 0
    store.save_run(run)
    store.set_task_state(task.id, TaskState.DONE)
    runner.finish(task.agent, 0)

    events = await fresh_manager(config, store, runner, workspace, clock).reconcile()

    assert store.get_task_state(task.id) is TaskState.DONE
    assert store.get_agent(task.agent) is None
    assert task.agent in runner.closed
    assert not kinds(events, TaskRequeued)


async def test_finalizing_never_walks_a_merged_task_backwards(manager, store, runner):
    """Found on a live machine: `buddy merge` from a second terminal while no
    session was ticking, then a later tick noticing the pane had died."""
    task = make_task(store, title="already merged")
    await manager.submit(task)
    store.set_task_state(task.id, TaskState.MERGED)

    runner.finish(task.agent, 0)
    await manager.tick()

    assert store.get_task_state(task.id) is TaskState.MERGED


# -- the discard sweep -----------------------------------------------------


async def test_discarded_branches_are_swept_hourly_and_one_failure_stays_contained(
    manager, store, workspace, clock
):
    from buddy.models import BranchDeleted

    doomed, stubborn = make_task(store), make_task(store)
    for task in (doomed, stubborn):
        store.create_task(task, TaskState.QUEUED)
        store.set_task_state(task.id, TaskState.DISCARDED)
    workspace.prune_explodes.add(stubborn.id)

    events = await manager.tick()
    assert [e.task_id for e in kinds(events, BranchDeleted)] == [doomed.id]
    assert sorted(workspace.pruned) == sorted([doomed.id, stubborn.id])
    assert "webapp" in workspace.worktree_prunes

    clock.advance(minutes=30)
    await manager.tick()
    assert len(workspace.pruned) == 2, "not again within the hour"

    clock.advance(minutes=31)
    await manager.tick()
    assert len(workspace.pruned) == 4


async def test_what_a_checkpoint_kept_out_of_git_is_said_when_the_task_ends(
    manager, store, runner, workspace, config
):
    task = make_task(store)
    await manager.submit(task)
    workspace.withheld = {(task.id, 1): [".env: holds secrets by its nature"]}
    runner.finish(task.agent, 0)

    [finished] = kinds(await manager.tick(), TaskFinished)

    assert "Kept out of the commit, left in the worktree: .env" in finished.summary
    result = json.loads(config.paths.result_file(task.id).read_text())
    assert result["withheld_from_commit"] == [".env: holds secrets by its nature"]


# -- nothing an attempt resolved outlives it -------------------------------


def _script(config: Config, task: TaskSpec) -> Path:
    return config.paths.run_script(task.id)


async def test_a_kill_removes_the_run_script(manager, store, config):
    """`run.sh` holds the attempt's resolved API keys. Only a pane dying on its
    own used to remove it; a kill left the keys on disk indefinitely - found
    by running a real task in real tmux and killing it."""
    task = make_task(store)
    await manager.submit(task)
    assert _script(config, task).exists()

    await manager.kill_agent(task.agent)

    assert not _script(config, task).exists()


async def test_a_preemption_removes_the_victims_run_script(capped, store, config):
    await fill(capped, store, priority=4)
    urgent = make_task(store, priority=1, title="urgent")
    proposal = kinds(await capped.submit(urgent), PreemptionProposal)[0]
    victim = store.get_task(proposal.victim_task_id)

    await capped.accept_preemption(proposal.proposal_id)

    assert not _script(config, victim).exists()
    assert _script(config, urgent).exists(), "the task it made room for is running"


async def test_a_timeout_replaces_the_run_script_rather_than_keeping_it(
    manager, store, config, clock
):
    task = make_task(store, max_runtime=timedelta(hours=2))
    await manager.submit(task)
    clock.advance(hours=3)
    await manager.tick()
    # Requeued and restarted at once: the script there now is attempt 2's.
    assert "ATTEMPT=2" in _script(config, task).read_text()

    clock.advance(hours=3)
    await manager.tick()

    assert not _script(config, task).exists(), "a second timeout ends it for good"


async def test_shutdown_removes_every_run_script(manager, store, config):
    tasks = [make_task(store) for _ in range(3)]
    for task in tasks:
        await manager.submit(task)

    await manager.shutdown()

    assert [task.id for task in tasks if _script(config, task).exists()] == []


async def test_finishing_removes_the_run_script_even_when_the_checkpoint_fails(
    manager, store, runner, workspace, config
):
    task = make_task(store)
    await manager.submit(task)
    runner.finish(task.agent, 0)

    async def full_disk(run, outcome=None):
        raise OSError("No space left on device")

    workspace.checkpoint = full_disk
    await manager.tick()

    assert not _script(config, task).exists()


# -- one task that cannot start never holds up the queue --------------------


async def test_a_task_whose_checkout_cannot_be_made_fails_alone(manager, store, runner, workspace):
    """Reproduced against real git: a project whose base branch has no commit
    made `Workspace.create` raise out of `schedule`, on every tick. The same
    task stayed first in line and failed again each second, and the healthy
    task behind it never started."""
    broken = make_task(store, priority=1, title="broken")
    healthy = make_task(store, priority=2, title="healthy")
    real = workspace.create

    async def create(task):
        if task.id == broken.id:
            raise WorkspaceError("git rev-parse main failed: unknown revision")
        return await real(task)

    workspace.create = create
    store.create_task(broken, TaskState.QUEUED)
    store.create_task(healthy, TaskState.QUEUED)

    events = await manager.tick()

    assert store.get_task_state(healthy.id) is TaskState.RUNNING
    assert store.get_task_state(broken.id) is TaskState.ERROR
    [failed] = [e for e in kinds(events, TaskFinished) if e.task_id == broken.id]
    assert "could not create its checkout" in failed.summary
    assert "unknown revision" in failed.summary


# -- merging and discarding wait for the agent ------------------------------


async def test_a_running_task_is_still_working(manager, store, runner):
    task = make_task(store, agent="scout")
    await manager.submit(task)

    reason = manager.still_working(task.id)

    assert reason is not None
    assert "as scout" in reason and "buddy kill scout" in reason

    runner.finish("scout", 0)
    await manager.tick()
    assert manager.still_working(task.id) is None


# -- a dead pane without an exit status --------------------------------------


@pytest.mark.parametrize(("code", "state"), [(0, TaskState.DONE), (3, TaskState.ERROR)])
async def test_a_dead_pane_with_no_status_is_read_from_the_log(
    manager, store, runner, config, code, state
):
    """tmux has no exit status for a process killed by a signal, and on Linux
    a task that had finished was recorded as an error with exit code -1. The
    wrapper's DONE line carries the harness's exit code for exactly this."""
    from buddy.logs import DONE_SENTINEL

    task = make_task(store)
    await manager.submit(task)
    write_log(
        store, config, task.id, f"working\n{DONE_SENTINEL} {task.id} {code} 2026-01-01T00:00:00Z\n"
    )
    runner.panes_state[task.agent] = PaneStatus(exists=True, dead=True, exit_code=None, signal=1)

    [finished] = kinds(await manager.tick(), TaskFinished)

    assert finished.exit_code == code
    assert store.get_task_state(task.id) is state


async def test_a_pane_killed_by_a_signal_before_it_finished_reports_the_signal(
    manager, store, runner, config
):
    task = make_task(store)
    await manager.submit(task)
    write_log(store, config, task.id, "working, and then nothing\n")
    runner.panes_state[task.agent] = PaneStatus(exists=True, dead=True, exit_code=None, signal=9)

    [finished] = kinds(await manager.tick(), TaskFinished)

    assert finished.exit_code == 137, "128 + SIGKILL, as a shell would say"
    assert store.get_task_state(task.id) is TaskState.ERROR


async def test_a_dead_pane_waits_briefly_for_tmux_to_say_how_it_ended(
    manager, store, runner, config, clock
):
    """Measured on Linux: tmux marks a pane dead a moment before it has
    reaped the process, so for one poll there is no status, no signal and -
    if the log pipe is behind - no DONE line either. Finalizing then turned
    a success into an error; it waits instead, but not forever."""
    task = make_task(store)
    await manager.submit(task)
    write_log(store, config, task.id, "working\n")
    runner.panes_state[task.agent] = PaneStatus(exists=True, dead=True, exit_code=None)

    assert kinds(await manager.tick(), TaskFinished) == []
    assert store.get_task_state(task.id) is TaskState.RUNNING

    # The next poll has it.
    runner.panes_state[task.agent] = PaneStatus(exists=True, dead=True, exit_code=0)
    [finished] = kinds(await manager.tick(), TaskFinished)
    assert finished.exit_code == 0
    assert store.get_task_state(task.id) is TaskState.DONE


async def test_a_dead_pane_that_never_says_how_it_ended_is_still_finalized(
    manager, store, runner, config, clock
):
    task = make_task(store)
    await manager.submit(task)
    write_log(store, config, task.id, "working\n")
    runner.panes_state[task.agent] = PaneStatus(exists=True, dead=True, exit_code=None)
    await manager.tick()

    clock.advance(seconds=11)
    [finished] = kinds(await manager.tick(), TaskFinished)

    assert finished.exit_code == -1
    assert store.get_task_state(task.id) is TaskState.ERROR


# -- a sandboxed run's git identity ------------------------------------------


def _sandboxed(tmp_path: Path, repo: Path) -> Config:
    """The test config with its harness in a container, and its project a
    real repository that commits as You."""
    repo.mkdir(parents=True)
    for args in (
        ["init", "-q"],
        ["config", "user.name", "You"],
        ["config", "user.email", "you@example.com"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    config = tmp_path / "config.toml"
    config.write_text(
        config.read_text()
        + f"sandbox = \"docker\"\nsandbox_command = '{DEFAULT_SANDBOX_COMMAND}'\n"
    )
    return Config.load(home=tmp_path)


async def test_a_sandboxed_run_commits_as_its_project_does(
    config, tmp_path, store, runner, workspace, clock, manager
):
    """A container has no `~/.gitconfig`, and git there refuses to commit.
    The project's own identity, asked of git by the workspace, reaches the
    run's script - and a run that is not sandboxed never asks."""
    boxed = _sandboxed(tmp_path, config.project("webapp").path)
    sandboxed = AgentManager(
        boxed,
        store,
        runner,
        workspace,
        adapter_for=lambda name: ClaudeCodeAdapter(boxed.harness(name)),
        now=clock,
    )
    task = make_task(store)

    await sandboxed.submit(task)

    script = await asyncio.to_thread(boxed.paths.run_script(task.id).read_text)
    for variable in ("GIT_AUTHOR_NAME=You", "GIT_COMMITTER_EMAIL=you@example.com"):
        assert f"-e {variable}" in script
    assert await manager._commit_identity(task) == {}
