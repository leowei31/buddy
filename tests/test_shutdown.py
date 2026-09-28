"""Shutting Buddy down: stop the work, keep the work.

The whole feature rests on one property: the **branch** carries the
work and the **worktree** is a disposable checkout of it. So these tests are
mostly about what *survives* - against a real git repo, because "the branch
still has it" is not a claim worth asserting against a fake.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from buddy.config import Config
from buddy.harnesses.claude_code import ClaudeCodeAdapter
from buddy.manager import AgentManager
from buddy.models import RunOutcome, TaskRun, TaskSpec, TaskState
from buddy.state import Store
from buddy.workspace import Workspace, branch_name


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


@pytest.fixture
def config(tmp_path: Path, repo: Path) -> Config:
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{repo}"\nbase_branch = "main"\n'
        '[harness.claude_code]\ncommand = "claude -p < {prompt_path}"\n'
    )
    config = Config.load(home=tmp_path)
    config.paths.ensure()
    return config


@pytest.fixture
def store(config: Config) -> Store:
    with Store(config.paths.db) as opened:
        yield opened


class FakeRunner:
    """tmux, without tmux. Shutdown never inspects a pane."""

    def __init__(self) -> None:
        self.killed: list[str] = []
        self.session_killed = False

    async def panes(self) -> dict:
        return {}

    async def spawn(self, agent: str, run, script: Path) -> None:
        run.log_path.parent.mkdir(parents=True, exist_ok=True)
        run.log_path.touch()

    async def kill(self, agent: str) -> None:
        self.killed.append(agent)

    async def close_window(self, agent: str) -> None: ...

    async def kill_session(self) -> None:
        self.session_killed = True

    async def close_pipe(self, agent: str) -> bool:
        return True


@pytest.fixture
def manager(config: Config, store: Store) -> AgentManager:
    return AgentManager(
        config,
        store,
        FakeRunner(),
        Workspace(config),
        adapter_for=lambda name: ClaudeCodeAdapter(config.harness(name)),
    )


def make_task(store: Store, **overrides) -> TaskSpec:
    defaults = {
        "id": store.next_task_id(),
        "title": "Add a greeting",
        "brief": "# Goal\nSay hello\n",
        "harness": "claude_code",
        "project": "webapp",
        "priority": 2,
    }
    return TaskSpec(**(defaults | overrides))


async def start(manager: AgentManager, store: Store, **overrides) -> TaskSpec:
    task = make_task(store, **overrides)
    await manager.submit(task)
    return task


# -- what a shutdown stops -------------------------------------------------


async def test_a_running_agent_is_stopped_and_its_worktree_removed(manager, store, config):
    task = await start(manager, store)
    worktree = config.worktree_path("webapp", task.id)
    (worktree / "work.txt").write_text("half-finished\n")

    report = await manager.shutdown()

    assert report.stopped and task.id in report.stopped[0]
    assert manager.runner.killed, "the agent's process tree was never killed"
    assert task.id in report.removed
    assert not worktree.exists()


async def test_the_work_survives_on_its_branch(manager, store, config, repo):
    """The whole point. The worktree is a checkout; the branch is the work."""
    task = await start(manager, store)
    worktree = config.worktree_path("webapp", task.id)
    (worktree / "work.txt").write_text("half-finished\n")

    await manager.shutdown()

    assert not worktree.exists()
    branch = branch_name(task)
    assert branch in git(repo, "branch", "--list", branch)
    # And the uncommitted file is on it, because shutdown checkpoints first.
    assert "work.txt" in git(repo, "show", "--name-only", "--format=", branch)


async def test_diff_and_merge_still_work_without_a_worktree(manager, store, config, repo):
    """Keeping task history would be pointless if it referred to nothing."""
    task = await start(manager, store)
    (config.worktree_path("webapp", task.id) / "work.txt").write_text("finished\n")
    await manager.shutdown()

    workspace = Workspace(config)
    assert "work.txt" in await workspace.diff_stat(task)
    await workspace.merge(task)
    assert (repo / "work.txt").read_text() == "finished\n"


# -- what a shutdown keeps -------------------------------------------------


async def test_memory_and_the_conversation_are_untouched(manager, store, config):
    """Whether memory survives a shutdown, answered the only way that counts.

    Memory lives in `state.db`, which a worktree removal never touches - so
    keeping it is not an extra feature, it is what happens unless something
    goes out of its way to destroy it.
    """
    store.remember("prefers base.en for speech")
    store.log_turn("user", "spawn something on webapp")
    store.log_turn("buddy", "scout is on it")
    task = await start(manager, store)

    await manager.shutdown()

    assert [fact["fact"] for fact in store.memories()] == ["prefers base.en for speech"]
    assert [turn["text"] for turn in store.recent_turns(10)][-1] == "scout is on it"
    assert store.search_turns("spawn"), "the FTS index still answers"
    assert store.get_task(task.id) is not None
    assert store.runs_for(task.id), "the attempt's history is still there"


async def test_the_log_of_every_attempt_is_kept(manager, store, config):
    task = await start(manager, store)
    log = store.latest_run(task.id).log_path
    log.write_text("__BUDDY_START__ ...\nwhat the agent said\n")

    await manager.shutdown()

    assert log.exists()
    assert "what the agent said" in log.read_text()


# -- and what happens next -------------------------------------------------


async def test_a_stopped_task_is_requeued_not_killed(manager, store):
    """`kill` means kill; a shutdown is not that. The task goes back on
    the queue at its original priority so it resumes."""
    task = await start(manager, store)
    await manager.shutdown()

    assert store.get_task_state(task.id) is TaskState.QUEUED
    run = store.latest_run(task.id)
    assert run.outcome is RunOutcome.INTERRUPTED
    assert store.get_task(task.id).attempt == 2


async def test_a_resumed_task_rebuilds_its_worktree_from_its_branch(manager, store, config):
    """The bug this feature would have created.

    `create()` used `worktree add -b <branch>`, which fails outright once the
    branch exists - "a branch named ... already exists". After a shutdown that
    keeps branches and removes worktrees, that is *always* the case, so every
    resumed task would have failed to start.
    """
    task = await start(manager, store)
    worktree = config.worktree_path("webapp", task.id)
    (worktree / "work.txt").write_text("half-finished\n")
    await manager.shutdown()
    assert not worktree.exists()

    # Next session: the queue is picked up again.
    await manager.schedule()

    assert worktree.exists(), "the resumed attempt could not rebuild its worktree"
    # And it resumed from the checkpoint, not from an empty base branch.
    assert (worktree / "work.txt").read_text() == "half-finished\n"
    assert store.get_task_state(task.id) is TaskState.RUNNING


async def test_unmerged_work_is_named_rather_than_silently_left(manager, store, config):
    task = await start(manager, store)
    (config.worktree_path("webapp", task.id) / "work.txt").write_text("unmerged\n")

    report = await manager.shutdown()

    assert [task_id for task_id, _ in report.unmerged] == [task.id]
    assert report.unmerged[0][1] == branch_name(task)


async def test_keep_worktrees_stops_the_agents_and_leaves_the_checkouts(manager, store, config):
    task = await start(manager, store)
    worktree = config.worktree_path("webapp", task.id)

    report = await manager.shutdown(remove_worktrees=False)

    assert report.stopped
    assert report.removed == []
    assert worktree.exists()


async def test_shutting_down_an_idle_buddy_does_nothing_alarming(manager, store):
    report = await manager.shutdown()
    assert report.stopped == []
    assert report.removed == []
    assert report.failed == []


async def test_one_worktree_that_will_not_go_does_not_stop_the_rest(
    manager, store, config, monkeypatch
):
    first = await start(manager, store, title="one")
    second = await start(manager, store, title="two")
    real = Workspace.remove_worktree

    async def stubborn(self, task: TaskSpec) -> None:
        if task.id == first.id:
            raise OSError("Device or resource busy")
        await real(self, task)

    monkeypatch.setattr(Workspace, "remove_worktree", stubborn)
    report = await manager.shutdown()

    assert second.id in report.removed
    assert any(task_id == first.id for task_id, _ in report.failed)


# -- the finished-task case ------------------------------------------------


async def test_a_finished_task_keeps_its_branch_and_loses_its_worktree(
    manager, store, config, repo
):
    task = await start(manager, store)
    worktree = config.worktree_path("webapp", task.id)
    (worktree / "done.txt").write_text("all done\n")
    run: TaskRun = store.latest_run(task.id)
    await Workspace(config).checkpoint(run, RunOutcome.DONE)
    store.set_task_state(task.id, TaskState.DONE)
    store.remove_agent(task.agent)  # finished, so no longer running

    await manager.shutdown()

    assert not worktree.exists()
    assert branch_name(task) in git(repo, "branch", "--list", branch_name(task))
    assert store.get_task_state(task.id) is TaskState.DONE, "a finished task is not requeued"
