"""A merge conflict, from "that conflicts" to merged, against real git.

The brain and the workspace are real; only tmux is the manager tests' fake,
because what runs the agent is not what is under test here. The agent's own
part - `git merge main`, resolve, commit - is done by the test, in the
worktree Buddy created for it, exactly as a harness would.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from buddy.brain import Brain
from buddy.config import Config
from buddy.harnesses.claude_code import ClaudeCodeAdapter
from buddy.manager import AgentManager
from buddy.models import TaskSpec, TaskState
from buddy.providers.base import ToolCall
from buddy.state import Store
from buddy.workspace import Workspace, branch_name
from tests.test_context import ScriptedProvider
from tests.test_manager import Clock, FakeRunner


def git(cwd: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=T", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "webapp"
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    (path / "README.md").write_text("hello\n")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "initial")
    return path


@pytest.fixture
def world(tmp_path: Path, repo: Path):
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{repo}"\n'
        '[harness.claude_code]\ncommand = "claude -p < {prompt_path}"\n'
    )
    config = Config.load(home=tmp_path)
    with Store(config.paths.db) as store:
        store.ensure_slots()
        workspace = Workspace(config)
        manager = AgentManager(
            config,
            store,
            FakeRunner(),
            workspace,
            adapter_for=lambda name: ClaudeCodeAdapter(config.harness(name)),
            now=Clock(),
        )
        brain = Brain(
            config, store, manager, workspace, ScriptedProvider(), confirm=lambda p, t: True
        )
        yield config, store, workspace, brain


async def finished_task(store, workspace, task_id: str, title: str, name: str, text: str):
    task = TaskSpec(
        id=task_id,
        title=title,
        brief=f"# Task: {title}",
        harness="claude_code",
        project="webapp",
        priority=3,
    )
    store.create_task(task, TaskState.DONE)
    checkout = await workspace.create(task)
    (checkout.path / name).write_text(text)
    git(checkout.path, "add", "-A")
    git(checkout.path, "commit", "-qm", title)
    return task


async def say(brain: Brain, tool: str, **arguments) -> str:
    return (await brain.call_tool(ToolCall("c", tool, arguments))).text


async def test_a_conflict_is_resolved_by_an_agent_and_lands_with_the_original_work(world, repo):
    config, store, workspace, brain = world
    limits = await finished_task(
        store, workspace, "t-0001", "Add rate limits", "README.md", "agent: rate limits\n"
    )
    other = await finished_task(store, workspace, "t-0003", "Unrelated docs", "DOCS.md", "docs\n")
    (repo / "README.md").write_text("you: a different change\n")
    git(repo, "commit", "-qam", "your change on main")

    # 1. The merge conflicts: reported, nothing touched, and the fix offered.
    said = await say(brain, "propose_merge", task_id="t-0001")
    assert "conflict" in said and "resolve_conflict" in said
    assert store.get_task_state("t-0001") is TaskState.DONE
    assert git(repo, "status", "--porcelain") == ""

    # 2. "Have someone fix it."
    said = await say(brain, "resolve_conflict", task_id="t-0001")
    [fix] = [t for t in store.recent_tasks() if t.resolves == "t-0001"]
    assert fix.start_from == branch_name(limits)
    assert "git merge main" in fix.brief and "Do not rebase" in fix.brief
    assert "Add rate limits" in fix.brief
    fix_tree = config.worktree_path("webapp", fix.id)
    tip = git(repo, "rev-parse", branch_name(limits)).strip()
    # The fix starts on the conflicting work, not on main.
    assert git(fix_tree, "rev-parse", "HEAD").strip() == tip
    # Asking twice does not start two agents on the same conflict.
    assert fix.id in await say(brain, "resolve_conflict", task_id="t-0001")
    assert len([t for t in store.recent_tasks() if t.resolves == "t-0001"]) == 1

    # 3. While it is pending, nothing else merges into the project.
    held = await say(brain, "propose_merge", task_id=other.id)
    assert "not merged" in held and fix.id in held
    assert store.get_task_state(other.id) is TaskState.DONE

    # 4. The agent's part, in the worktree Buddy made for it.
    git(fix_tree, "merge", "main", check=False)  # conflicts, as it must
    (fix_tree / "README.md").write_text("you: a different change\nagent: rate limits\n")
    git(fix_tree, "add", "-A")
    git(fix_tree, "commit", "-qm", "Merge main into the rate limits work (kept both)")
    store.set_task_state(fix.id, TaskState.DONE)

    # 5. The fix lands; the original counts as merged; the hold is gone.
    said = await say(brain, "propose_merge", task_id=fix.id)
    assert "merged" in said and "t-0001" in said
    assert (repo / "README.md").read_text() == "you: a different change\nagent: rate limits\n"
    assert store.get_task_state("t-0001") is TaskState.MERGED
    assert store.get_task_state(fix.id) is TaskState.MERGED
    assert "merged" in await say(brain, "propose_merge", task_id=other.id)
    assert (repo / "DOCS.md").exists()


async def test_resolving_is_withheld_while_brainstorming(world):
    _, store, workspace, brain = world
    await finished_task(store, workspace, "t-0001", "x", "a.txt", "a\n")
    brain.start_brainstorm()
    assert "resolve_conflict" not in {tool.name for tool in brain.tool_defs()}
    assert (await say(brain, "resolve_conflict", task_id="t-0001")).startswith("refused")


async def test_a_branch_with_a_secret_is_not_merged_by_the_brain_and_cannot_be_talked_through(
    world, repo
):
    import random
    import string

    _, store, workspace, brain = world
    rng = random.Random(3)
    lookalike = "sk-" + "proj-" + "".join(rng.choice(string.ascii_letters) for _ in range(60))
    content = f'K = "{lookalike}"\n'
    task = await finished_task(store, workspace, "t-0001", "leaky", "client.py", content)

    said = await say(brain, "propose_merge", task_id="t-0001")

    assert said.startswith("not merged") and "client.py:1: openai key" in said
    assert lookalike not in said, "the refusal must not hand the secret to the model"
    assert not (repo / "client.py").exists()
    assert store.get_task_state(task.id) is TaskState.DONE


async def test_the_brain_will_not_merge_or_discard_a_task_that_is_still_running(world, repo):
    """Both remove the worktree. With the agent still in it, that deleted its
    uncommitted work while it kept running - reproduced through `buddy merge`
    against a live task in real tmux - and a merge took whatever half of the
    work was committed at that moment."""
    _, store, workspace, brain = world
    task = await finished_task(store, workspace, "t-0001", "busy", "a.txt", "half done\n")
    store.set_task_state(task.id, TaskState.RUNNING)
    worktree = workspace.config.worktree_path("webapp", task.id)

    merged = await say(brain, "propose_merge", task_id=task.id)
    discarded = await say(brain, "discard_task", task_id=task.id)

    assert merged.startswith("not merged") and "still running" in merged
    assert discarded.startswith("not discarded") and "still running" in discarded
    assert worktree.exists()
    assert not (repo / "a.txt").exists()
    assert store.get_task_state(task.id) is TaskState.RUNNING
