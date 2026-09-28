# ruff: noqa: F811 - pytest fixtures are imported by name and used as parameters
"""Brainstorming before anything starts (buddy.ideation).

The promise is "nothing starts until you say so", and every test here is
about that promise holding structurally: tools that are not offered, calls
refused at dispatch, a confirmation `trust_mode` cannot switch off, and a
hand-off that launches exactly what was drafted.
"""

from __future__ import annotations

from pathlib import Path

from buddy.brain import Brain, Tier
from buddy.config import Config
from buddy.ideation import ACTING_TOOLS, Brainstorm, Draft
from buddy.models import TaskState
from buddy.providers.base import Role, ToolCall
from buddy.workspace import Workspace
from tests.test_context import (  # noqa: F401 - pytest fixtures, used by name
    ScriptedProvider,
    config,
    manager,
    project,
    provider,
    store,
)


def make_brain(config, store, manager, provider, *, confirm=None, **kwargs) -> Brain:
    return Brain(
        config,
        store,
        manager,
        Workspace(config),
        provider,
        confirm=confirm or (lambda prompt, tier: True),
        **kwargs,
    )


async def call(brain: Brain, name: str, **arguments) -> str:
    return (await brain.call_tool(ToolCall("c", name, arguments))).text


def draft(brain: Brain, **fields):
    return call(brain, "draft_brief", **({"project": "webapp", "title": "t", "goal": "g"} | fields))


# -- the mode --------------------------------------------------------------


async def test_brainstorming_withholds_every_tool_that_starts_or_changes_work(
    config, store, manager, provider
):
    brain = make_brain(config, store, manager, provider)
    acting = {d.name for d in brain.tool_defs()} & ACTING_TOOLS
    assert acting == ACTING_TOOLS, "outside a brainstorm, everything is offered"
    assert "draft_brief" not in {d.name for d in brain.tool_defs()}

    await call(brain, "start_brainstorm")
    offered = {d.name for d in brain.tool_defs()}
    assert not offered & ACTING_TOOLS
    assert {"draft_brief", "drop_draft", "hand_off", "read_file", "grep"} <= offered
    assert "start_brainstorm" not in offered


async def test_a_spawn_called_from_memory_is_refused_mid_brainstorm(
    config, store, manager, provider
):
    """Not offered is not enough on its own: a model can emit a tool call it
    remembers from earlier in the conversation."""
    brain = make_brain(config, store, manager, provider)
    brain.start_brainstorm()
    said = await call(brain, "spawn_agent", project="webapp", title="t", goal="g")
    assert said.startswith("refused: brainstorming is on")
    assert store.recent_tasks() == []


async def test_the_model_is_told_it_is_brainstorming_and_what_is_drafted(
    config, store, manager, provider
):
    brain = make_brain(config, store, manager, ScriptedProvider(["let's think"]))
    brain.start_brainstorm()
    await draft(brain, title="Rate limiting")
    await brain.send("what are the options?")

    request = brain.provider.requests[-1]
    state = request["messages"][-1]
    assert state.role is Role.SYSTEM
    assert "Brainstorming is ON" in state.text
    assert "d1  Rate limiting" in state.text
    # And the system prompt itself did not change, so the cache survives.
    assert "Brainstorming is ON" not in request["system"]


async def test_brainstorm_first_starts_every_session_brainstorming(
    tmp_path, project, store, manager, provider
):
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{project}"\n'
        '[harness.claude_code]\ncommand = "claude -p < {prompt_path}"\n'
        "[brain]\nbrainstorm_first = true\n"
    )
    eager = Config.load(home=tmp_path)
    brain = make_brain(eager, store, manager, provider)
    assert brain.brainstorm.active
    assert not {d.name for d in brain.tool_defs()} & ACTING_TOOLS


# -- drafts ----------------------------------------------------------------


async def test_drafts_are_recorded_revised_and_dropped(config, store, manager, provider):
    brain = make_brain(config, store, manager, provider)
    brain.start_brainstorm()

    assert (await draft(brain, title="Add limits")).startswith("drafted d1")
    assert (await draft(brain, title="Docs", after=["d1"])).startswith("drafted d2")
    assert (await draft(brain, draft_id="d1", title="Add rate limits")).startswith("revised d1")
    assert brain.brainstorm.drafts["d1"].title == "Add rate limits"

    await call(brain, "drop_draft", draft_id="d1")
    assert list(brain.brainstorm.drafts) == ["d2"]
    assert brain.brainstorm.drafts["d2"].after == [], "nothing waits on a draft that is gone"
    assert store.recent_tasks() == []


async def test_a_draft_is_checked_the_way_a_spawn_would_be(config, store, manager, provider):
    brain = make_brain(config, store, manager, provider)
    brain.start_brainstorm()
    assert "not a configured harness" in await draft(brain, harness="codex")
    assert "no draft d9" in await draft(brain, after=["d9"])
    assert "neither a draft id nor a task id" in await draft(brain, after=["tomorrow"])
    assert "not a model identifier" in await draft(brain, model="x; rm -rf ~")
    assert "unknown project" in (await draft(brain, project="nope")).lower()
    assert brain.brainstorm.drafts == {}


async def test_drafts_survive_a_restart(config, store, manager, provider):
    brain = make_brain(config, store, manager, provider)
    brain.start_brainstorm()
    await draft(brain, title="Carried over")

    again = make_brain(config, store, manager, provider)
    assert again.brainstorm.active
    assert again.brainstorm.drafts["d1"].title == "Carried over"


# -- handing off -----------------------------------------------------------


async def test_hand_off_always_asks_even_with_trust_mode_on(
    tmp_path, project, store, manager, provider
):
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{project}"\n'
        '[harness.claude_code]\ncommand = "claude -p < {prompt_path}"\n'
        "[buddy]\ntrust_mode = true\n"
    )
    trusting = Config.load(home=tmp_path)
    asked: list[tuple[str, Tier]] = []
    brain = make_brain(
        trusting, store, manager, provider, confirm=lambda p, t: asked.append((p, t)) or False
    )
    brain.start_brainstorm()
    await draft(brain, title="Add limits")

    said = await call(brain, "hand_off")
    assert asked and asked[0][1] is Tier.ALWAYS
    assert "d1  Add limits" in asked[0][0]
    assert "still brainstorming" in said
    assert store.recent_tasks() == []
    assert brain.brainstorm.active


async def test_hand_off_launches_exactly_what_was_drafted_in_dependency_order(
    config, store, manager, provider
):
    brain = make_brain(config, store, manager, provider)
    brain.start_brainstorm()
    # d1 ends up depending on d2, so launch order must not be draft order.
    await draft(brain, title="Write the docs", definition_of_done="docs merged")
    await draft(brain, title="Add rate limits", goal="429 after 100 req/min", priority=1)
    await draft(
        brain, draft_id="d1", title="Write the docs", after=["d2"], definition_of_done="docs merged"
    )

    said = await call(brain, "hand_off")

    tasks = {task.title: task for task in store.recent_tasks()}
    limits, docs = tasks["Add rate limits"], tasks["Write the docs"]
    assert docs.depends_on == [limits.id]
    assert limits.priority == 1
    assert "429 after 100 req/min" in limits.brief
    assert "docs merged" in docs.brief
    assert "Brainstorming is off" in said
    assert not brain.brainstorm.active and brain.brainstorm.drafts == {}
    assert {d.name for d in brain.tool_defs()} >= ACTING_TOOLS


async def test_a_draft_that_cannot_start_stays_a_draft_and_so_does_what_needs_it(
    config, store, manager, provider
):
    from buddy.harnesses.base import PreflightReport, Requirement

    async def preflight(name: str) -> PreflightReport:
        return PreflightReport(
            harness=name,
            binary="/bin/true",
            requirements=(Requirement("headless", True),),
            authenticated=False,
            auth_detail="run `claude` once to log in",
        )

    brain = make_brain(config, store, manager, provider, preflight=preflight)
    brain.start_brainstorm()
    await draft(brain, title="Base")
    await draft(brain, title="On top", after=["d1"])

    launched, kept = await brain.launch_drafts()

    assert launched == []
    assert any("not signed in" in line for line in kept)
    assert any("waits on d1" in line for line in kept)
    assert list(brain.brainstorm.drafts) == ["d1", "d2"]
    assert brain.brainstorm.active, "brainstorming only ends when every draft is out"
    assert store.recent_tasks() == []


async def test_go_typed_at_the_terminal_is_told_to_the_model(config, store, manager, provider):
    brain = make_brain(config, store, manager, ScriptedProvider(["noted"]))
    brain.start_brainstorm()
    await draft(brain, title="Add limits")
    await brain.launch_drafts()

    await brain.send("thanks")
    first = brain.provider.requests[-1]["messages"][0]
    assert first.role is Role.USER
    assert first.text.startswith("(Buddy: The user typed /go. Started: d1 -> t-")
    assert first.text.endswith("thanks")
    assert store.get_task_state(store.recent_tasks()[0].id) in (TaskState.RUNNING, TaskState.QUEUED)


def test_a_circle_of_drafts_is_reported_not_broken(tmp_path: Path):
    storm = Brainstorm(path=tmp_path / "brainstorm.json")
    storm.drafts = {
        "d1": Draft(id="d1", project="p", title="a", goal="g", after=["d2"]),
        "d2": Draft(id="d2", project="p", title="b", goal="g", after=["d1"]),
    }
    try:
        storm.ordered()
    except ValueError as exc:
        assert "circle" in str(exc)
    else:
        raise AssertionError("a cycle was ordered")


def test_a_corrupt_brainstorm_file_is_not_a_broken_session(tmp_path: Path):
    (tmp_path / "brainstorm.json").write_text("{not json")
    storm = Brainstorm.load(tmp_path)
    assert not storm.active and storm.drafts == {}


async def test_a_launch_that_fails_to_schedule_is_not_launched_twice(
    config, store, manager, provider
):
    """`submit` saves the task, then schedules it. When scheduling raised -
    tmux unreachable - the draft stayed a draft while its task already
    existed, so the next /go created a second copy of the same work."""
    from buddy.tmux_runner import TmuxError

    brain = make_brain(config, store, manager, provider)
    brain.start_brainstorm()
    await draft(brain, title="Only once")

    async def unreachable(*args, **kwargs):
        raise TmuxError("tmux new-session failed (1): no server")

    real = manager.runner.ensure_session
    manager.runner.ensure_session = unreachable
    launched, kept = await brain.launch_drafts()
    manager.runner.ensure_session = real

    assert kept == [] and len(launched) == 1 and "queued" in launched[0]
    assert brain.brainstorm.drafts == {}
    await brain.launch_drafts()
    assert [t.title for t in store.recent_tasks()] == ["Only once"]
