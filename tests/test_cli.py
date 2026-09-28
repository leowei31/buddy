"""The command surface.

The happy path is proven end to end elsewhere; what is worth pinning here
is that the refusals refuse, with a message that says what to do.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import buddy
from buddy.cli import app

runner = CliRunner()


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    """An isolated ~/.buddy, so tests never touch the real one."""
    monkeypatch.setenv("BUDDY_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path / "webapp"}"\n'
        '[harness.claude_code]\ncommand = "claude -p < {prompt_path}"\n'
    )
    return tmp_path


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert buddy.__version__ in result.stdout


def test_bare_invocation_is_the_session_loop():
    """`buddy` with no subcommand runs the manager loop, so it is not
    invoked here: it would never return. Its two moving parts, reconcile and
    tick, are covered by test_reconcile.py and test_manager.py."""
    from buddy.cli import _session, main_callback

    assert main_callback.__doc__
    assert callable(_session)


def test_every_section_11_command_is_present():
    listed = runner.invoke(app, ["--help"]).stdout
    for command in (
        "doctor",
        "spawn",
        "status",
        "watch",
        "attach",
        "logs",
        "diff",
        "merge",
        "kill",
        "reprioritize",
        "history",
        "compact",
        "memory",
        "discard",
        "setup",
        "update",
        "uninstall",
        "config",
    ):
        assert command in listed


# -- buddy config ------------------------------------------------------


def test_config_shows_where_it_is_and_that_it_is_valid(home: Path):
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.stdout
    assert str(home / "config.toml") in result.stdout
    assert "valid" in result.stdout
    assert "webapp" in result.stdout


def test_config_reports_a_broken_file_without_needing_it_to_load(home: Path):
    """The one time you most need this command is the one time every other
    command refuses to start."""
    (home / "config.toml").write_text("[buddy]\nmax_concurrent = 'lots'\n")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 2
    assert "buddy.max_concurrent" in result.stdout
    assert "buddy config edit" in result.stdout


def _editor(tmp_path: Path, *contents: str) -> str:
    """An $EDITOR that writes each of `contents` on successive runs."""
    script = tmp_path / "editor.py"
    queue = tmp_path / "editor-queue.json"
    import json
    import sys

    queue.write_text(json.dumps(list(contents)))
    script.write_text(
        "import json, pathlib, sys\n"
        f"q = pathlib.Path({str(queue)!r}); items = json.loads(q.read_text())\n"
        "pathlib.Path(sys.argv[-1]).write_text(items.pop(0)); q.write_text(json.dumps(items))\n"
    )
    return f"{sys.executable} {script}"


def test_config_edit_opens_the_editor_and_checks_the_result(home: Path, tmp_path, monkeypatch):
    fine = (home / "config.toml").read_text() + "\n[buddy]\nmax_concurrent = 3\n"
    monkeypatch.setenv("EDITOR", _editor(tmp_path, fine))
    result = runner.invoke(app, ["config", "edit"])
    assert result.exit_code == 0, result.stdout
    assert "valid" in result.stdout
    assert "max_concurrent = 3" in (home / "config.toml").read_text()


def test_config_edit_that_breaks_the_file_says_how_and_does_not_hide_it(
    home: Path, tmp_path, monkeypatch
):
    monkeypatch.setenv("EDITOR", _editor(tmp_path, "[buddy]\ntrust_mode = 'yes'\n"))
    result = runner.invoke(app, ["config", "edit"])
    assert result.exit_code == 2
    assert "buddy.trust_mode" in result.stdout


def test_spawn_refuses_an_unknown_project(home: Path):
    result = runner.invoke(app, ["spawn", "nosuchproject", "do a thing"])
    assert result.exit_code == 1
    assert "unknown project" in result.stdout
    assert "webapp" in result.stdout  # names the ones it does know


def test_a_broken_config_is_reported_not_traced(home: Path):
    (home / "config.toml").write_text("[brain]\nprovider = 'hal9000'\n")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 2
    assert "config error" in result.stdout


def test_diff_of_an_unknown_task(home: Path):
    result = runner.invoke(app, ["diff", "t-9999"])
    assert result.exit_code == 2
    assert "no such task" in result.stdout


def test_merge_of_an_unknown_task(home: Path):
    result = runner.invoke(app, ["merge", "t-9999", "--yes"])
    assert result.exit_code == 2
    assert "no such task" in result.stdout


def test_logs_of_an_unknown_task(home: Path):
    result = runner.invoke(app, ["logs", "t-9999"])
    assert result.exit_code == 2
    assert "no such task" in result.stdout


def test_the_store_and_layout_are_created_on_first_use(home: Path):
    runner.invoke(app, ["diff", "t-9999"])
    assert (home / "state.db").exists()
    assert (home / "tasks").is_dir()
    assert (home / "worktrees").is_dir()


# -- the session loop ----------------------------------------


async def test_a_provider_failure_ends_the_exchange_not_the_session(home, monkeypatch, capsys):
    """Found by running `buddy` with no API key: the error escaped as a rich
    traceback and took the session, its 1s tick, and the dashboard with it."""
    from buddy.cli import Runtime, _converse
    from buddy.providers.base import ProviderError

    class Boom:
        provider = SimpleNamespace(name="anthropic", model="claude-opus-5")
        pending_events: list = []
        brainstorm = SimpleNamespace(active=False, drafts={})

        async def send(self, utterance, on_text=None):
            raise ProviderError("anthropic stream failed: could not resolve authentication")

    said = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(said))
    monkeypatch.setattr(Runtime, "brain", lambda self: Boom())

    await _converse(Runtime())

    printed = capsys.readouterr().out
    assert "could not resolve authentication" in printed
    assert "still up" in printed
    assert "Traceback" not in printed
    # It reached the second utterance, which means the loop survived the first.
    assert "Stopped." in printed


async def test_brainstorm_then_go_from_the_terminal(home, monkeypatch, capsys):
    """The whole flow a user types: talk it through, see the drafts, /go.

    Through `_converse` itself - the loop the terminal, the overlay and voice
    all share - with a real Brain and scripted model replies. tmux and git
    are the manager tests' fakes, so nothing here can touch a real session.
    """
    from buddy.brain import Brain
    from buddy.cli import Runtime, _converse
    from buddy.providers.base import ToolCall
    from buddy.workspace import Workspace
    from tests.test_context import ScriptedProvider
    from tests.test_manager import FakeRunner, FakeWorkspace

    runtime = Runtime()
    runtime.manager.runner = FakeRunner()
    runtime.manager.workspace = FakeWorkspace(home / "fake-worktrees")
    model = ScriptedProvider(
        [
            ToolCall("1", "start_brainstorm", {}),
            "Happy to think it through. What limits did you have in mind?",
            ToolCall(
                "2",
                "draft_brief",
                {"project": "webapp", "title": "Add rate limits", "goal": "429 past 100/min"},
            ),
            "Drafted it as d1.",
        ]
    )
    brain = Brain(
        runtime.config,
        runtime.store,
        runtime.manager,
        Workspace(runtime.config),
        model,
        confirm=lambda prompt, tier: True,
    )
    monkeypatch.setattr(Runtime, "brain", lambda self: brain)
    typed = iter(["let's brainstorm rate limiting", "100 a minute", "/drafts", "/go", "exit"])
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(typed))

    await _converse(runtime)

    printed = capsys.readouterr().out
    assert "What limits did you have in mind?" in printed
    assert "d1  Add rate limits  (webapp)" in printed
    assert "started  d1 -> t-0001" in printed
    assert "Brainstorming is off." in printed
    # The prompt said so while it was true, and stopped saying so after /go.
    assert "you (brainstorming)> " in prompts[1]
    assert prompts[-1].endswith("you> ")
    # Exactly the draft, now a real task, and nothing before /go.
    [task] = runtime.store.recent_tasks()
    assert task.title == "Add rate limits" and "429 past 100/min" in task.brief
    assert runtime.manager.runner.spawned == [("Monday", "t-0001", 1)]


# -- setup, update, uninstall ----------------------------------------


def test_setup_refuses_an_unknown_step_and_lists_the_real_ones(home: Path):
    result = runner.invoke(app, ["setup", "--force", "nonsense"])
    assert result.exit_code == 1
    assert "unknown step" in result.stdout
    for step in ("platform", "keys", "config", "doctor"):
        assert step in result.stdout


async def test_purge_refuses_while_a_branch_still_holds_work(home: Path, monkeypatch, capsys):
    """Unmerged work is the one thing here that cannot be recreated from anywhere else."""
    from buddy.cli import Runtime, _uninstall

    runtime = Runtime()
    monkeypatch.setattr("buddy.cli.Runtime", lambda *a, **k: runtime)

    async def unmerged(_runtime):
        return [("t-0001", "buddy/t-0001-thing")]

    monkeypatch.setattr("buddy.cli._unmerged_work", unmerged)

    code = await _uninstall(purge=True, yes=True, force=False)
    printed = capsys.readouterr().out
    assert code == 1
    assert "Refusing to purge" in printed
    assert "t-0001 on buddy/t-0001-thing" in printed
    assert "buddy merge" in printed and "--force" in printed
    still_there = await asyncio.to_thread(home.exists)
    assert still_there, "the home directory must still be there"


async def test_doctor_treats_a_projectless_buddy_as_a_problem(tmp_path: Path, monkeypatch, capsys):
    """It used to print "nothing can be spawned" and "Everything checks out"
    in the same table, which is how a real setup looked finished and then
    dead-ended in conversation."""
    from buddy.cli import run_doctor

    monkeypatch.setenv("BUDDY_HOME", str(tmp_path))
    # No [projects] block at all - the state a skipped setup prompt leaves.
    (tmp_path / "config.toml").write_text(
        '[harness.claude_code]\ncommand = "claude -p < {prompt_path}"\n'
    )

    async def reachable(_self):
        from buddy.providers.base import Capabilities, ProbeResult

        return ProbeResult("anthropic", "m", True, Capabilities())

    monkeypatch.setattr("buddy.providers.anthropic.AnthropicProvider.probe", reachable)
    problems = await run_doctor(tmp_path)
    printed = capsys.readouterr().out

    assert problems >= 1, "a Buddy that cannot spawn anything is not healthy"
    assert "Everything checks out" not in printed
    assert "buddy setup --force config" in printed


def test_the_brain_is_told_how_to_register_a_project():
    """The brain can only pass on what it is told. "none configured" is a
    fact without a remedy, so it improvised a sentence the user could not
    act on."""
    from buddy.brain import Brain

    assert "buddy setup --force config" in Brain.NO_PROJECTS
    assert "restart" in Brain.NO_PROJECTS


async def test_merge_from_the_cli_waits_for_a_pending_conflict_fix(home: Path, capsys):
    from buddy.cli import Runtime, _merge
    from buddy.models import TaskSpec, TaskState

    runtime = Runtime()
    for spec in (
        TaskSpec(
            id="t-0001", title="a", brief="b", harness="claude_code", project="webapp", priority=3
        ),
        TaskSpec(
            id="t-0002",
            title="fix",
            brief="b",
            harness="claude_code",
            project="webapp",
            priority=3,
            resolves="t-0009",
            start_from="buddy/t-0009-x",
        ),
    ):
        runtime.store.create_task(spec, TaskState.DONE if spec.id == "t-0001" else TaskState.QUEUED)
    runtime.store.close()

    import typer

    with pytest.raises(typer.Exit):
        await _merge("t-0001", None, yes=True)
    printed = capsys.readouterr().out
    assert "t-0002 is resolving merge conflicts" in printed
    assert "--force" in printed


def test_spawn_when_tmux_cannot_start_says_so_instead_of_a_traceback(home: Path, monkeypatch):
    """Found running `buddy spawn` end to end with a tmux socket path too long
    to bind: the error surfaced as a full rich traceback."""
    from buddy.tmux_runner import TmuxError, TmuxRunner

    async def refuse(self, *args, **kwargs):
        raise TmuxError("tmux new-session failed (1): error connecting (File name too long)")

    monkeypatch.setattr(TmuxRunner, "ensure_session", refuse)
    monkeypatch.setattr("buddy.harnesses.claude_code.ClaudeCodeAdapter.preflight", _usable_report)
    result = runner.invoke(app, ["spawn", "webapp", "do a thing"])
    assert result.exit_code == 1
    assert "t-0001 is queued, but tmux would not start it" in result.stdout
    assert "Traceback" not in result.stdout


async def _usable_report(self):
    from buddy.harnesses.base import PreflightReport, Requirement

    return PreflightReport(
        harness="claude_code", binary="/bin/true", requirements=(Requirement("headless", True),)
    )


async def test_merge_and_discard_from_the_cli_wait_for_a_running_task(home: Path, capsys):
    """Not overridable with --force, which answers the conflict hold: nothing
    makes deleting a working agent's checkout safe."""
    import typer

    from buddy.cli import Runtime, _discard, _merge
    from buddy.models import TaskSpec, TaskState

    runtime = Runtime()
    runtime.store.create_task(
        TaskSpec(
            id="t-0001", title="a", brief="b", harness="claude_code", project="webapp", priority=3
        ),
        TaskState.RUNNING,
    )
    runtime.store.close()

    with pytest.raises(typer.Exit):
        await _merge("t-0001", None, yes=True, force=True)
    assert "not merging: t-0001 is still running" in capsys.readouterr().out
    with pytest.raises(typer.Exit):
        await _discard("t-0001", yes=True)
    assert "not discarding: t-0001 is still running" in capsys.readouterr().out
