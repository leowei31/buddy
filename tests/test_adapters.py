"""Invocation strings, waiting_patterns, parse_result - pure functions.

Plus a live preflight against the installed `claude`, which is the only way
a harness's four requirements can be answered honestly.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from buddy.config import HarnessConfig
from buddy.harnesses.base import PreflightReport, Requirement, run_command
from buddy.harnesses.claude_code import (
    DEFAULT_COMMAND,
    DEFAULT_WAITING_PATTERNS,
    ClaudeCodeAdapter,
)
from buddy.logs import strip_ansi
from buddy.models import TaskRun


def make_adapter(**overrides) -> ClaudeCodeAdapter:
    block = {"command": DEFAULT_COMMAND, "waiting_patterns": list(DEFAULT_WAITING_PATTERNS)}
    block.update(overrides)
    return ClaudeCodeAdapter(HarnessConfig.from_dict("claude_code", block))


def make_run(tmp_path: Path) -> TaskRun:
    return TaskRun(
        task_id="t-0142",
        attempt=1,
        agent="scout",
        worktree=tmp_path / "wt",
        branch="buddy/t-0142-x",
        base_ref="abc",
        log_path=tmp_path / "attempt-1.log",
    )


# -- invocation -----------------------------------------------------


def test_invocation_reads_the_brief_from_the_prompt_file(tmp_path: Path):
    adapter = make_adapter()
    command = adapter.invocation(make_run(tmp_path), tmp_path / "prompt.md")
    assert str(tmp_path / "prompt.md") in command
    assert "-p" in command
    assert "--permission-mode" in command
    # No cd, no logging, no sentinels: the wrapper adds those.
    assert "cd " not in command
    assert "__BUDDY_" not in command


def test_invocation_includes_the_model_when_one_is_chosen(tmp_path: Path):
    adapter = make_adapter()
    assert "--model claude-sonnet-5" in adapter.invocation(
        make_run(tmp_path), tmp_path / "p.md", model="claude-sonnet-5"
    )


def test_invocation_omits_the_model_flag_when_there_is_none(tmp_path: Path):
    command = make_adapter().invocation(make_run(tmp_path), tmp_path / "p.md")
    assert "--model" not in command


def test_a_task_model_beats_the_harness_default(tmp_path: Path):
    adapter = make_adapter(default_model="claude-haiku-4-5-20251001")
    assert "--model claude-sonnet-5" in adapter.invocation(
        make_run(tmp_path), tmp_path / "p.md", model="claude-sonnet-5"
    )
    assert "--model claude-haiku-4-5-20251001" in adapter.invocation(
        make_run(tmp_path), tmp_path / "p.md"
    )


def test_the_command_template_comes_from_config_not_code(tmp_path: Path):
    """Updating a harness's flags is an edit, not a release."""
    adapter = make_adapter(command="claude --brand-new-flag {prompt_path}")
    assert adapter.invocation(make_run(tmp_path), tmp_path / "p.md") == (
        f"claude --brand-new-flag {tmp_path / 'p.md'}"
    )


# -- waiting patterns ----------------------------------------------


@pytest.mark.parametrize(
    "line",
    ["Overwrite config.toml? (y/n)", "Do you want to proceed?", "❯ 1. Yes"],
)
def test_waiting_patterns_catch_a_prompt_that_slipped_past_auto_approve(line: str):
    adapter = make_adapter()
    assert any(pattern.search(line) for pattern in adapter.waiting_patterns)


def test_ordinary_output_is_not_mistaken_for_a_prompt():
    adapter = make_adapter()
    for line in ["Running tests...", "3 passed", "Wrote src/main.py"]:
        assert not any(pattern.search(line) for pattern in adapter.waiting_patterns)


# -- parse_result ---------------------------------------------------

RESULT_JSON = (
    '{"type":"result","subtype":"success","is_error":false,"duration_ms":1200,'
    '"num_turns":3,"result":"Added retry logic and two tests.",'
    '"session_id":"abc","total_cost_usd":0.02}'
)
FAILED_JSON = (
    '{"type":"result","subtype":"success","is_error":true,"duration_ms":443,'
    '"num_turns":1,"result":"API Error: 404 model not found","session_id":"x"}'
)


def test_parse_result_extracts_the_final_message():
    summary = make_adapter().parse_result(f"some noise\n{RESULT_JSON}\n", 0)
    assert summary.ok
    assert summary.summary == "Added retry logic and two tests."
    assert summary.detail["num_turns"] == 3
    assert summary.detail["parsed"] is True


def test_parse_result_reads_is_error_not_subtype():
    """Verified against CLI 1.0.128: a failed run still says
    subtype="success"."""
    summary = make_adapter().parse_result(FAILED_JSON, 1)
    assert not summary.ok
    assert summary.detail["subtype"] == "success"
    assert summary.detail["is_error"] is True
    assert "404" in summary.summary


def test_the_exit_code_wins_when_the_harness_disagrees():
    """The pane's exit status is the completion signal, never the log."""
    summary = make_adapter().parse_result(RESULT_JSON, 1)
    assert summary.ok is False
    assert summary.detail["agrees_with_exit_code"] is False


def test_parse_result_survives_ansi_and_terminal_noise():
    noisy = f"\x1b[32mrunning\x1b[0m\r\n\x1b]0;title\x07{RESULT_JSON}\r\n"
    summary = make_adapter().parse_result(noisy, 0)
    assert summary.summary == "Added retry logic and two tests."


def test_parse_result_takes_the_last_result_object():
    log = f'{{"type":"result","is_error":true,"result":"old"}}\n{RESULT_JSON}\n'
    assert make_adapter().parse_result(log, 0).summary == "Added retry logic and two tests."


def test_parse_result_ignores_non_result_json():
    log = f'{{"type":"assistant","result":"chatter"}}\n{RESULT_JSON}\n'
    assert make_adapter().parse_result(log, 0).summary == "Added retry logic and two tests."


def test_unparseable_log_falls_back_to_the_tail_and_says_so():
    summary = make_adapter().parse_result("line one\nline two\ncrashed hard\n", 137)
    assert summary.detail["parsed"] is False
    assert "crashed hard" in summary.summary
    assert summary.ok is False


def test_an_empty_log_with_a_clean_exit_is_still_ok():
    assert make_adapter().parse_result("", 0).ok is True


# -- preflight ------------------------------------------------------


def test_preflight_report_summarises_its_verdict():
    good = PreflightReport(
        harness="h",
        binary="/usr/bin/h",
        version="1.0",
        requirements=(Requirement("headless", True),),
    )
    assert good.ok
    assert "ok" in good.summary()

    bad = PreflightReport(
        harness="h",
        binary="/usr/bin/h",
        requirements=(Requirement("auto_approve", False, "no flag"),),
    )
    assert not bad.ok
    assert "auto_approve" in bad.summary()
    assert not PreflightReport(harness="h").installed


async def test_preflight_reports_a_missing_binary():
    class Missing(ClaudeCodeAdapter):
        binary = "definitely-not-installed-xyz"

    report = await Missing(HarnessConfig.from_dict("x", {"command": "x"})).preflight()
    assert not report.installed
    assert not report.ok
    assert "not on PATH" in report.summary()


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude is not installed")
async def test_live_preflight_answers_all_four_requirements():
    """The only honest way to answer the four requirements: ask the installed binary."""
    report = await make_adapter().preflight()
    assert report.installed
    assert {r.key for r in report.requirements} == {
        "headless",
        "prompt_input",
        "auto_approve",
        "exit_codes",
    }
    assert report.ok, report.summary()
    assert report.version


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude is not installed")
async def test_the_installed_version_has_no_input_file_flag():
    """Older releases had `--input-file`; this is what is actually there."""
    _, out, err = await run_command("claude", "--help")
    assert "--input-file" not in strip_ansi(out + err)
    assert "--print" in strip_ansi(out + err)
    report = await make_adapter().preflight()
    assert any("stdin" in note for note in report.notes)


# -- the auto-approve mode has to be one that actually runs unattended -----


def test_the_default_command_does_not_leave_bash_blocked():
    """Verified against CLI 1.0.128: acceptEdits denies Bash, so the agent
    cannot run the `git commit` Buddy's working rules ask for."""
    assert "bypassPermissions" in DEFAULT_COMMAND
    assert "acceptEdits" not in DEFAULT_COMMAND


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude is not installed")
async def test_preflight_warns_about_a_half_approving_mode():
    adapter = make_adapter(command="claude -p --permission-mode acceptEdits < {prompt_path}")
    report = await adapter.preflight()
    assert any("denies Bash" in note for note in report.notes)


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude is not installed")
async def test_preflight_warns_when_nothing_auto_approves():
    report = await make_adapter(command="claude -p < {prompt_path}").preflight()
    assert any("no auto-approve flag" in note for note in report.notes)


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude is not installed")
async def test_the_default_command_draws_no_warning():
    report = await make_adapter().preflight()
    assert not any("denies Bash" in note for note in report.notes)
