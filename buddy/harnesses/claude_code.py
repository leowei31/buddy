"""Claude Code adapter.

Every flag here was read from `claude --help` and confirmed by running it,
never remembered. Against the installed CLI (1.0.128):

* `-p` / `--print` runs non-interactively and exits            -> requirement 1
* the prompt arrives on stdin, so no quoting layer touches it  -> requirement 2
* `--permission-mode` selects an auto-approving mode           -> requirement 3
* a failed run exits non-zero                                  -> requirement 4

`--output-format json` gives `parse_result` a structured final message
instead of scraped prose. Note that its `subtype` field reads `"success"`
even for a failed run, so `is_error` is the field to trust - and neither is
the completion signal, which is the pane's exit status.

The command template lives in config.toml; nothing here hardcodes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from buddy.harnesses.base import (
    BaseAdapter,
    PreflightReport,
    ResultSummary,
)
from buddy.logs import strip_ansi
from buddy.models import TaskRun

#: Written into config.toml by `buddy setup` and editable
#: there when the CLI's flags change.
#:
#: `bypassPermissions`, not `acceptEdits`: acceptEdits auto-approves file
#: edits but still denies Bash, so the agent cannot run the `git commit` that
#: Buddy's working rules ask of it, and every task would end dirty and lean on
#: the WIP checkpoint. Verified against 1.0.128, which reported the denial in
#: `permission_denials`. Unattended agents must not block on permission
#: prompts at all (requirement 3); what makes that safe is the worktree
#: and branch boundary, with a container as the recommended second one.
#: `stream-json`, not `json`. `json` emits a single object when the run is
#: over, so a nine-minute task leaves a three-line log and a blank pane for
#: nine minutes - nothing to watch, nothing for the dashboard to show, and
#: `last_output_at` frozen at the spawn time, which makes a healthy long task
#: indistinguishable from a stalled one to the stall detector.
#:
#: `--verbose` is required for stream-json under `--print`; the final
#: `{"type": "result"}` object is still the last line either way, so
#: `parse_result` is unchanged. Verified against CLI 1.0.128.
DEFAULT_COMMAND = (
    "claude -p --output-format stream-json --verbose"
    " --permission-mode bypassPermissions {model_flag} < {prompt_path}"
)

#: Modes that leave some tool class blocked, and so are not "unattended".
PARTIAL_APPROVE_MODES = ("acceptEdits", "default", "plan")

#: A y/n that slipped past auto-approve.
DEFAULT_WAITING_PATTERNS = (r"\(y/n\)", r"Do you want to proceed\?", r"❯\s*1\.\s*Yes")


class ClaudeCodeAdapter(BaseAdapter):
    name = "claude_code"
    binary = "claude"
    default_command = DEFAULT_COMMAND
    default_waiting_patterns = DEFAULT_WAITING_PATTERNS
    #: The CLI's own `sonnet` alias resolves to a retired model on 1.0.128, so
    #: a task that names no model would 404 on its first call.
    default_model = "claude-sonnet-5"
    install_hint = "npm install -g @anthropic-ai/claude-code, then run `claude` once to log in"
    npm_package = "@anthropic-ai/claude-code"
    #: Claude Code signs in with its own subscription; exporting an API key
    #: into it would silently move a subscriber onto per-token billing.
    credential_env: ClassVar[dict[str, str]] = {}

    async def preflight(self) -> PreflightReport:
        if await self.locate() is None:
            return PreflightReport(harness=self.name, where=self.where)

        help_text = await self.help_text()
        requirements = (
            self.flag_requirement("headless", help_text, "--print", "-p"),
            self.flag_requirement("prompt_input", help_text, "--print"),
            self.flag_requirement(
                "auto_approve",
                help_text,
                "--permission-mode",
                "--dangerously-skip-permissions",
            ),
            await self.check_exit_codes(),
        )
        notes = []
        if "--input-file" not in help_text:
            # Older releases had one; this is the version actually installed.
            notes.append("no --input-file on this version; the brief is delivered on stdin")
        notes.extend(self._permission_notes())
        return await self.report(requirements, notes)

    def _permission_notes(self) -> list[str]:
        """Warn when the configured command cannot actually run unattended.

        A mode that still denies Bash looks fine in `--help` and fails only at
        run time, as a task that quietly never commits (requirement 3).
        """
        command = self.config.command
        if "--dangerously-skip-permissions" in command or "bypassPermissions" in command:
            return []
        for mode in PARTIAL_APPROVE_MODES:
            if f"--permission-mode {mode}" in command:
                return [
                    f"command uses --permission-mode {mode}, which still denies Bash: "
                    "the agent will not be able to commit its own work. "
                    "Use bypassPermissions inside the worktree boundary."
                ]
        return ["command names no auto-approve flag; the harness may block on a prompt"]

    def invocation(self, run: TaskRun, prompt_path: Path, *, model: str | None = None) -> str:
        return super().invocation(run, prompt_path, model=model)

    def describe_activity(self, log_text: str, lines: int = 20) -> str:
        """The stream-json events as readable lines."""
        return "\n".join(describe_events(log_text, limit=lines))

    def parse_result(self, log_text: str, exit_code: int) -> ResultSummary:
        """Pull the final result object out of the log.

        The log is raw pane output, so the JSON is surrounded by escape codes
        and whatever else was on screen; the last object with
        `"type": "result"` is the run's own verdict.
        """
        payload = _last_result_object(log_text)
        if payload is None:
            return ResultSummary(
                ok=exit_code == 0,
                summary=_fallback_summary(log_text),
                exit_code=exit_code,
                detail={"parsed": False},
            )
        # `is_error`, not `subtype`: a failed run still reports
        # subtype="success" (verified against 1.0.128).
        reported_ok = not payload.get("is_error", False)
        return ResultSummary(
            ok=exit_code == 0 and reported_ok,
            summary=str(payload.get("result", "")).strip(),
            exit_code=exit_code,
            detail={
                "parsed": True,
                "is_error": payload.get("is_error"),
                "subtype": payload.get("subtype"),
                "num_turns": payload.get("num_turns"),
                "duration_ms": payload.get("duration_ms"),
                "total_cost_usd": payload.get("total_cost_usd"),
                "session_id": payload.get("session_id"),
                "agrees_with_exit_code": reported_ok == (exit_code == 0),
            },
        )


#: Events worth showing. The rest - init banners, usage accounting - are
#: noise on a three-line card.
def describe_events(log_text: str, limit: int = 20) -> list[str]:
    """The stream as lines a human can read.

    Raw stream-json in an agent card is no more use than a blank one, and the layer rule
    says only this module may know what Claude Code's output looks like - so
    the translation lives here rather than in the manager or the dashboard.
    """
    described: list[str] = []
    for line in strip_ansi(log_text).splitlines():
        line = line.strip()
        if not line.startswith("{"):
            if line and not line.startswith("__BUDDY_"):
                described.append(line)
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        rendered = _describe_event(event)
        if rendered:
            described.append(rendered)
    return described[-limit:] if limit > 0 else described


def _describe_event(event: dict) -> str:
    kind = event.get("type")
    if kind == "system" and event.get("subtype") == "init":
        model = event.get("model") or ""
        return f"started{f' on {model}' if model else ''}"
    if kind == "result":
        outcome = "failed" if event.get("is_error") else "done"
        turns = event.get("num_turns")
        return f"{outcome} after {turns} turns" if turns else outcome
    if kind in ("assistant", "user"):
        message = event.get("message")
        if not isinstance(message, dict):
            return ""
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text", "").strip():
                return _one_line(block["text"])
            if block.get("type") == "tool_use":
                return f"{block.get('name', 'tool')}: {_tool_detail(block.get('input') or {})}"
            if block.get("type") == "tool_result":
                return ""  # the call itself already said what was happening
    return ""


def _tool_detail(payload: dict) -> str:
    """The one field of a tool call that says what it is doing."""
    for key in ("command", "file_path", "path", "pattern", "query", "url", "description"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _one_line(value)
    return ""


def _one_line(text: str, width: int = 90) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def _last_result_object(log_text: str) -> dict | None:
    clean = strip_ansi(log_text)
    for line in reversed(clean.splitlines()):
        line = line.strip()
        if not line.startswith("{") or '"type"' not in line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "result":
            return payload
    return None


def _fallback_summary(log_text: str, lines: int = 5) -> str:
    """No parseable result: the tail is better than nothing, and the brain is
    told the summary is unparsed."""
    clean = [line for line in strip_ansi(log_text).splitlines() if line.strip()]
    return "\n".join(clean[-lines:])
