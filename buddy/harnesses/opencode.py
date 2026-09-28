"""OpenCode adapter.

Every flag here was read from `opencode run --help` and confirmed by running
the real binary. Against opencode 1.18.31:

* `opencode run` runs non-interactively and exits              -> requirement 1
* the prompt is a positional argument                          -> requirement 2
* `--auto` approves anything not explicitly denied              -> requirement 3
* a failed run exits 1, including a rejected API key           -> requirement 4

A real tool-using run - write a file, `git commit` it, report back - ran to
exit 0 with this exact command line; `tests/fixtures/harness_logs/opencode_*`
are that run and a rejected-key run.

**The prompt must not arrive on stdin.** `opencode run < prompt.md` does not
fail: it starts a session, reaches the model, and never produces another byte
or exits. Measured on 1.18.31, with and without credentials. The prompt goes
in as an argument instead, after `--` so a brief that begins with "-" is not
read as a flag, and through `"$(cat ...)"` so its contents are data to bash
rather than code: quotes, `$(...)` and backticks inside it arrive verbatim.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from buddy.harnesses import stream
from buddy.harnesses.base import (
    BaseAdapter,
    PreflightReport,
    Requirement,
    ResultSummary,
)

#: Written into config.toml by `buddy setup` and editable there.
DEFAULT_COMMAND = 'opencode run --format json --auto {model_flag} -- "$(cat {prompt_path})"'

#: None, because `opencode run` never waits. A permission set to "ask" is
#: answered by `--auto`; without it the request is *rejected* - the run prints
#: "permission requested: ...; auto-rejecting", carries on and exits 0 with the
#: work not done (verified, 1.18.31). That is a result problem rather than a
#: waiting one, so `parse_result` counts rejections instead.
DEFAULT_WAITING_PATTERNS: tuple[str, ...] = ()

_REJECTED = "rejected permission"

#: The input field of each built-in tool that says what it is doing.
_TOOL_DETAIL = ("command", "filePath", "path", "pattern", "url", "query", "description")

_STDIN_DELIVERY = re.compile(r"<\s*\{prompt_path\}")


class OpenCodeAdapter(BaseAdapter):
    name = "opencode"
    binary = "opencode"
    default_command = DEFAULT_COMMAND
    default_waiting_patterns = DEFAULT_WAITING_PATTERNS
    help_args = ("run", "--help")
    install_hint = "npm install -g opencode-ai, then `opencode auth login` for your provider"
    npm_package = "opencode-ai"
    #: OpenCode reads each provider's standard variable itself.
    credential_env: ClassVar[dict[str, str]] = {
        "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY": "OPENAI_API_KEY",
    }

    async def preflight(self) -> PreflightReport:
        if await self.locate() is None:
            return PreflightReport(harness=self.name, where=self.where)
        help_text = await self.help_text()
        requirements = (
            self.flag_requirement("headless", help_text, "--format"),
            _argument_requirement(help_text),
            self.flag_requirement("auto_approve", help_text, "--auto"),
            await self.check_exit_codes(),
        )
        return await self.report(requirements, self._command_notes())

    def _command_notes(self) -> list[str]:
        command = self.config.command
        notes: list[str] = []
        if _STDIN_DELIVERY.search(command):
            notes.append(
                "command feeds the prompt on stdin, where opencode run hangs forever "
                'without exiting; pass it as an argument: -- "$(cat {prompt_path})"'
            )
        if "--auto" not in command:
            notes.append(
                "command has no --auto: opencode run rejects every permission request "
                "and still exits 0, so tasks will report success with work left undone"
            )
        return notes

    async def auth_status(self) -> tuple[bool | None, str]:
        """OpenCode runs without any credentials, on its own free default model,
        so "not signed in" is never a reason a task cannot start. What is worth
        knowing is *which* model a task will get."""
        code, out, err = await self.probe("auth", "list")
        text = stream.one_line(out + err, width=400)
        match = re.search(r"(\d+)\s+credentials?", text)
        if code == 0 and match and int(match.group(1)) > 0:
            return True, f"{match.group(1)} provider credential(s)"
        if key := self.credential_in_env(*self.credential_env.values()):
            return True, f"{key} is set"
        return None, (
            "no provider credentials: tasks run on OpenCode's free default model "
            "unless a model is named or `opencode auth login` is run"
        )

    def describe_activity(self, log_text: str, lines: int = 20) -> str:
        return "\n".join(describe_events(log_text, limit=lines))

    def parse_result(self, log_text: str, exit_code: int) -> ResultSummary:
        text = ""
        error = ""
        tools = 0
        rejected = 0
        finished = False
        tokens = 0
        cost = 0.0
        session_id = None
        for event in stream.objects(log_text):
            session_id = event.get("sessionID") or session_id
            kind = event.get("type")
            part = event.get("part") or {}
            if kind == "text" and str(part.get("text", "")).strip():
                text = str(part["text"]).strip()
            elif kind == "tool_use":
                tools += 1
                state = part.get("state") or {}
                if state.get("status") == "error" and _REJECTED in str(state.get("error", "")):
                    rejected += 1
            elif kind == "step_finish":
                finished = part.get("reason") == "stop" or finished
                tokens += int((part.get("tokens") or {}).get("total") or 0)
                cost += float(part.get("cost") or 0)
            elif kind == "error":
                error = _error_message(event.get("error"))

        if not (text or error or tools or finished):
            return ResultSummary(
                ok=exit_code == 0,
                summary=stream.fallback_summary(log_text),
                exit_code=exit_code,
                detail={"parsed": False},
            )
        # A rejected tool call is work the brief asked for and did not get.
        reported_ok = not error and not rejected
        summary = text or error
        if rejected:
            summary = f"{rejected} tool call(s) were refused permission. {summary}".strip()
        return ResultSummary(
            ok=exit_code == 0 and reported_ok,
            summary=summary,
            exit_code=exit_code,
            detail={
                "parsed": True,
                "error": error or None,
                "finished": finished,
                "tool_calls": tools,
                "rejected_tool_calls": rejected,
                "tokens": tokens,
                "cost": cost,
                "session_id": session_id,
                "agrees_with_exit_code": reported_ok == (exit_code == 0),
            },
        )


def _argument_requirement(help_text: str) -> Requirement:
    if "message" in help_text:
        return Requirement("prompt_input", True, "takes the prompt as a positional argument")
    return Requirement("prompt_input", False, "`opencode run --help` lists no message argument")


def _error_message(error: object) -> str:
    if not isinstance(error, dict):
        return str(error or "").strip()
    data = error.get("data")
    if isinstance(data, dict) and data.get("message"):
        return str(data["message"]).strip()
    return str(error.get("name", "error")).strip()


def describe_events(log_text: str, limit: int = 20) -> list[str]:
    """OpenCode's JSON events as sentences for a slot card."""
    described: list[str] = []
    for event in stream.events(log_text):
        if isinstance(event, str):
            described.append(stream.one_line(event))
            continue
        line = _describe(event)
        if line:
            described.append(line)
    return stream.last_lines(described, limit)


def _describe(event: dict[str, Any]) -> str:
    kind = event.get("type")
    part = event.get("part") or {}
    if kind == "text":
        return stream.one_line(part.get("text", "")) if str(part.get("text", "")).strip() else ""
    if kind == "tool_use":
        state = part.get("state") or {}
        tool = part.get("tool", "tool")
        detail = stream.detail_of(state.get("input"), _TOOL_DETAIL)
        line = f"{tool}: {detail}" if detail else str(tool)
        if state.get("status") == "error":
            refused = _REJECTED in str(state.get("error", ""))
            line += " (refused permission)" if refused else " (failed)"
        return stream.one_line(line)
    if kind == "error":
        return stream.one_line(f"error: {_error_message(event.get('error'))}")
    if kind == "step_finish" and part.get("reason") == "stop":
        return "done"
    return ""
