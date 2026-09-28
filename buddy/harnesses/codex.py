"""Codex adapter.

Every flag here was read from `codex exec --help` and confirmed by running the
real binary, never remembered. Against codex-cli 0.154.0:

* `codex exec` runs non-interactively and exits                 -> requirement 1
* `-` reads the prompt from stdin, untouched by any quoting      -> requirement 2
* `--dangerously-bypass-approvals-and-sandbox` never asks         -> requirement 3
* a failed turn exits 1, including a rejected API key            -> requirement 4

The success path was verified end to end against a scripted Responses API
server: the real binary ran a real shell command, committed in a real git
repository and exited 0. `tests/fixtures/harness_logs/codex_*.jsonl` are
those runs.

Three things about Codex that are easy to get wrong, all found by running it:

* **Its sandbox is bypassed, deliberately.** Under `--sandbox
  workspace-write` the agent can edit and commit in its worktree - Codex
  allows the worktree's `.git` - but a write anywhere else is "Operation not
  permitted" and the network is off (`CODEX_SANDBOX_NETWORK_DISABLED`). So
  `npm install`, a Go or Cargo build, anything with a cache in `$HOME`, fails
  - one tool error at a time, inside a run that can still end "successfully".
  Measured on 0.154.0 against a real worktree. What stands in for Codex's
  sandbox is the same as for every harness: Buddy's worktree and branch, with
  a container as the second boundary.
* **It reads `CODEX_API_KEY`, not `OPENAI_API_KEY`.** In exec mode an
  exported OPENAI_API_KEY is ignored and the run fails with "Missing bearer".
* **It never gives up on an unreachable API.** It prints "Reconnecting...
  waiting for network" every few seconds, forever. `is_progress` is what
  stops that reading as a healthy run.
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
#:
#: `--skip-git-repo-check` is harmless in a worktree and necessary in a
#: container, where the worktree's `.git` points at a path that is not
#: mounted. `--color never` keeps escape codes out of the JSON lines.
DEFAULT_COMMAND = (
    "codex exec --json --color never --dangerously-bypass-approvals-and-sandbox"
    " --skip-git-repo-check {model_flag} - < {prompt_path}"
)

#: None. `codex exec` has no interactive approval surface to match - the one
#: way it is known to block (openai/codex#4565, a forced approval under the
#: bypass flag) prints nothing at all, which is what stall detection is for.
#: Patterns invented without a prompt to match would only ever misfire.
DEFAULT_WAITING_PATTERNS: tuple[str, ...] = ()

#: Retry chatter: output that is not the run moving.
_RETRYING = re.compile(r"^\s*\{\s*\"type\"\s*:\s*\"error\"\s*,\s*\"message\"\s*:\s*\"Reconnecting")
_STDERR_LOG = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z\s+(ERROR|WARN|INFO)\s")

#: `/bin/zsh -lc "..."` - the shell Codex wraps every command in.
_SHELL_WRAPPER = re.compile(r"""^/\S*sh\s+-l?c\s+(?P<q>["'])(?P<body>.*)(?P=q)$""", re.S)


class CodexAdapter(BaseAdapter):
    name = "codex"
    binary = "codex"
    default_command = DEFAULT_COMMAND
    default_waiting_patterns = DEFAULT_WAITING_PATTERNS
    help_args = ("exec", "--help")
    install_hint = "npm install -g @openai/codex, then `codex login` (or set CODEX_API_KEY)"
    npm_package = "@openai/codex"
    credential_env: ClassVar[dict[str, str]] = {"OPENAI_API_KEY": "CODEX_API_KEY"}

    async def preflight(self) -> PreflightReport:
        if await self.locate() is None:
            return PreflightReport(harness=self.name, where=self.where)
        help_text = await self.help_text()
        requirements = (
            self.flag_requirement("headless", help_text, "--json"),
            _stdin_requirement(help_text),
            self.flag_requirement(
                "auto_approve", help_text, "--dangerously-bypass-approvals-and-sandbox"
            ),
            await self.check_exit_codes(),
        )
        return await self.report(requirements, self._command_notes())

    def _command_notes(self) -> list[str]:
        command = self.config.command
        notes: list[str] = []
        if "--dangerously-bypass-approvals-and-sandbox" not in command:
            notes.append(
                "command keeps Codex's own sandbox: writes outside the worktree are denied "
                "and the network is off, so installs and most builds will fail mid-task"
            )
        if "--json" not in command:
            notes.append("command has no --json: the slot card will show raw text only")
        return notes

    async def auth_status(self) -> tuple[bool | None, str]:
        code, _, _ = await self.probe("login", "status")
        if code == 0:
            return True, "logged in"
        if key := self.credential_in_env("CODEX_API_KEY"):
            return True, f"{key} is set"
        if self.credential_in_env("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY is set, but codex exec only reads CODEX_API_KEY"
        return False, "run `codex login`, or set CODEX_API_KEY in [harness.codex.env]"

    def is_progress(self, new_output: str) -> bool:
        lines = [line for line in new_output.splitlines() if line.strip()]
        if not lines:
            return False
        return not all(_RETRYING.match(line) or _STDERR_LOG.match(line) for line in lines)

    def describe_activity(self, log_text: str, lines: int = 20) -> str:
        return "\n".join(describe_events(log_text, limit=lines))

    def parse_result(self, log_text: str, exit_code: int) -> ResultSummary:
        """The last agent message, and whether the turn completed.

        `turn.completed` and `turn.failed` are Codex's own verdict. Neither is
        the completion signal - that is the pane's exit status - but a
        run that exits 0 without completing its turn is worth saying so.
        """
        message = ""
        failure = ""
        completed = False
        usage: dict[str, Any] = {}
        commands = failed_commands = 0
        thread_id = None
        for event in stream.objects(log_text):
            kind = event.get("type")
            if kind == "thread.started":
                thread_id = event.get("thread_id")
            elif kind == "turn.completed":
                completed = True
                usage = event.get("usage") or {}
            elif kind == "turn.failed":
                failure = _error_message(event.get("error"))
            elif kind == "item.completed":
                item = event.get("item") or {}
                if item.get("type") == "agent_message" and str(item.get("text", "")).strip():
                    message = str(item["text"]).strip()
                elif item.get("type") == "command_execution":
                    commands += 1
                    if item.get("exit_code") not in (0, None):
                        failed_commands += 1

        if not completed and not failure and not message:
            return ResultSummary(
                ok=exit_code == 0,
                summary=stream.fallback_summary(log_text),
                exit_code=exit_code,
                detail={"parsed": False},
            )
        reported_ok = completed and not failure
        return ResultSummary(
            ok=exit_code == 0 and reported_ok,
            summary=message or failure,
            exit_code=exit_code,
            detail={
                "parsed": True,
                "turn_completed": completed,
                "error": failure or None,
                "thread_id": thread_id,
                "commands": commands,
                "failed_commands": failed_commands,
                "usage": usage,
                "agrees_with_exit_code": reported_ok == (exit_code == 0),
            },
        )


def _stdin_requirement(help_text: str) -> Requirement:
    if "stdin" in help_text:
        return Requirement("prompt_input", True, "reads the prompt from stdin with `-`")
    return Requirement("prompt_input", False, "`codex exec --help` no longer mentions stdin")


def _error_message(error: object) -> str:
    if isinstance(error, dict):
        return str(error.get("message", "")).strip()
    return str(error or "").strip()


def unwrap_shell(command: str) -> str:
    """`/bin/zsh -lc "git status"` -> `git status`, which is what the agent ran."""
    match = _SHELL_WRAPPER.match(command.strip())
    if not match:
        return command
    body = match.group("body")
    if match.group("q") == '"':
        body = body.replace('\\"', '"').replace("\\\\", "\\")
    return body


def describe_events(log_text: str, limit: int = 20) -> list[str]:
    """Codex's JSON lines as sentences for a slot card."""
    described: list[str] = []
    for event in stream.events(log_text):
        if isinstance(event, str):
            if not _STDERR_LOG.match(event):
                described.append(stream.one_line(event))
            continue
        line = _describe(event)
        if line:
            described.append(line)
    return stream.last_lines(described, limit)


def _describe(event: dict[str, Any]) -> str:
    kind = event.get("type")
    if kind == "thread.started":
        return "started"
    if kind == "turn.completed":
        return "done"
    if kind == "turn.failed":
        return stream.one_line(f"failed: {_error_message(event.get('error'))}")
    if kind == "error":
        message = str(event.get("message", ""))
        if message.startswith("Reconnecting"):
            return stream.one_line(f"retrying: {message}")
        return stream.one_line(f"error: {message}")
    if kind not in ("item.started", "item.completed"):
        return ""

    item = event.get("item") or {}
    item_type = item.get("type")
    if item_type == "command_execution":
        command = stream.one_line(unwrap_shell(str(item.get("command", ""))))
        if kind == "item.started":
            return f"$ {command}"
        exit_code = item.get("exit_code")
        return f"exit {exit_code}: {command}" if exit_code not in (0, None) else ""
    if kind == "item.started":
        return ""
    if item_type == "agent_message":
        return stream.one_line(item.get("text", ""))
    if item_type == "file_change":
        paths = [str(c.get("path", "")) for c in item.get("changes") or [] if isinstance(c, dict)]
        return stream.one_line("edited " + ", ".join(p for p in paths if p)) if paths else ""
    if item_type == "mcp_tool_call":
        return stream.one_line(f"{item.get('server', 'mcp')}.{item.get('tool', 'tool')}")
    if item_type == "web_search":
        return stream.one_line(f"search: {item.get('query', '')}")
    if item_type == "error":
        return stream.one_line(f"warning: {item.get('message', '')}")
    return ""
