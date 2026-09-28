"""Antigravity adapter.

Every flag here was read from `agy --help` and confirmed against the real
binary, never remembered. Against Antigravity CLI 1.2.2:

* `-p` / `--print` runs one prompt non-interactively             -> requirement 1
* the prompt is the value of `-p`                                -> requirement 2
* `--dangerously-skip-permissions` sets "always-proceed"          -> requirement 3
* a failed run exits 1; `--help` exits 0 and a bad flag exits 2   -> requirement 4

**What is and is not verified.** Every failure path was run for real: a
rejected API key (exit 1, a `result` event with `status: "ERROR"`), and a
signed-out run. The *success* path was not, because it needs a Google account
or a Gemini API key, and no fake can stand in: unlike Codex and OpenCode,
`agy` cannot be pointed at another server. `parse_result` reads the documented
`result` envelope and trusts the exit code over it, so an unexpected success
shape degrades to "exit 0, summary from the tail" rather than to a wrong
verdict. `tests/fixtures/harness_logs/antigravity_*` are the real runs.

Four things about `agy` that are easy to get wrong, all found by running it:

* **Print mode gives up after five minutes** unless `--print-timeout` says
  otherwise, and nothing in the headless documentation mentions it. Without
  the flag every task longer than five minutes is cut off.
* **`-p` takes a value**, and stdin is not an alternative: `agy -p
  --output-format stream-json` sends "--output-format" as the prompt. So the
  brief is attached to the flag as `-p="$(cat prompt.md)"`, where bash treats
  its contents as data.
* **A brief that starts with "/" is a slash command** unless
  `--disable-slash-commands` is passed.
* **Signed out, it does not fail.** It prints a sign-in URL and waits for
  someone to paste a code. That is what the waiting patterns match, so the
  slot says it needs you instead of looking busy.
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
#: `--print-timeout 24h` rather than a figure tied to `max_runtime`: Buddy's
#: own timeout is the one that decides, and this only has to be longer.
DEFAULT_COMMAND = (
    "agy --output-format stream-json --dangerously-skip-permissions"
    ' --disable-slash-commands --print-timeout 24h {model_flag} -p="$(cat {prompt_path})"'
)

#: Seen verbatim on a signed-out run of 1.2.2.
DEFAULT_WAITING_PATTERNS = (
    r"Waiting for authentication",
    r"paste the authorization code",
)

_STDIN_DELIVERY = re.compile(r"<\s*\{prompt_path\}")

#: `step_type` values that are bookkeeping rather than work.
_QUIET_STEPS = frozenset({"user_input", "planner_response", "checkpoint"})


class AntigravityAdapter(BaseAdapter):
    name = "antigravity"
    binary = "agy"
    default_command = DEFAULT_COMMAND
    default_waiting_patterns = DEFAULT_WAITING_PATTERNS
    install_hint = (
        "curl -fsSL https://antigravity.google/cli/install.sh | bash, "
        "then run `agy` once to sign in"
    )
    #: A Gemini API key alone does nothing: `agy` also needs `modelProvider`
    #: set in its own settings file, which is the user's to write.
    credential_env: ClassVar[dict[str, str]] = {}

    async def preflight(self) -> PreflightReport:
        if await self.locate() is None:
            return PreflightReport(harness=self.name, where=self.where)
        help_text = await self.help_text()
        requirements = (
            self.flag_requirement("headless", help_text, "--print"),
            _prompt_requirement(help_text),
            self.flag_requirement("auto_approve", help_text, "--dangerously-skip-permissions"),
            await self.check_exit_codes(),
        )
        return await self.report(requirements, self._command_notes(help_text))

    def _command_notes(self, help_text: str) -> list[str]:
        command = self.config.command
        notes: list[str] = []
        if "--print-timeout" in help_text and "--print-timeout" not in command:
            notes.append(
                "command has no --print-timeout: agy stops print mode after 5 minutes by "
                "default, so any longer task is cut off"
            )
        if _STDIN_DELIVERY.search(command):
            notes.append('agy -p does not read stdin; attach the brief: -p="$(cat {prompt_path})"')
        if "--dangerously-skip-permissions" not in command:
            notes.append(
                "command does not skip permissions: headless agy soft-denies tool calls "
                "that need approval"
            )
        return notes

    async def auth_status(self) -> tuple[bool | None, str]:
        """`agy models` refuses in under a second when signed out, and never
        spends a model call when signed in."""
        code, _, err = await self.probe("models", seconds=30)
        if code == 0:
            return True, "signed in (or API-key mode configured)"
        said = stream.one_line(err.strip().splitlines()[-1] if err.strip() else "")
        return False, f"run `agy` once to sign in ({said})" if said else "run `agy` once to sign in"

    def describe_activity(self, log_text: str, lines: int = 20) -> str:
        return "\n".join(describe_events(log_text, limit=lines))

    def parse_result(self, log_text: str, exit_code: int) -> ResultSummary:
        result: dict[str, Any] | None = None
        conversation_id = None
        for event in stream.objects(log_text):
            if event.get("event") == "result" and isinstance(event.get("result"), dict):
                result = event["result"]
            elif event.get("event") == "init":
                conversation_id = event.get("conversation_id")

        if result is None:
            return ResultSummary(
                ok=exit_code == 0,
                summary=stream.fallback_summary(log_text),
                exit_code=exit_code,
                detail={"parsed": False},
            )
        status = str(result.get("status", "")).upper()
        error = str(result.get("error") or "").strip()
        reported_ok = status != "ERROR" and not error
        return ResultSummary(
            ok=exit_code == 0 and reported_ok,
            summary=str(result.get("response") or "").strip() or error,
            exit_code=exit_code,
            detail={
                "parsed": True,
                "status": status or None,
                "error": error or None,
                "num_turns": result.get("num_turns"),
                "duration_seconds": result.get("duration_seconds"),
                "usage": result.get("usage"),
                "conversation_id": result.get("conversation_id") or conversation_id,
                "agrees_with_exit_code": reported_ok == (exit_code == 0),
            },
        )


def _prompt_requirement(help_text: str) -> Requirement:
    if "--print" in help_text:
        return Requirement("prompt_input", True, "takes the prompt as the value of -p")
    return Requirement("prompt_input", False, "`agy --help` lists no --print")


def describe_events(log_text: str, limit: int = 20) -> list[str]:
    """`agy`'s stream-json as sentences for a slot card."""
    described: list[str] = []
    for event in stream.events(log_text):
        if isinstance(event, str):
            if not event.startswith("https://"):  # a sign-in URL is noise on a card
                described.append(stream.one_line(event))
            continue
        line = _describe(event)
        if line:
            described.append(line)
    return stream.last_lines(described, limit)


def _describe(event: dict[str, Any]) -> str:
    kind = event.get("event")
    if kind == "init":
        return "started"
    if kind == "result":
        result = event.get("result") or {}
        if str(result.get("status", "")).upper() == "ERROR" or result.get("error"):
            return stream.one_line(f"failed: {result.get('error') or 'error'}")
        return "done"
    if kind == "step_update":
        step = event.get("step_update") or {}
        step_type = str(step.get("step_type", ""))
        if step.get("state") != "DONE" or not step_type or step_type in _QUIET_STEPS:
            return ""
        if step_type == "error_message":
            return "error"
        return stream.one_line(step_type.replace("_", " "))
    return ""
