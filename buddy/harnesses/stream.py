"""Reading a harness's JSON-lines log.

Every supported CLI can print one JSON object per line, and every one of
them wraps it in the same noise: the pane's escape codes, Buddy's own
sentinels, and whatever the CLI wrote to stderr on the way. What each object
*means* belongs to that harness's adapter; getting the objects out of a pane
log is the same job four times, so it lives here once.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from buddy.logs import strip_ansi

#: How wide one described line may be on an agent card.
WIDTH = 90


def events(log_text: str) -> Iterator[dict[str, Any] | str]:
    """Each line of the log: a parsed object, or the plain text it was.

    Plain lines are yielded too, stripped, because a harness that fails
    before its JSON starts - a bad flag, a missing login - says why in prose,
    and that sentence is the most useful thing on the card. Buddy's own
    sentinels and blank lines are dropped.
    """
    for raw in strip_ansi(log_text).splitlines():
        line = raw.strip()
        if not line or line.startswith("__BUDDY_"):
            continue
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if isinstance(parsed, dict):
                yield parsed
                continue
        yield line


def objects(log_text: str) -> list[dict[str, Any]]:
    """Only the parsed objects, in order."""
    return [event for event in events(log_text) if isinstance(event, dict)]


def one_line(text: object, width: int = WIDTH) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def detail_of(payload: object, keys: tuple[str, ...]) -> str:
    """The first field of a tool call's input that says what it is doing."""
    if not isinstance(payload, dict):
        return ""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return one_line(value)
    return ""


def fallback_summary(log_text: str, lines: int = 5) -> str:
    """No parseable result: the last few plain lines are better than nothing."""
    clean = [line for line in strip_ansi(log_text).splitlines() if line.strip()]
    return "\n".join(clean[-lines:])


def last_lines(described: list[str], limit: int) -> list[str]:
    return described[-limit:] if limit > 0 else described
