"""Reading what a task wrote.

`pipe-pane` captures the pane raw, ANSI escapes and all, which is what
the dashboard's xterm.js view wants and what everything else has to
undo. The manager reads tails for stall and waiting-input detection,
adapters parse the same text for a result summary, and the brain is
handed it with the escapes stripped.

The wrapper script's sentinels are parsed here too. They are the redundant
completion record, never the signal: `pane_dead_status` is the signal. The
sentinel carries the task id so a stale line from a previous task in the same
window can never be mistaken for this one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: CSI / OSC escape sequences, plus the carriage returns a terminal leaves behind.
_ANSI = re.compile(
    r"""
    \x1b \[ [0-?]* [ -/]* [@-~]     # CSI ... final byte
    | \x1b \] .*? (?: \x07 | \x1b\\ )  # OSC ... BEL or ST
    | \x1b [@-Z\\-_]                # two-character escapes
    """,
    re.VERBOSE | re.DOTALL,
)

START_SENTINEL = "__BUDDY_START__"
DONE_SENTINEL = "__BUDDY_DONE__"

_DONE_RE = re.compile(rf"{DONE_SENTINEL}\s+(\S+)\s+(-?\d+)")
_START_RE = re.compile(rf"{START_SENTINEL}\s+(\S+)\s+attempt=(\d+)")


def strip_ansi(text: str) -> str:
    """Plain text, for anything that is not a terminal."""
    return _ANSI.sub("", text).replace("\r\n", "\n").replace("\r", "\n")


def tail(text: str, lines: int) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:]) if lines > 0 else ""


def read_tail(path: Path, lines: int = 60, *, max_bytes: int = 256_000, clean: bool = True) -> str:
    """The end of a log without reading a log that may be hundreds of MB.

    Reads at most the last `max_bytes`, which is why the first line of the
    result may be a partial one; callers want a tail, not a parse.
    """
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        raw = handle.read()
    text = raw.decode(errors="replace")
    if clean:
        text = strip_ansi(text)
    return tail(text, lines)


@dataclass(frozen=True)
class Sentinels:
    """What the wrapper script recorded about this attempt."""

    started: bool = False
    attempt: int | None = None
    finished: bool = False
    exit_code: int | None = None


#: How much of each end of a log `read_sentinels` reads. The start sentinel
#: is the first thing run.sh prints and the done sentinel the last, so they
#: are always within this of an end.
SENTINEL_WINDOW = 256_000


def rotated(path: Path) -> Path:
    """The previous log `logpipe.py` rotated this one into."""
    return path.with_name(path.name + ".1")


def _ends(path: Path, window: int = SENTINEL_WINDOW) -> str:
    """The first and last `window` bytes of a file, never the middle."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            head = handle.read(window)
            if size <= 2 * window:
                return (head + handle.read()).decode(errors="replace")
            handle.seek(size - window)
            return (head + b"\n" + handle.read()).decode(errors="replace")
    except OSError:
        return ""


def read_sentinels(path: Path, task_id: str) -> Sentinels:
    """Find this task's sentinels in a log.

    Matching on the task id is the point: a window is reused across tasks, so
    a `__BUDDY_DONE__` from a previous occupant must never be read as this
    one's completion.

    Only the ends are read, of the log and of the file it was last rotated
    into - this runs on the restart path, where logs are at their largest.
    """
    text = strip_ansi(_ends(rotated(path)) + "\n" + _ends(path))
    started = attempt = None
    for match in _START_RE.finditer(text):
        if match.group(1) == task_id:
            started, attempt = True, int(match.group(2))
    exit_code = None
    finished = False
    for match in _DONE_RE.finditer(text):
        if match.group(1) == task_id:
            finished, exit_code = True, int(match.group(2))
    return Sentinels(
        started=bool(started),
        attempt=attempt,
        finished=finished,
        exit_code=exit_code,
    )
