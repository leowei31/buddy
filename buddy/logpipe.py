"""The `pipe-pane` target: a pane's output, appended to its log, rotated at a cap.

Run by tmux as `python -I logpipe.py <log> <max-bytes>`, so it imports nothing
but the standard library and starts in tens of milliseconds.

Why not `cat >> log`, which is what this replaced: rotating a log that `cat`
is writing means swapping `cat` for another one, and tmux tears the old pipe
down with bytes still in flight. Measured against real tmux, a mid-burst swap
lost two to four lines every time. A cap nobody could apply without losing
output was a cap in name only, so logs grew without bound.

Here one process holds the pane's stream from start to finish and decides
when to switch files itself, so a byte cannot fall between them. Bytes are
written as they arrive, never line-buffered: a progress bar or a spinner with
no newline still reaches the dashboard live. A file never exceeds the cap, and
a rotation lands on a line end whenever the bytes crossing the cap contain
one. The price of being live is that a line straddling the cap can be split
across the two files; nothing is lost, and they concatenate back exactly.

One previous file is kept, `<log>.1`; an attempt's logs never exceed twice
the cap on disk.
"""

from __future__ import annotations

import os
import sys

CHUNK = 64 * 1024


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _open(path: str) -> tuple[int, int]:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    return fd, os.fstat(fd).st_size


def _boundary(chunk: bytes, limit: int) -> int:
    """Bytes of `chunk[:limit]` up to and including its last line end, or 0.

    A newline first. A bare carriage return second, because a progress bar
    redraws its line with `\r` and may never print a newline at all - but
    never the `\r` of a `\r\n`, which would split one line ending in two.
    """
    newline = chunk.rfind(b"\n", 0, limit)
    if newline >= 0:
        return newline + 1
    at = chunk.rfind(b"\r", 0, limit)
    while at >= 0 and at + 1 < len(chunk) and chunk[at + 1 : at + 2] == b"\n":
        at = chunk.rfind(b"\r", 0, at)
    return at + 1 if at >= 0 else 0


def pump(source: int, path: str, max_bytes: int) -> None:
    """Copy `source` into `path`, rotating to `path.1` past `max_bytes`.

    A file never exceeds the cap. A rotation lands at the last line end that
    fits; failing that, before the next chunk when the file already ends a
    line; failing both, at exactly the cap.
    """
    rotated = path + ".1"
    out, size = _open(path)
    ends_a_line = True
    try:
        while True:
            chunk = os.read(source, CHUNK)
            if not chunk:
                return
            while chunk:
                room = max_bytes - size
                if max_bytes <= 0 or len(chunk) <= room:
                    _write_all(out, chunk)
                    size += len(chunk)
                    ends_a_line = chunk.endswith((b"\n", b"\r"))
                    break
                cut = _boundary(chunk, room)
                if cut == 0 and not (ends_a_line and size > 0):
                    cut = room  # no line end fits and none to rotate at: cut at the cap
                if cut:
                    _write_all(out, chunk[:cut])
                    size += cut
                    ends_a_line = chunk[:cut].endswith((b"\n", b"\r"))
                    chunk = chunk[cut:]
                os.close(out)
                os.replace(path, rotated)
                out, size = _open(path)
                ends_a_line = True
    finally:
        os.close(out)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: logpipe.py <log> <max-bytes>", file=sys.stderr)
        return 2
    pump(sys.stdin.fileno(), argv[0], int(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
