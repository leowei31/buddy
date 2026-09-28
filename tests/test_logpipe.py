"""The rotating `pipe-pane` writer (buddy/logpipe.py).

`test_tmux_runner.py` proves one rotation against real tmux. This proves the
invariants across many, with every rotated file kept for inspection: the
files, in order, are exactly the input; no file exceeds the cap unless one
line alone is longer than it; and a file is only ever cut at a line end.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from buddy import logpipe


def run_pump(tmp_path: Path, data: list[bytes], cap: int, monkeypatch) -> list[bytes]:
    """Feed `data` chunk by chunk; return every file's bytes, oldest first."""
    log = tmp_path / "attempt-1.log"
    kept: list[Path] = []
    real_replace = os.replace

    def keep_every_rotation(src, dst):
        # Instead of overwriting `.1`, number each rotation so none is lost.
        target = tmp_path / f"rotated-{len(kept):04d}"
        real_replace(src, target)
        kept.append(target)

    monkeypatch.setattr(logpipe.os, "replace", keep_every_rotation)
    read_end, write_end = os.pipe()
    chunks = iter(data)
    real_read = os.read

    def read_one_chunk(fd, n):
        # One scripted chunk per read, so chunk boundaries are the test's.
        if fd != read_end:
            return real_read(fd, n)
        return next(chunks, b"")

    monkeypatch.setattr(logpipe.os, "read", read_one_chunk)
    try:
        logpipe.pump(read_end, str(log), cap)
    finally:
        os.close(read_end)
        os.close(write_end)
    return [path.read_bytes() for path in kept] + [log.read_bytes()]


@pytest.mark.parametrize("seed", range(25))
def test_many_rotations_keep_every_byte_in_order_and_respect_the_cap(seed, tmp_path, monkeypatch):
    rng = random.Random(seed)
    cap = rng.choice([64, 200, 1000])
    lines = [
        (b"x" * rng.choice([0, 3, 40, 150, 900, 1500])) + rng.choice([b"\n", b"\r\n"])
        for _ in range(rng.randint(50, 400))
    ]
    stream = b"".join(lines)
    # Chunk boundaries land anywhere, including mid-line.
    data, i = [], 0
    while i < len(stream):
        step = rng.randint(1, 2048)
        data.append(stream[i : i + step])
        i += step

    files = run_pump(tmp_path, data, cap, monkeypatch)

    assert b"".join(files) == stream, "a byte was lost, duplicated or reordered"
    assert all(len(body) <= cap for body in files), "a file exceeded the cap"


def test_a_rotation_lands_on_a_line_end_when_one_fits(tmp_path, monkeypatch):
    chunks = [b"a" * 50 + b"\n", b"b" * 20 + b"\n" + b"c" * 30]
    first, second = run_pump(tmp_path, chunks, 64, monkeypatch)
    assert first == b"a" * 50 + b"\n"
    assert second == b"b" * 20 + b"\n" + b"c" * 30


def test_a_rotation_never_splits_a_crlf(tmp_path, monkeypatch):
    chunk = b"a" * 40 + b"\r\n" + b"b" * 30 + b"\r\n"
    first, second = run_pump(tmp_path, [chunk], 60, monkeypatch)
    assert first.endswith(b"\r\n") and not second.startswith(b"\n")


def test_a_progress_bar_that_never_prints_a_newline_still_rotates(tmp_path, monkeypatch):
    """`\r`-only output was the way a capped log could still grow forever."""
    frames = [b"\r[" + b"=" * n + b"]" for n in range(40)]
    files = run_pump(tmp_path, frames, 200, monkeypatch)
    assert len(files) > 1
    assert all(len(body) <= 200 for body in files)
    assert all(body.endswith(b"]") or body.endswith(b"\r") for body in files[:-1])


def test_no_cap_means_one_file(tmp_path, monkeypatch):
    files = run_pump(tmp_path, [b"a\n" * 5000], 0, monkeypatch)
    assert files == [b"a\n" * 5000]


def test_output_without_a_newline_is_written_immediately(tmp_path, monkeypatch):
    """A spinner or a progress bar never ends its line; the dashboard must
    still see it, so nothing waits for a newline."""
    files = run_pump(tmp_path, [b"\r[=====>    ] 50%", b"\r[==========] 100%"], 1000, monkeypatch)
    assert files == [b"\r[=====>    ] 50%\r[==========] 100%"]


def test_a_log_is_private(tmp_path, monkeypatch):
    run_pump(tmp_path, [b"secret-ish output\n"], 0, monkeypatch)
    assert (tmp_path / "attempt-1.log").stat().st_mode & 0o777 == 0o600
