"""`kill_and_reap` actually bounds the wait, even when the process does not
cooperate (buddy/processes.py).

Found by a real timeout that should have been impossible: a probe declared
`seconds=25` blocked past pytest's own 300s ceiling, because `proc.kill()`
followed by an unbounded `await proc.wait()` assumed a killed process reaps
promptly - true until the reason it needed killing was exactly that
assumption failing (a process stuck in an uninterruptible wait ignores
SIGKILL until whatever it is blocked on resolves, which may be never).

A real reproduction needs a process genuinely stuck in an uninterruptible
kernel wait, which is not something a test can safely construct. What is
tested instead is the actual mechanism: `wait()` is bounded regardless of
whether the process ever exits.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from buddy.processes import REAP_SECONDS, communicate, kill_and_reap


class NeverReaps:
    """Stands in for a process that ignores SIGKILL - `wait()` never returns."""

    def __init__(self) -> None:
        self.killed = False

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        await asyncio.sleep(10_000)  # far longer than any test should wait
        return 0  # pragma: no cover - never reached


class ReapsImmediately:
    def __init__(self) -> None:
        self.killed = False

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return -9


async def test_a_process_that_ignores_the_kill_does_not_hang_the_caller():
    proc = NeverReaps()
    start = time.monotonic()

    await kill_and_reap(proc, seconds=0.2)

    assert proc.killed
    assert time.monotonic() - start < 2.0, "kill_and_reap must not wait past its bound"


async def test_a_process_that_reaps_normally_is_not_held_for_the_full_bound():
    proc = ReapsImmediately()
    start = time.monotonic()

    await kill_and_reap(proc, seconds=5.0)

    assert proc.killed
    assert time.monotonic() - start < 1.0, "a clean reap must not wait out the bound"


async def test_the_default_bound_is_short():
    """A caller that does not pass `seconds` still gets a short bound - the
    whole point is that a hang here is now measured in seconds, not never."""
    assert REAP_SECONDS <= 10.0


async def test_a_process_that_has_already_exited_is_not_an_error():
    """`ProcessLookupError` from `kill()` on an already-dead process must not
    propagate - this runs from exception handlers, where a second failure
    would replace the one being reported."""

    class AlreadyGone:
        def kill(self) -> None:
            raise ProcessLookupError

        async def wait(self) -> int:
            return 0

    await kill_and_reap(AlreadyGone())  # must not raise


async def test_real_subprocess_that_ignores_sigterm_but_not_sigkill(tmp_path):
    """Against a real OS process, not a stand-in: one that traps SIGTERM,
    proving `kill()` here is SIGKILL and the process actually dies."""
    script = tmp_path / "stubborn.sh"
    script.write_text("#!/bin/sh\ntrap '' TERM\nsleep 30\n")
    script.chmod(0o755)

    proc = await asyncio.create_subprocess_exec(str(script))
    await asyncio.sleep(0.3)  # let the trap install

    start = time.monotonic()
    await kill_and_reap(proc, seconds=5.0)
    elapsed = time.monotonic() - start

    assert proc.returncode is not None, "the process must actually be reaped"
    assert elapsed < 5.0, "SIGKILL cannot be trapped, so this must not need the fallback bound"


async def test_a_read_that_is_cancelled_kills_and_reaps_the_process():
    """Cancelled mid-read - the session's tick, stopped when a conversation
    ended - a process was left running with nothing waiting on it, and was
    reported "still running" once the loop that could reap it was gone."""
    proc = await asyncio.create_subprocess_exec(
        "sleep", "30", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    reading = asyncio.create_task(communicate(proc))
    await asyncio.sleep(0.2)

    reading.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reading

    assert proc.returncode is not None, "killed and reaped, not left running"
    assert proc._transport.is_closing(), "and its pipes released"
