"""Running a subprocess without leaking its pipes, and without hanging on it.

Every module that shells out - git, tmux, a harness CLI, the package manager
- creates an asyncio subprocess with two pipes, and `communicate()` does not
close their transports. They are released when the process object is
collected, which in a long-running session is fine and at interpreter
teardown is not: Python warns `unclosed transport`, and under the suite's
`filterwarnings = ["error"]` that warning lands on whichever test happened to
be running when the collector got there.

That is how `release()` was found - a green suite that failed one arbitrary
test per run. The fix belongs in one place rather than in all eight.

`kill_and_reap()` exists for the same reason, found the same way: a probe
timing out is supposed to be the end of the story - `proc.kill()` then
`await proc.wait()` - but `wait()` has no bound of its own. A process stuck
in an uninterruptible wait against something genuinely broken (a wedged
Docker socket was what surfaced it) does not die from SIGKILL, so the second
await hangs too, indefinitely, and the very timeout that was supposed to
guarantee a probe never blocks forever is defeated by its own cleanup step.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any


def release(proc: Any) -> None:
    """Close a finished process's pipe transports.

    Safe to call more than once, and safe on a process that has already been
    cleaned up - which is why it is written to be called from a `finally`.
    """
    transport = getattr(proc, "_transport", None)
    if transport is None:
        return
    with contextlib.suppress(Exception):
        transport.close()


#: How long a killed process gets to actually exit before it is abandoned
#: rather than waited on further. Reaping is normally instant; this is only
#: ever reached by the process that would otherwise have hung forever.
REAP_SECONDS = 5.0


async def kill_and_reap(proc: Any, *, seconds: float = REAP_SECONDS) -> None:
    """`proc.kill()`, then wait for it to actually exit - but not forever.

    A process blocked in an uninterruptible wait ignores SIGKILL until
    whatever it is blocked on resolves, which may be never. Every caller of
    this was written as `proc.kill(); await proc.wait()`, on the assumption
    that a killed process reaps quickly - true until the reason it needed
    killing was exactly that assumption failing. Orphaning a process Buddy
    can no longer wait on is the right trade: worse than a clean reap, far
    better than a CLI that hangs on its own declared timeout.
    """
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    with contextlib.suppress(Exception):
        async with asyncio.timeout(seconds):
            await proc.wait()


async def communicate(proc: Any) -> tuple[bytes, bytes]:
    """`proc.communicate()`, with the pipes released however it ends - and,
    when it is interrupted, the process killed and reaped before the
    interruption carries on.

    Cancelled mid-read, a process used to be left running with nothing
    waiting on it: its pipes went unclosed, and `Popen` warned that it was
    still running when it was collected - after the loop that could have
    reaped it was gone. The session's tick, stopped at the end of a
    conversation, was how - found hammering one test on Linux under load.
    """
    try:
        out, err = await proc.communicate()
    except BaseException:
        await kill_and_reap(proc)
        raise
    finally:
        release(proc)
    return out or b"", err or b""
