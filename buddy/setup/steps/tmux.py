"""Step 11: the `buddy` session with its seven windows.

Idempotent because `ensure_session` is: it creates what is missing and
touches nothing that already exists, which is the same call the orchestrator
makes on every start.
"""

from __future__ import annotations

from buddy.models import SLOT_NAMES
from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.tmux_runner import TmuxError, TmuxRunner


class TmuxStep(BaseStep):
    name = "tmux"
    title = "tmux session"
    number = "11"

    async def check(self, ctx: SetupContext) -> CheckResult:
        runner = TmuxRunner(socket_name=ctx.tmux_socket)
        try:
            panes = await runner.panes()
        except (TmuxError, FileNotFoundError) as exc:
            return CheckResult(False, str(exc))
        present = sorted(set(panes) & set(SLOT_NAMES))
        missing = [name for name in SLOT_NAMES if name not in panes]
        return CheckResult(
            satisfied=not missing,
            detail=(
                f"{len(present)} of {len(SLOT_NAMES)} windows"
                + (f", missing {', '.join(missing)}" if missing else "")
            ),
            pins={"slots": ",".join(SLOT_NAMES)},
        )

    async def act(self, ctx: SetupContext) -> None:
        try:
            await TmuxRunner(socket_name=ctx.tmux_socket).ensure_session()
        except (TmuxError, FileNotFoundError) as exc:
            raise self.fail(
                f"could not create the tmux session: {exc}",
                hint="Check that tmux runs at all: `tmux -V`, then `tmux new -d -s probe`.",
            ) from exc

    async def verify(self, ctx: SetupContext) -> str:
        result = await self.check(ctx)
        if not result.satisfied:
            raise self.fail(f"the session is not right: {result.detail}")
        return f"session 'buddy' with {', '.join(SLOT_NAMES)}"
