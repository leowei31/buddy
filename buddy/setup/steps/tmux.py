"""Step 11: tmux, ready to give each agent a window of its own.

Nothing is created here. An agent's window is made when it starts and
removed when it ends, so there is no fixed set of windows to prepare - only
the question of whether the tmux Buddy will drive can do what it needs:
answer at all, and report a pane's exit status, which is how Buddy knows an
agent has finished.
"""

from __future__ import annotations

from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.tmux_runner import MIN_TMUX_VERSION, TmuxError, TmuxRunner


class TmuxStep(BaseStep):
    name = "tmux"
    title = "tmux"
    number = "11"

    async def check(self, ctx: SetupContext) -> CheckResult:
        runner = TmuxRunner(socket_name=ctx.tmux_socket)
        try:
            found = await runner.check_version()
        except (TmuxError, FileNotFoundError) as exc:
            return CheckResult(False, str(exc))
        version = ".".join(map(str, found))
        return CheckResult(
            satisfied=True,
            detail=f"tmux {version}: each agent gets its own window in the '{runner.session}' "
            "session as it starts",
            pins={"tmux": version},
        )

    async def act(self, ctx: SetupContext) -> None:
        # Only reached when the check failed, and there is nothing setup can
        # do about a tmux that will not run: step 2 installs it.
        found = await self.check(ctx)
        needed = ".".join(map(str, MIN_TMUX_VERSION))
        raise self.fail(
            f"tmux is not usable: {found.detail}",
            hint=f"Buddy needs tmux {needed} or later: `tmux -V`, then "
            "`buddy setup --force packages`.",
        )

    async def verify(self, ctx: SetupContext) -> str:
        result = await self.check(ctx)
        if not result.satisfied:
            raise self.fail(f"tmux is not usable: {result.detail}")
        return result.detail
