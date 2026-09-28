"""Step 12: run the same `buddy doctor` the user will run.

Deliberately not a private re-implementation. If setup's last word and
`buddy doctor` could disagree, one of them is lying, and the one people
believe is the one they can run again tomorrow.
"""

from __future__ import annotations

from buddy.setup import BaseStep, CheckResult, SetupContext


class DoctorStep(BaseStep):
    name = "doctor"
    title = "Doctor"
    number = "12"

    async def check(self, ctx: SetupContext) -> CheckResult:
        # Always runs: it is the summary, not a thing to be satisfied.
        return CheckResult(False, "not run yet")

    async def act(self, ctx: SetupContext) -> None:
        from buddy.cli import run_doctor

        problems = await run_doctor(home=ctx.home)
        ctx.results["doctor_problems"] = str(problems)

    async def verify(self, ctx: SetupContext) -> str:
        problems = int(ctx.results.get("doctor_problems", "0"))
        if problems:
            # Not a failure of setup: doctor has already printed the named
            # list of what is wrong and how to fix it, which is what this
            # step exists to produce.
            return f"{problems} thing(s) still need attention - see the table above"
        return "everything green"
