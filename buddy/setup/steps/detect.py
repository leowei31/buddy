"""Step 1: the platform report every later step reads.

The only step with nothing to act on: it observes and
writes, and everything downstream decides from what it wrote rather than
probing again.
"""

from __future__ import annotations

from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.setup.platform import PlatformReport, detect


class PlatformStep(BaseStep):
    name = "platform"
    title = "Platform"
    number = "1"

    async def check(self, ctx: SetupContext) -> CheckResult:
        if ctx.report is None:
            ctx.report = await detect()
        report = ctx.report
        written = PlatformReport.read(ctx.paths.platform_report)
        return CheckResult(
            # Satisfied only when what is on disk still describes this
            # machine: a GPU appearing, or Docker starting, has to reach the
            # steps that decide on it.
            satisfied=written == report,
            detail=", ".join(f"{name} {value}" for name, value in report.rows()),
            pins={
                "system": report.system,
                "arch": report.arch,
                "package_manager": report.package_manager,
                "gpu": report.gpu.kind,
                "docker": "running" if report.docker.running else "no",
            },
        )

    async def act(self, ctx: SetupContext) -> None:
        report = ctx.platform()
        if not report.supported:
            raise self.fail(
                f"{report.system or 'this OS'} is not supported. Buddy is built on tmux, so "
                "Windows is out of scope; WSL2 works and is treated as Linux.",
                hint="Run Buddy under WSL2, macOS, or Linux.",
            )
        for name, value in report.rows():
            ctx.ui.detail(f"{name:>16}: {value}")
        report.write(ctx.paths.platform_report)
