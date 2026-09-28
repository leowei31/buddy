"""Step 3: Docker, and on Linux with an NVIDIA GPU the container toolkit.

Docker is needed by two optional things: the local Fish TTS server and
the harness sandbox. Neither is required to orchestrate agents,
so a machine without Docker is told what it is giving up and setup carries on
- stopping here would be stopping for something Buddy does not need to run.

On macOS setup does not install Docker: Desktop and OrbStack are GUI
applications with licence terms, and setup never installs things the
user did not see listed.
"""

from __future__ import annotations

from dataclasses import replace

from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.setup.platform import detect_docker, run

DESKTOP_URL = "https://www.docker.com/products/docker-desktop/"
ORBSTACK_URL = "https://orbstack.dev"

#: The convenience script Docker themselves publish for Linux.
LINUX_INSTALL = "curl -fsSL https://get.docker.com | sh"


class DockerStep(BaseStep):
    name = "docker"
    title = "Docker"
    number = "3"

    async def check(self, ctx: SetupContext) -> CheckResult:
        docker = await detect_docker()
        # The report is refreshed here because this is the one step that can
        # change it: step 7 asks `ctx.platform()` whether Docker is running.
        if ctx.report is not None:
            ctx.report = replace(ctx.report, docker=docker)

        report = ctx.platform()
        wants_toolkit = report.system == "linux" and report.gpu.kind == "cuda"
        satisfied = docker.running and (docker.nvidia_toolkit or not wants_toolkit)
        pins = {"docker": docker.version, "nvidia_toolkit": str(docker.nvidia_toolkit).lower()}
        return CheckResult(satisfied=satisfied, detail=docker.describe(), pins=pins)

    async def act(self, ctx: SetupContext) -> None:
        report = ctx.platform()
        docker = report.docker

        if not docker.installed:
            if report.is_macos:
                ctx.ui.warn(
                    "Docker is not installed. It is only needed for the local Fish TTS "
                    "server and the harness sandbox, so setup will carry on without it."
                )
                ctx.ui.detail(f"Docker Desktop: {DESKTOP_URL}")
                ctx.ui.detail(f"OrbStack (lighter, Mac-native): {ORBSTACK_URL}")
                ctx.ui.detail("Install one, then: buddy setup --force docker")
                return
            ctx.ui.say("Docker is not installed.")
            ctx.ui.warn(f"The official installer needs sudo: {LINUX_INSTALL}")
            if not ctx.ui.confirm("Run it now?", default=False):
                ctx.ui.detail("Skipped. TTS will use the cloud backend and the sandbox is off.")
                return
            code, out, err = await run("sh", "-c", LINUX_INSTALL, seconds=900)
            if code != 0:
                raise self.fail(
                    f"the Docker install script failed ({code}): "
                    f"{(err or out).strip().splitlines()[-1:] or ['no output']}",
                    hint=f"Run it yourself to see the whole error: {LINUX_INSTALL}",
                )
            return

        if not docker.running:
            ctx.ui.warn(f"Docker is installed but not running ({docker.detail}).")
            ctx.ui.detail(
                "Start Docker Desktop or OrbStack"
                if report.is_macos
                else "Start it with: sudo systemctl start docker"
            )
            return

        if report.system == "linux" and report.gpu.kind == "cuda" and not docker.nvidia_toolkit:
            ctx.ui.warn(
                "An NVIDIA GPU is present but the container toolkit is not, so the local "
                "Fish server would not see the GPU."
            )
            ctx.ui.detail(
                "Install nvidia-container-toolkit for your distribution, then: "
                "buddy setup --force docker"
            )

    async def verify(self, ctx: SetupContext) -> str:
        """Never fails the run.

        Docker being absent costs local TTS and the sandbox, neither of which
        Buddy needs to schedule agents. Saying so and moving on is the honest
        outcome; stopping setup would not be.
        """
        result = await self.check(ctx)
        if result.satisfied:
            return result.detail
        return f"{result.detail} - local TTS and the harness sandbox are unavailable"
