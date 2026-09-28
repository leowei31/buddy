"""Steps 7 and 7b: the Fish TTS backend, local if the hardware allows.

One step, two act paths: local is the primary and cloud is the fallback.
Setup never stands up a local server on a machine that cannot run it well -
it configures cloud and *says why*, rather than shipping a three-second
turn-around.

What this step does not do is measure time to first audio. Fish's wire
format lives in `voice/tts.py` and nowhere else, and `buddy doctor` is what
drives it for a real synthesis. This step configures the backend, proves the
container or the credential is there, and points at `buddy doctor` for the
timing.
"""

from __future__ import annotations

from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.setup.platform import MIN_LOCAL_TTS_VRAM_MB, run
from buddy.setup.steps.keys import FISH_KEY_ENV, secret

CONTAINER = "buddy-fish"
IMAGE = "fishaudio/fish-speech:server-cuda"
LOCAL_URL = "http://127.0.0.1:8080"
CLOUD_URL = "https://api.fish.audio"


class TtsStep(BaseStep):
    name = "tts"
    title = "Text to speech"
    number = "7"

    async def applies(self, ctx: SetupContext) -> str:
        if ctx.options.no_voice:
            return "--no-voice"
        return ""

    def _wants_local(self, ctx: SetupContext) -> tuple[bool, str]:
        """Local or cloud, and the reason, which is always said."""
        if ctx.options.cloud_tts:
            return False, "--cloud-tts was passed"
        report = ctx.platform()
        gpu = report.gpu
        if gpu.kind != "cuda":
            found = (
                "no GPU" if gpu.kind == "none" else f"{gpu.kind}, and the Fish image is CUDA-only"
            )
            return False, f"local needs an NVIDIA GPU ({found})"
        if not gpu.can_run_local_tts:
            return False, (
                f"local needs about {MIN_LOCAL_TTS_VRAM_MB // 1000} GB of VRAM and this GPU "
                f"has {gpu.vram_mb} MiB"
            )
        if not report.docker.running:
            return False, "local runs in Docker, which is not running here"
        return True, "NVIDIA GPU with enough VRAM, and Docker is running"

    async def contribute(self, ctx: SetupContext) -> None:
        local, _ = self._wants_local(ctx)
        self._plan_local(ctx) if local else self._plan_cloud(ctx)

    async def check(self, ctx: SetupContext) -> CheckResult:
        local, why = self._wants_local(ctx)
        if local:
            code, out, _ = await run("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER)
            healthy = code == 0 and out.strip() == "true"
            return CheckResult(
                satisfied=healthy,
                detail=f"{CONTAINER} is running" if healthy else f"{CONTAINER} is not running",
                pins={"tts_backend": "fish_local", "image": IMAGE},
            )
        has_key = bool(secret(FISH_KEY_ENV))
        return CheckResult(
            satisfied=has_key,
            detail=f"fish_cloud ({why})" if has_key else f"fish_cloud, but no {FISH_KEY_ENV}",
            pins={"tts_backend": "fish_cloud"},
        )

    async def act(self, ctx: SetupContext) -> None:
        local, why = self._wants_local(ctx)
        if not local:
            # Configure cloud and say exactly why local was skipped.
            ctx.ui.say(f"Using cloud TTS: {why}.")
            return

        ctx.ui.say(f"Standing up the local Fish server: {why}.")
        ctx.ui.detail(f"Pulling {IMAGE} - this is several GB.")
        if not ctx.ui.confirm("Pull it now?"):
            ctx.ui.detail("Skipped; configuring cloud instead.")
            return

        code, out, err = await run("docker", "pull", IMAGE, seconds=3600)
        if code != 0:
            raise self.fail(
                f"docker pull {IMAGE} failed ({code}): {(err or out).strip()[-300:]}",
                hint=f"Pull it yourself, then: buddy setup --force {self.name}",
            )

        await run("docker", "rm", "-f", CONTAINER)
        code, out, err = await run(
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER,
            "--restart",
            "unless-stopped",
            "--gpus",
            "all",
            # Bound to loopback. The server has no auth of its own, so
            # the only thing keeping it private is that nothing else can
            # reach it.
            "-p",
            "127.0.0.1:8080:8080",
            "-v",
            f"{ctx.paths.home / 'fish' / 'checkpoints'}:/opt/fish-speech/checkpoints",
            IMAGE,
            seconds=300,
        )
        if code != 0:
            raise self.fail(
                f"could not start {CONTAINER} ({code}): {(err or out).strip()[-300:]}",
                hint="Check `docker logs buddy-fish`.",
            )

    def _plan_local(self, ctx: SetupContext) -> None:
        voice = ctx.section("voice")
        voice["tts_backend"] = "fish_local"
        voice["tts_fallback"] = "fish_cloud" if secret(FISH_KEY_ENV) else ""
        block = ctx.section("voice", "fish_local")
        block["base_url"] = LOCAL_URL
        block["managed"] = True
        block["streaming"] = "auto"  # the client probes for the websocket at startup
        block["checkpoints"] = str(ctx.paths.home / "fish" / "checkpoints")
        block["references"] = str(ctx.paths.home / "fish" / "references")

    def _plan_cloud(self, ctx: SetupContext) -> None:
        voice = ctx.section("voice")
        voice["tts_backend"] = "fish_cloud"
        voice["tts_fallback"] = ""
        block = ctx.section("voice", "fish_cloud")
        block["base_url"] = CLOUD_URL
        block["api_key_env"] = FISH_KEY_ENV
        block.setdefault("model", "s2-pro")

    async def verify(self, ctx: SetupContext) -> str:
        local, _ = self._wants_local(ctx)
        owed = "`buddy doctor` measures its time to first audio"
        if local:
            code, out, _ = await run("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER)
            if code != 0 or out.strip() != "true":
                raise self.fail(
                    f"{CONTAINER} did not stay running",
                    hint="Check `docker logs buddy-fish`.",
                )
            return f"fish_local on {LOCAL_URL} ({owed})"

        if not secret(FISH_KEY_ENV):
            # Not a failure: Buddy runs perfectly well without a voice, and
            # stopping setup over it would be stopping over a nicety.
            ctx.ui.warn(f"No {FISH_KEY_ENV}, so Buddy will not speak. Everything else works.")
            return f"fish_cloud configured but unauthenticated - set {FISH_KEY_ENV} to enable it"
        return f"fish_cloud on {CLOUD_URL} ({owed})"
