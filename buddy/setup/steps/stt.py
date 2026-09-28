"""Step 6: cache a whisper model and prove it transcribes.

The verify is the point. A model that downloaded but cannot run - wrong
compute type, a broken cache, a CPU without the instructions CTranslate2
wants - is indistinguishable from a working one until the first time you
speak, which is the worst moment to find out. So setup transcribes a bundled
two-second clip and compares it to the text it knows is in it.
"""

from __future__ import annotations

from pathlib import Path

from buddy.setup import BaseStep, CheckResult, SetupContext
from buddy.voice import stt

ASSETS = Path(__file__).resolve().parent.parent / "assets"
TEST_CLIP = ASSETS / "test-clip.wav"
TEST_CLIP_TEXT = "Buddy, this is a test of the microphone."


class SttStep(BaseStep):
    name = "stt"
    title = "Speech to text"
    number = "6"

    async def applies(self, ctx: SetupContext) -> str:
        if ctx.options.no_voice:
            return "--no-voice"
        return ""

    def _model(self, ctx: SetupContext) -> str:
        return stt.model_for(ctx.platform().gpu.kind)

    async def contribute(self, ctx: SetupContext) -> None:
        voice = ctx.section("voice")
        voice["stt_backend"] = "whisper"
        voice["stt_model"] = f"faster-whisper:{self._model(ctx)}"

    async def check(self, ctx: SetupContext) -> CheckResult:
        model = self._model(ctx)
        pins = {"stt_model": model}
        try:
            cached = stt.is_cached(model, ctx.platform().gpu.kind)
        except stt.SttNotInstalled as exc:
            return CheckResult(False, str(exc), pins)
        return CheckResult(
            satisfied=cached,
            detail=f"faster-whisper:{model}" if cached else f"{model} is not cached yet",
            pins=pins,
        )

    async def act(self, ctx: SetupContext) -> None:
        model = self._model(ctx)
        size = stt.MODEL_SIZES_MB.get(model, 0)
        try:
            if stt.is_cached(model, ctx.platform().gpu.kind):
                return
        except stt.SttNotInstalled as exc:
            raise self.fail(
                str(exc),
                hint="Or run setup with --no-voice to skip speech entirely.",
            ) from exc

        # Nothing multi-hundred-megabyte downloads without saying how
        # big it is and getting a yes.
        ctx.ui.say(f"The {model} whisper model is about {size} MB and is not cached yet.")
        if not ctx.ui.confirm("Download it now?"):
            raise self.fail(
                "declined the whisper model download.",
                hint="Re-run with --no-voice, or allow the download next time.",
            )
        await stt.ensure_model(model, ctx.platform().gpu.kind)

    async def verify(self, ctx: SetupContext) -> str:
        model = self._model(ctx)
        if not TEST_CLIP.is_file():  # pragma: no cover - the clip ships in the wheel
            raise self.fail(f"the bundled test clip is missing: {TEST_CLIP}")
        try:
            result = await stt.transcribe_file(
                TEST_CLIP, model=model, gpu_kind=ctx.platform().gpu.kind
            )
        except stt.SttError as exc:
            raise self.fail(str(exc), hint="Re-run: buddy setup --force stt") from exc

        if not stt.looks_like(result.text, TEST_CLIP_TEXT):
            raise self.fail(
                f"the model loaded but transcribed the test clip as {result.text!r}, "
                f"which is not {TEST_CLIP_TEXT!r}",
                hint="A corrupt cache is the usual cause; delete it and re-run --force stt.",
            )
        return f"faster-whisper:{model} transcribed the test clip correctly"
