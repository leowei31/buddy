"""Step 9: enumerate audio devices and pick defaults.

Recording a level sample and playing a tone both need a device the user is
sitting in front of, so setup does the part that can be done without one -
find the devices, choose the defaults, prove they open - and leaves the
"did you hear that?" to the first voice session.
"""

from __future__ import annotations

from buddy.setup import BaseStep, CheckResult, SetupContext

HINT = (
    "Install the voice extra: uv tool install 'buddy-orchestrator[voice]' "
    "(or run setup with --no-voice)."
)


def _devices():
    """(input, output) device lists, or a reason there are none."""
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        # OSError: sounddevice is installed but PortAudio is not, which is
        # exactly what step 2 was for.
        return None, None, f"{type(exc).__name__}: {exc}"
    try:
        devices = sounddevice.query_devices()
    except Exception as exc:  # noqa: BLE001 - no audio subsystem at all
        return None, None, str(exc)
    inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
    outputs = [d for d in devices if d.get("max_output_channels", 0) > 0]
    return inputs, outputs, ""


class AudioStep(BaseStep):
    name = "audio"
    title = "Audio devices"
    number = "9"

    async def applies(self, ctx: SetupContext) -> str:
        if ctx.options.no_voice:
            return "--no-voice"
        return ""

    async def contribute(self, ctx: SetupContext) -> None:
        inputs, outputs, problem = _devices()
        if problem or not inputs or not outputs:
            return
        voice = ctx.section("voice")
        voice.setdefault("input_device", inputs[0]["name"])
        voice.setdefault("output_device", outputs[0]["name"])

    async def check(self, ctx: SetupContext) -> CheckResult:
        inputs, outputs, problem = _devices()
        if problem:
            return CheckResult(False, problem)
        if not inputs:
            return CheckResult(False, "no input device, so there is nothing to talk into")
        if not outputs:
            return CheckResult(False, "no output device, so Buddy has nothing to speak through")
        # What is *configured*, not what this run happens to have planned:
        # a machine set up yesterday must not be asked again today.
        from buddy.config import Config

        voice = Config.load(home=ctx.home).voice
        planned = ctx.section("voice")
        chosen_in = planned.get("input_device") or voice.input_device
        chosen_out = planned.get("output_device") or voice.output_device
        return CheckResult(
            satisfied=bool(chosen_in and chosen_out),
            detail=(
                f"{len(inputs)} input, {len(outputs)} output"
                if chosen_in and chosen_out
                else f"{len(inputs)} input, {len(outputs)} output, none chosen yet"
            ),
            pins={"input": str(chosen_in), "output": str(chosen_out)},
        )

    async def act(self, ctx: SetupContext) -> None:
        inputs, outputs, problem = _devices()
        if problem:
            raise self.fail(f"audio is unavailable: {problem}", hint=HINT)
        if not inputs or not outputs:
            raise self.fail(
                "this machine has no usable microphone or speaker.",
                hint="Plug one in, or run setup with --no-voice.",
            )
        voice = ctx.section("voice")
        voice["input_device"] = inputs[0]["name"]
        voice["output_device"] = outputs[0]["name"]
        ctx.ui.detail(f"input:  {voice['input_device']}")
        ctx.ui.detail(f"output: {voice['output_device']}")

    async def verify(self, ctx: SetupContext) -> str:
        voice = ctx.section("voice")
        return f"in {voice.get('input_device')!r}, out {voice.get('output_device')!r}"
