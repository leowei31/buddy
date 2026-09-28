"""Speech to text: the model, the microphone, and push-to-talk.

Only this module may know the STT engine, so everything `faster_whisper` is
lives here - including the half `buddy setup` drives (pick a model for the
hardware, cache it, prove it transcribes).

**On "push-to-talk".** A terminal cannot give it: a TTY
reports key *presses* and never releases, so "hold the key while you speak"
is not expressible without OS-level input hooks, which on macOS means
accessibility permissions and a native dependency. What is implemented is
press-to-start, press-to-stop on the same key - the same turn-taking
guarantee (Buddy never listens unless you asked it to) with the interaction a
terminal can actually support. `hands_free` listens for a pause instead.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Sized to the hardware, not to a wish. The turbo model is
#: worth its download only where there is a GPU to run it on.
MODELS: dict[str, str] = {
    "cuda": "large-v3-turbo",
    "mps": "small",
    "none": "small",
}

#: Roughly, so setup can say what it is about to download before it starts
#: (nothing large downloads without stating its size).
MODEL_SIZES_MB: dict[str, int] = {"large-v3-turbo": 1600, "small": 480}


class SttError(Exception):
    pass


class SttNotInstalled(SttError):
    """`faster-whisper` is an optional extra, and saying so beats a traceback."""

    def __init__(self) -> None:
        super().__init__(
            "speech to text needs the 'faster-whisper' package. Install it with: "
            "uv tool install 'buddy-orchestrator[voice]' (or `uv sync --extra voice` "
            "from a source checkout)."
        )


def model_for(gpu_kind: str) -> str:
    """The whisper model this hardware should run."""
    return MODELS.get(gpu_kind, MODELS["none"])


def compute_type_for(gpu_kind: str) -> tuple[str, str]:
    """(device, compute_type) for `WhisperModel`.

    CTranslate2 has no Metal backend, so Apple Silicon runs on the CPU like
    any other machine without CUDA; int8 is what makes that bearable.
    """
    if gpu_kind == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str = ""
    duration: float = 0.0


#: Loaded models, by (name, hardware). Loading one costs seconds and holds
#: hundreds of megabytes, and a session transcribes once per utterance - so
#: reloading per call made every turn slower than the last as the abandoned
#: copies piled up. Measured at 8.4s, then 10.6s, then 13.7s for the same
#: 2-second clip before this cache existed.
_LOADED: dict[tuple[str, str], Any] = {}


def _load(name: str, gpu_kind: str, *, download: bool):
    cached = _LOADED.get((name, gpu_kind))
    if cached is not None:
        return cached

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SttNotInstalled() from exc

    device, compute = compute_type_for(gpu_kind)
    try:
        model = WhisperModel(
            name,
            device=device,
            compute_type=compute,
            local_files_only=not download,
        )
    except Exception as exc:  # noqa: BLE001 - a download or a bad model name
        raise SttError(f"could not load the {name!r} whisper model: {exc}") from exc
    _LOADED[(name, gpu_kind)] = model
    return model


def unload_models() -> None:
    """Drop the cache. For tests, and for `switch` if a model ever changes."""
    _LOADED.clear()


def is_cached(name: str, gpu_kind: str = "none") -> bool:
    """Whether the model is already on disk, without reaching the network.

    Loads it if it is there, which is the only way to know - and keeps it,
    since the next thing anyone does with a cached model is use it.
    """
    try:
        _load(name, gpu_kind, download=False)
    except SttNotInstalled:
        raise
    except SttError:
        return False
    return True


async def ensure_model(name: str, gpu_kind: str = "none") -> str:
    """Download the model if it is not cached. Returns the name.

    The load is genuinely slow and genuinely blocking, so it goes to a thread
    rather than stalling the loop.
    """
    await asyncio.to_thread(_load, name, gpu_kind, download=True)
    return name


async def transcribe_file(path: Path, *, model: str, gpu_kind: str = "none") -> Transcript:
    """Transcribe an audio file. Setup uses it on a bundled clip to prove the
    model works before anyone speaks to it."""
    return await asyncio.to_thread(_transcribe, path, model, gpu_kind)


def _transcribe(path: Path, model: str, gpu_kind: str) -> Transcript:
    loaded = _load(model, gpu_kind, download=False)
    try:
        segments, info = loaded.transcribe(str(path), beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:  # noqa: BLE001 - a bad file is not a crash
        raise SttError(f"could not transcribe {path}: {exc}") from exc
    return Transcript(
        text=text,
        language=getattr(info, "language", "") or "",
        duration=float(getattr(info, "duration", 0.0) or 0.0),
    )


def looks_like(said: str, expected: str) -> bool:
    """Whether a transcript is close enough to the known text of the clip.

    Word overlap, not equality: whisper punctuates and capitalises as it sees
    fit, and setup is testing that the model runs, not that it is perfect.
    """
    words = {word.strip(".,!?").lower() for word in said.split() if word.strip(".,!?")}
    wanted = {word.strip(".,!?").lower() for word in expected.split() if word.strip(".,!?")}
    if not wanted:
        return False
    return len(words & wanted) >= max(1, (len(wanted) + 1) // 2)


# --------------------------------------------------------------------------
# The microphone
# --------------------------------------------------------------------------

#: What whisper wants. Recording at anything else means resampling for no gain.
SAMPLE_RATE = 16000

#: A hard stop, so a forgotten keypress cannot record until the disk fills.
MAX_RECORDING_SECONDS = 120


class Microphone:
    """Records while asked to, into memory, at whisper's sample rate."""

    def __init__(self, *, device: Any = None, sample_rate: int = SAMPLE_RATE) -> None:
        self.device = device
        self.sample_rate = sample_rate

    def available(self) -> tuple[bool, str]:
        try:
            import sounddevice
        except (ImportError, OSError) as exc:
            return False, str(exc)
        try:
            devices = sounddevice.query_devices()
        except Exception as exc:  # noqa: BLE001 - no audio subsystem at all
            return False, str(exc)
        inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if not inputs:
            return False, "no input device"
        return True, str(self.device if self.device is not None else inputs[0]["name"])

    def record(self, stop: threading.Event, *, seconds: float = MAX_RECORDING_SECONDS):
        """Block until `stop` is set, returning float32 mono samples.

        Blocking is right here: this runs on the voice thread, whose
        whole job is to wait for a person.
        """
        try:
            import numpy
            import sounddevice
        except (ImportError, OSError) as exc:
            raise SttNotInstalled() from exc

        frames: list[Any] = []

        def collect(indata, _frames, _time, status) -> None:
            frames.append(indata.copy())

        with sounddevice.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=collect,
        ):
            stop.wait(timeout=seconds)
        if not frames:
            return numpy.zeros(0, dtype="float32")
        return numpy.concatenate(frames, axis=0).reshape(-1)


async def transcribe_samples(samples: Any, *, model: str, gpu_kind: str = "none") -> Transcript:
    """Transcribe recorded audio without going through a file."""
    return await asyncio.to_thread(_transcribe_samples, samples, model, gpu_kind)


def _transcribe_samples(samples: Any, model: str, gpu_kind: str) -> Transcript:
    loaded = _load(model, gpu_kind, download=False)
    try:
        segments, info = loaded.transcribe(samples, beam_size=1)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:  # noqa: BLE001
        raise SttError(f"could not transcribe the recording: {exc}") from exc
    return Transcript(
        text=text,
        language=getattr(info, "language", "") or "",
        duration=float(getattr(info, "duration", 0.0) or 0.0),
    )


class PushToTalk:
    """Press once to start listening, press again to stop.

    Reads single keypresses from the terminal in cbreak mode, so it never
    waits for Enter and never echoes what is typed while Buddy is listening.
    A terminal that is not a tty - a pipe, CI - simply reports that it cannot
    do this, rather than blocking on a stdin nobody is typing into.
    """

    def __init__(self, key: str = " ", *, stream=None) -> None:
        #: `push_to_talk` names something like "ctrl+space", which a TTY
        #: cannot distinguish from other control codes; the printable part is
        #: what is matched, and any key stops a recording in progress.
        self.key = (key or " ")[-1:] or " "
        self.stream = stream or sys.stdin

    def available(self) -> bool:
        try:
            return self.stream.isatty()
        except (AttributeError, ValueError):
            return False

    def wait_for_key(self, stop: threading.Event) -> str | None:
        """The next keypress, or None if asked to stop first."""
        if not self.available():
            return None
        import select
        import termios
        import tty

        fd = self.stream.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop.is_set():
                ready, _, _ = select.select([fd], [], [], 0.1)
                if ready:
                    return self.stream.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        return None


# --------------------------------------------------------------------------
# Hands-free turn-taking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VadSettings:
    """When speech starts, and - the harder half - when it has stopped."""

    threshold: float = 0.5
    #: Anything shorter is a cough, a door, or a keyboard.
    min_speech_ms: int = 300
    #: How long a pause has to be before it counts as the end of a turn.
    #: Too short and Buddy interrupts you mid-thought; too long and every
    #: exchange has a dead second in it.
    silence_ms: int = 900
    #: A hard stop, so a stuck microphone cannot record forever.
    max_utterance_s: float = 60.0
    #: Kept from before speech was detected, so the first word is not clipped.
    preroll_ms: int = 300


class VoiceActivity:
    """Streaming speech detection over a rolling window.

    `faster_whisper`'s Silero VAD answers "where is the speech in this
    buffer", which is a question about a recording. Turn-taking needs "is
    someone talking *now*", so the buffer is a short trailing window that
    moves, and the transitions between windows are what start and end a turn.
    """

    def __init__(self, settings: VadSettings | None = None, *, sample_rate: int = SAMPLE_RATE):
        self.settings = settings or VadSettings()
        self.sample_rate = sample_rate
        self._captured: list[Any] = []
        self._preroll: list[Any] = []
        self._speaking = False
        self._silence_samples = 0
        self._speech_samples = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def _has_speech(self, frame: Any) -> bool:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        options = VadOptions(
            threshold=self.settings.threshold,
            # The window is already short; letting the VAD apply its own
            # two-second silence rule on top would hide every gap.
            min_silence_duration_ms=0,
            speech_pad_ms=0,
        )
        return bool(get_speech_timestamps(frame, options, sampling_rate=self.sample_rate))

    def feed(self, frame: Any) -> str:
        """One window of audio. Returns "quiet", "speech", or "ended"."""

        samples = len(frame)
        speech = self._has_speech(frame)

        if not self._speaking:
            preroll_samples = int(self.sample_rate * self.settings.preroll_ms / 1000)
            self._preroll.append(frame)
            while sum(len(part) for part in self._preroll) > preroll_samples + samples:
                self._preroll.pop(0)
            if not speech:
                return "quiet"
            # Start the turn with what was already in hand, so the first
            # syllable is not the one that gets lost.
            self._speaking = True
            self._captured = list(self._preroll)
            self._preroll = []
            self._speech_samples = samples
            self._silence_samples = 0
            return "speech"

        self._captured.append(frame)
        if speech:
            self._speech_samples += samples
            self._silence_samples = 0
        else:
            self._silence_samples += samples

        held = sum(len(part) for part in self._captured)
        long_enough = self._speech_samples >= self.sample_rate * self.settings.min_speech_ms / 1000
        silent_enough = self._silence_samples >= self.sample_rate * self.settings.silence_ms / 1000
        too_long = held >= self.sample_rate * self.settings.max_utterance_s

        if too_long or (silent_enough and long_enough):
            return "ended"
        if silent_enough and not long_enough:
            # A noise, not a turn. Forget it and go back to listening.
            self.reset()
            return "quiet"
        return "speech"

    def take(self) -> Any:
        """The utterance that just ended, and a clean slate."""
        import numpy

        captured = self._captured
        self.reset()
        if not captured:
            return numpy.zeros(0, dtype="float32")
        return numpy.concatenate(captured, axis=0).reshape(-1)

    def reset(self) -> None:
        self._captured = []
        self._preroll = []
        self._speaking = False
        self._silence_samples = 0
        self._speech_samples = 0


@runtime_checkable
class Capture(Protocol):
    """How an utterance is captured. Two implementations, one question.

    Hands-free replaces push-to-talk with a detector rather than adding a second
    code path, so `Listener` asks for the next utterance and never learns
    whether a key or a pause produced it.
    """

    def available(self) -> bool: ...
    def next_utterance(self, stop: threading.Event) -> Any: ...


class PushToTalkCapture:
    """Press to start, press again to stop."""

    def __init__(self, microphone: Microphone, keys: PushToTalk, *, on_state=None) -> None:
        self.microphone = microphone
        self.keys = keys
        self.on_state = on_state or (lambda _state: None)

    def available(self) -> bool:
        return self.microphone.available()[0] and self.keys.available()

    def next_utterance(self, stop: threading.Event) -> Any:
        import numpy

        if self.keys.wait_for_key(stop) is None:
            return numpy.zeros(0, dtype="float32")
        self.on_state("listening")
        finished = threading.Event()
        threading.Thread(
            target=self._wait_for_second_press, args=(finished, stop), daemon=True
        ).start()
        return self.microphone.record(finished)

    def _wait_for_second_press(self, finished: threading.Event, stop: threading.Event) -> None:
        self.keys.wait_for_key(stop)
        finished.set()


class HandsFree:
    """Listens continuously and hands over each utterance.

    The `PushToTalkCapture` alternative, with the same shape.
    """

    #: How much audio the VAD is asked about at a time. Short enough that the
    #: end of a turn is noticed promptly, long enough that Silero has
    #: something to work with.
    WINDOW_MS = 200

    def __init__(
        self,
        microphone: Microphone,
        settings: VadSettings | None = None,
        *,
        on_state=None,
    ) -> None:
        self.microphone = microphone
        self.detector = VoiceActivity(settings, sample_rate=microphone.sample_rate)
        self.on_state = on_state or (lambda _state: None)

    def available(self) -> bool:
        return self.microphone.available()[0]

    def next_utterance(self, stop: threading.Event) -> Any:
        """Block until someone says something, then return it."""
        import numpy
        import sounddevice

        window = int(self.microphone.sample_rate * self.WINDOW_MS / 1000)
        pending: list[Any] = []

        def collect(indata, _frames, _time, _status) -> None:
            pending.append(indata.copy().reshape(-1))

        with sounddevice.InputStream(
            samplerate=self.microphone.sample_rate,
            channels=1,
            dtype="float32",
            device=self.microphone.device,
            blocksize=window,
            callback=collect,
        ):
            while not stop.is_set():
                if not pending:
                    stop.wait(0.02)
                    continue
                frame = pending.pop(0)
                state = self.detector.feed(frame)
                if state == "speech" and not self.detector.speaking:
                    self.on_state("listening")
                if state == "ended":
                    return self.detector.take()
        return numpy.zeros(0, dtype="float32")
