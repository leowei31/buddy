"""The voice thread: two queues, one-request lookahead, and barge-in.

The voice layer lives in its own thread with text in and text out at the
brain's boundary, so nothing here knows what a task or an agent is - it turns
speech into a string, hands it over, and turns the answer back into sound.

The shape that matters: split the brain's reply at sentence
boundaries and keep exactly one request ahead. While sentence one is playing,
sentence two is already being synthesised; when sentence one ends, sentence
two's first bytes have usually arrived. That is what makes per-sentence HTTP
conversational without a stateful socket, and it is the guaranteed
path - the websocket is used when the backend has it, and nothing breaks when
it does not.

Barge-in is the other half: new speech stops playback *now*, drops the
audio in flight, clears the queue, and cancels the brain's stream, so Buddy
does not keep writing a reply nobody will hear.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from buddy.voice.tts import FallbackClient, FishClient, TtsError

#: Where a sentence may be broken for speech. Clause boundaries count: audio
#: should start at the first natural pause, not the first full stop.
_BOUNDARY = re.compile(r"(?<=[.!?…])\s+|(?<=[;:])\s+|\n{2,}")

#: Below this a fragment is not worth a request of its own; it is glued to
#: the next one instead.
MIN_SENTENCE_CHARS = 12

#: How far ahead to synthesise. One request; more would buy little
#: and waste tokens on speech that barge-in is about to throw away.
LOOKAHEAD = 1


def split_sentences(text: str, *, minimum: int = MIN_SENTENCE_CHARS) -> list[str]:
    """Split for speech, not for grammar.

    A very short fragment is merged forward rather than spoken alone, because
    a separate request for "OK." costs a whole round trip to say nothing.
    """
    pieces = [piece.strip() for piece in _BOUNDARY.split(text) if piece and piece.strip()]
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < minimum:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


class SentenceBuffer:
    """Accumulates streamed text and releases whole sentences as they finish.

    The brain streams token by token; this is what turns that into units a
    synthesiser can start on without waiting for the reply to end.
    """

    def __init__(self, *, minimum: int = MIN_SENTENCE_CHARS) -> None:
        self.minimum = minimum
        self._pending = ""

    def feed(self, chunk: str) -> list[str]:
        self._pending += chunk
        if not _BOUNDARY.search(self._pending):
            return []
        parts = split_sentences(self._pending, minimum=self.minimum)
        if not parts:
            return []
        # The last part may still be growing, so it stays behind unless the
        # text ended on a boundary.
        if not _BOUNDARY.search(self._pending[-2:]) and not self._pending.endswith(("\n", " ")):
            self._pending = parts.pop()
        else:
            self._pending = ""
        return [part for part in parts if len(part) >= self.minimum or self._pending == ""]

    def flush(self) -> list[str]:
        remaining, self._pending = self._pending.strip(), ""
        return [remaining] if remaining else []


# --------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------


class Playback(Protocol):
    """The speaker, behind a protocol so barge-in is testable without audio."""

    def write(self, audio: bytes) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


class NullPlayback:
    """Buddy with no speaker: everything still runs, nothing is heard."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.stops = 0

    def write(self, audio: bytes) -> None:
        self.written.append(audio)

    def stop(self) -> None:
        self.stops += 1
        self.written.clear()

    def close(self) -> None:
        return None


class SoundDevicePlayback:
    """Raw PCM straight to the output device.

    Buddy asks Fish for `pcm` precisely so this is a write with nothing in
    between - no decode, no temp file, and `stop()` can drop what is queued
    the instant someone starts talking.
    """

    def __init__(self, *, sample_rate: int = 44100, channels: int = 1, device: Any = None) -> None:
        import sounddevice

        self.sample_rate = sample_rate
        self._stream = sounddevice.RawOutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            device=device,
        )
        self._stream.start()

    def write(self, audio: bytes) -> None:
        # int16 frames must not be split mid-sample or the stream desyncs.
        usable = len(audio) - (len(audio) % 2)
        if usable:
            self._stream.write(audio[:usable])

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.abort()  # drop what is queued, do not drain it
        with contextlib.suppress(Exception):
            self._stream.start()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.stop()
            self._stream.close()


def open_playback(*, sample_rate: int = 44100, device: Any = None) -> Playback:
    """A real speaker if one can be opened, otherwise a silent one.

    No speaker is a reason to be quiet, never a reason to stop: the queue,
    the agents and the dashboard do not need audio.
    """
    try:
        return SoundDevicePlayback(sample_rate=sample_rate, device=device)
    except Exception:  # noqa: BLE001 - no device, no PortAudio, no permission
        return NullPlayback()


# --------------------------------------------------------------------------
# Narration
# --------------------------------------------------------------------------


@dataclass
class Spoken:
    """What a narration actually did, for tests and for `doctor`."""

    sentences: list[str] = field(default_factory=list)
    bytes_played: int = 0
    first_audio_seconds: float | None = None
    interrupted: bool = False
    errors: list[str] = field(default_factory=list)


class Narrator:
    """Text in, sound out, one request ahead.

    Two coroutines: one synthesising, one playing. They are separate so the
    speaker is never idle while a request is in flight, and so barge-in can
    stop both at once without either waiting on the other.
    """

    def __init__(
        self,
        client: FishClient | FallbackClient,
        playback: Playback,
        *,
        lookahead: int = LOOKAHEAD,
        clock=None,
    ) -> None:
        self.client = client
        self.playback = playback
        self.lookahead = max(1, lookahead)
        self._clock = clock or asyncio.get_event_loop().time
        self._current: asyncio.Task[Any] | None = None
        self._synthesis: list[asyncio.Task[Any]] = []
        self._interrupt = asyncio.Event()

    async def say(self, sentences: AsyncIterator[str] | Iterable[str]) -> Spoken:
        """Speak everything, or as much as barge-in allows."""
        self._interrupt.clear()
        result = Spoken()
        started = self._clock()

        source = sentences if isinstance(sentences, AsyncIterator) else _aiter(sentences)
        # Unbounded on purpose: the semaphore below is the only bound, and a
        # second one only creates ways for the two to deadlock each other.
        pipeline: asyncio.Queue[tuple[str, asyncio.Queue] | None] = asyncio.Queue()
        # The lookahead policy, and the only thing enforcing it: a permit is
        # taken before a sentence is synthesised and returned only once that
        # sentence has finished *playing*. With one permit spare, exactly one
        # request is in flight ahead of the speaker - bounding the
        # queue alone would not do it, because a fast server empties the
        # queue faster than the speaker drains it and every request goes out
        # at once.
        permits = asyncio.Semaphore(self.lookahead + 1)

        self._synthesis = []
        producer = asyncio.create_task(self._produce(source, pipeline, result, permits))
        consumer = asyncio.create_task(self._consume(pipeline, result, started, permits))
        self._current = consumer

        interrupted = asyncio.create_task(self._interrupt.wait())
        done, pending = await asyncio.wait(
            {consumer, interrupted}, return_when=asyncio.FIRST_COMPLETED
        )
        if interrupted in done:
            result.interrupted = True
        for task in (producer, consumer, interrupted, *self._synthesis):
            if task not in done:
                task.cancel()
        for task in (producer, consumer, interrupted, *self._synthesis):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._synthesis = []
        self._current = None
        return result

    def interrupt(self) -> None:
        """Barge-in: stop the sound now, drop what is queued.

        Synchronous on purpose - it is called from the input thread the
        instant speech is detected, and waiting for the loop to come round
        would be audible.
        """
        self.playback.stop()
        self._interrupt.set()

    async def _produce(
        self,
        sentences: AsyncIterator[str],
        pipeline: asyncio.Queue,
        result: Spoken,
        permits: asyncio.Semaphore,
    ) -> None:
        try:
            async for sentence in sentences:
                if not sentence.strip():
                    continue
                # Waits until the speaker has finished a sentence, which is
                # what keeps exactly one request ahead of playback.
                await permits.acquire()
                chunks: asyncio.Queue = asyncio.Queue()
                self._synthesis.append(
                    asyncio.create_task(self._synthesize(sentence, chunks, result))
                )
                await pipeline.put((sentence, chunks))
        finally:
            # `put_nowait`, and never a cancel: the sentinel has to land even
            # when this coroutine is being cancelled, and the requests still
            # in flight belong to sentences the consumer has yet to play.
            # Cancelling them here would silently drop audio; `say` cancels
            # them on barge-in, which is the only time they should die early.
            pipeline.put_nowait(None)

    async def _synthesize(self, sentence: str, chunks: asyncio.Queue, result: Spoken) -> None:
        try:
            async for audio in self.client.stream(sentence):
                await chunks.put(audio)
        except (TtsError, asyncio.CancelledError) as exc:
            if isinstance(exc, TtsError):
                result.errors.append(f"{sentence[:40]}: {exc}")
            raise
        finally:
            # Never `await` here: under cancellation an await raises at once
            # and the sentinel would never arrive, leaving the consumer
            # waiting on a queue nobody will ever finish.
            chunks.put_nowait(None)

    async def _consume(
        self,
        pipeline: asyncio.Queue,
        result: Spoken,
        started: float,
        permits: asyncio.Semaphore,
    ) -> None:
        while True:
            item = await pipeline.get()
            if item is None:
                return
            sentence, chunks = item
            result.sentences.append(sentence)
            try:
                while True:
                    audio = await chunks.get()
                    if audio is None:
                        break
                    if result.first_audio_seconds is None:
                        result.first_audio_seconds = self._clock() - started
                    result.bytes_played += len(audio)
                    # Blocking write, in a thread: the output stream applies
                    # backpressure, which is what keeps playback in real time.
                    await asyncio.to_thread(self.playback.write, audio)
            finally:
                permits.release()


async def _aiter(values: Iterable[str]) -> AsyncIterator[str]:
    for value in values:
        yield value


# --------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------


@dataclass
class VoiceTurn:
    """One thing the user said, on its way to the brain."""

    text: str
    barged_in: bool = False


class VoiceSession:
    """The two queues, and the barge-in rule that joins them.

    Buddy's loop owns the brain; this owns the microphone and the speaker.
    They meet at two queues of plain strings, which is the decoupling voice
    opens with - and the reason the whole orchestrator was testable by typing
    long before any of this existed.
    """

    def __init__(self, narrator: Narrator, *, on_barge_in=None) -> None:
        self.narrator = narrator
        self.heard: asyncio.Queue[VoiceTurn] = asyncio.Queue()
        #: Called when speech arrives while Buddy is mid-reply. `Speech` uses
        #: it to cancel the brain's stream, which is the half of barge-in the
        #: narrator cannot do for itself.
        self.on_barge_in = on_barge_in or (lambda: None)
        self._speaking = False

    @property
    def speaking(self) -> bool:
        return self._speaking

    async def speak(self, chunks: AsyncIterator[str] | Iterable[str]) -> Spoken:
        """Say a reply as it is written."""
        self._speaking = True
        try:
            return await self.narrator.say(_sentences_from(chunks))
        finally:
            self._speaking = False

    def heard_speech(self, text: str) -> None:
        """Called by the input thread when the user says something.

        If Buddy is mid-sentence this is a barge-in: playback stops, the
        synthesis in flight is dropped, and the caller cancels the brain's
        stream.
        """
        barged = self._speaking
        if barged:
            self.narrator.interrupt()
            self.on_barge_in()
        self.heard.put_nowait(VoiceTurn(text=text, barged_in=barged))


async def _sentences_from(chunks: AsyncIterator[str] | Iterable[str]) -> AsyncIterator[str]:
    """Streamed text in, whole sentences out."""
    buffer = SentenceBuffer()
    source = chunks if isinstance(chunks, AsyncIterator) else _aiter(chunks)
    async for chunk in source:
        for sentence in buffer.feed(chunk):
            yield sentence
    for sentence in buffer.flush():
        yield sentence


# --------------------------------------------------------------------------
# The listening thread
# --------------------------------------------------------------------------


class Listener:
    """Push-to-talk on its own thread, handing text to the session.

    A thread, not a task, because recording and reading the keyboard both
    block for as long as a person takes, which nothing may do on the loop.
    """

    def __init__(
        self,
        session: VoiceSession,
        *,
        capture,
        model: str,
        gpu_kind: str = "none",
        loop: asyncio.AbstractEventLoop | None = None,
        on_state=None,
    ) -> None:
        self.session = session
        self.capture = capture
        self.model = model
        self.gpu_kind = gpu_kind
        self.loop = loop
        self.on_state = on_state or (lambda _state: None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    #: How long to wait after a capture error before listening again.
    RETRY_SECONDS = 1.0

    def start(self) -> None:
        self.loop = self.loop or asyncio.get_running_loop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="buddy-voice", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        from buddy.voice import stt

        loop = self.loop
        if loop is None:
            raise RuntimeError("Listener._run needs the event loop; call start()")

        while not self._stop.is_set():
            self.on_state("idle")
            try:
                samples = self.capture.next_utterance(self._stop)
            except Exception as exc:  # noqa: BLE001 - one bad capture is not the end of listening
                # An overflowed input buffer, a device unplugged and plugged
                # back in: before this, any of them ended the thread, and
                # voice input stopped for the rest of the session in silence.
                self.on_state(f"microphone error ({exc}); still listening")
                self._stop.wait(self.RETRY_SECONDS)
                continue
            if self._stop.is_set() or samples is None or not len(samples):
                continue

            self.on_state("thinking")
            try:
                transcript = asyncio.run_coroutine_threadsafe(
                    stt.transcribe_samples(samples, model=self.model, gpu_kind=self.gpu_kind),
                    loop,
                ).result(timeout=120)
            except Exception as exc:  # noqa: BLE001 - a bad take is not a crash
                self.on_state(f"could not transcribe: {exc}")
                continue
            if not transcript.text.strip():
                continue
            # Straight onto the loop: `heard_speech` is what decides whether
            # this is a new turn or a barge-in.
            loop.call_soon_threadsafe(self.session.heard_speech, transcript.text)


# --------------------------------------------------------------------------
# What the CLI drives
# --------------------------------------------------------------------------


class Speech:
    """The whole voice layer behind four methods.

    `cli.py` starts it, asks for the next thing the user said, hands over a
    reply to speak, and stops it. Everything else - which Fish backend, which
    whisper model, which device, what counts as a sentence, what barge-in
    cancels - stays on this side of the line: text in, text out at the
    brain's boundary.
    """

    def __init__(self, config, resolver, *, playback: Playback | None = None) -> None:
        self.config = config
        self.resolver = resolver
        self._playback = playback
        self.client: FallbackClient | None = None
        self.session: VoiceSession | None = None
        self.listener: Listener | None = None
        self.health: Any = None
        #: The brain's in-flight turn, so barge-in can cancel it.
        self._answering: asyncio.Task[str] | None = None
        #: Something the person should know about the voice itself. The CLI
        #: prints it; by default it goes nowhere, like `_announce`.
        self.on_problem: Callable[[str], None] = lambda _message: None
        self._reported: set[str] = set()

    async def start(self) -> tuple[bool, str]:
        """Ready or not, and why. Never raises: a machine with no microphone
        is a machine you type to, not a failure."""
        from buddy.voice import stt
        from buddy.voice.tts import choose_client

        microphone = stt.Microphone(device=self.config.voice.input_device or None)
        usable, detail = microphone.available()
        if not usable:
            return False, f"no microphone ({detail})"

        hands_free = self.config.voice.hands_free
        capture: stt.HandsFree | stt.PushToTalkCapture
        if hands_free:
            capture = stt.HandsFree(
                microphone,
                stt.VadSettings(silence_ms=self.config.voice.silence_ms),
                on_state=self._announce,
            )
        else:
            capture = stt.PushToTalkCapture(
                microphone,
                stt.PushToTalk(self.config.voice.push_to_talk),
                on_state=self._announce,
            )
        if not capture.available():
            return False, (
                "hands-free needs a microphone this process can open"
                if hands_free
                else "push-to-talk needs a terminal, and this is not one"
            )

        model = self.config.voice.stt_model.split(":", 1)[-1]
        if not stt.is_cached(model, "none"):
            return False, f"the {model} whisper model is not cached; run `buddy setup --force stt`"

        client, speaking = await choose_client(self.config.voice, self.resolver)
        self.client = client
        tts_config = client.config
        self.health = await client.health()
        if not self.health.reachable and tts_config.is_cloud and not tts_config.api_key:
            return False, "no FISH_API_KEY, so there is nothing to speak with"
        if client.switched_because:
            self.on_problem(speaking)

        playback = self._playback or open_playback(sample_rate=tts_config.sample_rate)
        self.session = VoiceSession(Narrator(client, playback), on_barge_in=self._cancel_thinking)
        self.listener = Listener(
            self.session, capture=capture, model=model, on_state=self._announce
        )
        self.listener.start()
        how = (
            "Just talk - it listens for a pause to know you are done."
            if hands_free
            else f"Press {self.config.voice.push_to_talk!r} to talk, and again when you are done."
        )
        return True, f"Listening on {detail}, speaking through {tts_config.backend}. {how}"

    def _announce(self, state: str) -> None:
        return None  # the CLI overrides this when it wants to show state

    async def next_turn(self) -> str | None:
        """The next thing the user said, spoken."""
        if self.session is None:
            return None
        turn = await self.session.heard.get()
        return turn.text

    async def answer(self, brain, utterance: str, *, on_text=None) -> str:
        """Send a turn to the brain and speak the reply as it is written.

        The brain's stream is a task rather than an await so that barge-in can
        cancel it: Buddy must not keep generating a reply nobody is going to
        hear.
        """
        if self.session is None:
            return await brain.send(utterance, on_text=on_text)

        pieces: asyncio.Queue[str | None] = asyncio.Queue()

        def collect(chunk: str) -> None:
            if on_text:
                on_text(chunk)
            pieces.put_nowait(chunk)

        async def stream() -> AsyncIterator[str]:
            while True:
                chunk = await pieces.get()
                if chunk is None:
                    return
                yield chunk

        async def think() -> str:
            try:
                return await brain.send(utterance, on_text=collect)
            finally:
                pieces.put_nowait(None)

        self._answering = asyncio.create_task(think())
        speaking = asyncio.create_task(self.session.speak(stream()))
        try:
            reply = await self._answering
        except asyncio.CancelledError:
            speaking.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await speaking
            return ""
        finally:
            self._answering = None
        with contextlib.suppress(Exception):
            self._report(await speaking)
        return reply

    def _report(self, spoken: Spoken | None) -> None:
        """Say why Buddy went quiet, once per kind of failure.

        Synthesis errors were recorded on the result and read by nothing, so a
        voice session could stop speaking with no explanation. Repeating the
        same failure for every sentence would bury the conversation, so each
        distinct cause is said once.
        """
        if spoken is None or not spoken.errors:
            return
        cause = spoken.errors[-1].split(": ", 1)[-1]
        kind = cause.split(":", 2)[:2]
        key = ":".join(kind)
        if key in self._reported:
            return
        self._reported.add(key)
        self.on_problem(f"Buddy could not speak ({cause}); replies are still shown here.")

    def interrupt(self) -> None:
        """Someone started talking: stop speaking and stop thinking."""
        if self.session is not None:
            self.session.narrator.interrupt()
        self._cancel_thinking()

    def _cancel_thinking(self) -> None:
        if self._answering is not None and not self._answering.done():
            self._answering.cancel()

    async def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        if self.client is not None:
            await self.client.aclose()
            self.client = None
