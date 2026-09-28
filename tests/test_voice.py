"""The voice layer, against a server that speaks the real protocol.

`tests/fake_fish.py` is a Fish Audio server on a real socket framing real
MessagePack. Testing against it rather than a mock of Buddy's own client is
the point: what voice is exposed to is a wire format, so the thing worth
asserting is that Buddy's bytes are the bytes Fish documents - the msgpack
content type, the bearer header, the event names, and audio arriving before
the text has finished.

Nothing here opens a microphone or a speaker. `NullPlayback` and a recording
fake stand in for the devices, which is exactly what makes barge-in - the
hardest thing to get right - testable at all.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import time
from typing import Any

import pytest

from buddy.config import Config, EnvSecretResolver, VoiceSection
from buddy.voice.session import (
    Narrator,
    NullPlayback,
    SentenceBuffer,
    Spoken,
    VoiceSession,
    open_playback,
    split_sentences,
)
from buddy.voice.tts import FishClient, Health, TtsConfig, TtsError, config_for
from tests.fake_fish import FakeFish, audio_for

#: The speech engine is an optional extra, so a machine without it skips
#: these rather than erroring - the same stance `test_sandbox.py` takes about
#: Docker. An import error here reads like a broken build; a skip reads like
#: the truth, which is that this machine cannot run them.
needs_speech = pytest.mark.skipif(
    importlib.util.find_spec("faster_whisper") is None,
    reason="the voice extra is not installed (uv sync --extra voice)",
)


@pytest.fixture
async def fish():
    server = await FakeFish().start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.fixture
async def client(fish):
    opened = FishClient(TtsConfig(base_url=fish.base_url, streaming="auto"))
    try:
        yield opened
    finally:
        await opened.aclose()


async def _until(condition, seconds: float = 5.0) -> None:
    """Wait for something to be true, rather than for a guessed amount of time."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        await asyncio.sleep(0.005)


class Recording(NullPlayback):
    """A speaker that remembers, and can be told to take its time."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self.delay = delay
        self.all_writes: list[bytes] = []
        #: (when the write finished, its first byte), in order - a timeline
        #: the tests compare other events against, instead of a stopwatch.
        self.finished: list[tuple[float, bytes]] = []

    def write(self, audio: bytes) -> None:
        if self.delay:
            time.sleep(self.delay)
        self.all_writes.append(audio)
        self.finished.append((time.monotonic(), audio[:1]))
        super().write(audio)


# -- splitting for speech -------------------------------------------


def test_text_is_split_at_places_a_voice_would_pause():
    assert split_sentences("Scout is on it. Fixer is stuck.") == [
        "Scout is on it.",
        "Fixer is stuck.",
    ]
    # Clause boundaries count: audio should start at the first natural pause,
    # not the first full stop.
    assert split_sentences("It is queued; nothing is blocking it.") == [
        "It is queued;",
        "nothing is blocking it.",
    ]


def test_a_tiny_fragment_is_not_worth_its_own_request():
    """A round trip to say "OK." costs more than it saves."""
    assert split_sentences("OK. Scout has finished the migration.") == [
        "OK. Scout has finished the migration."
    ]


def test_streamed_text_becomes_sentences_before_the_reply_ends():
    buffer = SentenceBuffer()
    released: list[str] = []
    for chunk in ["Scout is ", "on it. Fix", "er needs input. ", "Nothing else."]:
        released.extend(buffer.feed(chunk))
    # The first sentence is available long before the last chunk arrives,
    # which is the whole reason for streaming at all.
    assert released[0] == "Scout is on it."
    released.extend(buffer.flush())
    assert released == ["Scout is on it.", "Fixer needs input.", "Nothing else."]


def test_text_with_no_boundary_is_held_until_flush():
    buffer = SentenceBuffer()
    assert buffer.feed("still going") == []
    assert buffer.flush() == ["still going"]


# -- the model cache --------------------------------------------------


@needs_speech
def test_a_model_is_loaded_once_not_once_per_utterance(monkeypatch):
    """Found by measuring the turn-around: 8.4s, then 10.6s, then 13.7s for
    the same two-second clip, because every call built a new WhisperModel and
    the abandoned ones piled up."""
    from buddy.voice import stt

    loads: list[tuple[str, str]] = []

    class Model:
        def transcribe(self, _audio, **_kwargs):
            return [], type("Info", (), {"language": "en", "duration": 1.0})()

    def counting(name, device=None, compute_type=None, local_files_only=None):
        loads.append((name, device))
        return Model()

    stt.unload_models()
    monkeypatch.setattr("faster_whisper.WhisperModel", counting)
    try:
        for _ in range(4):
            stt._load("small", "none", download=False)
        assert loads == [("small", "cpu")], f"loaded {len(loads)} times"
        # A different model, or different hardware, is a different entry.
        stt._load("base.en", "none", download=False)
        assert len(loads) == 2
    finally:
        stt.unload_models()


# -- the wire format ------------------------------------------------


async def test_the_request_is_msgpack_with_the_documented_fields(client, fish):
    await client.synthesize("Say this out loud.")

    assert fish.recorder.content_types == ["application/msgpack"]
    (request,) = fish.recorder.requests
    assert request["text"] == "Say this out loud."
    assert request["format"] == "pcm"
    assert request["latency"] == "balanced"
    assert request["normalize"] is True
    assert request["chunk_length"] == 200
    # An obvious guess is `streaming: true`; the documented API has no such
    # field, and the response is chunked either way.
    assert "streaming" not in request


async def test_audio_arrives_in_chunks_rather_than_all_at_once(client):
    chunks = [chunk async for chunk in client.stream("Streaming works here.")]
    assert len(chunks) > 1, "a single chunk means nothing was streamed"
    assert b"".join(chunks) == await client.synthesize("Streaming works here.")


async def test_the_model_reported_is_the_one_that_answered():
    """The model line moves fast, so the server names it, not a document."""
    server = await FakeFish(model="s3-preview").start()
    try:
        opened = FishClient(TtsConfig(base_url=server.base_url))
        health = await opened.health()
        assert health.reachable and health.model == "s3-preview"
        await opened.aclose()
    finally:
        await server.stop()


async def test_the_cloud_sends_a_bearer_token_and_the_local_server_does_not():
    server = await FakeFish(require_key=True).start()
    try:
        without = FishClient(TtsConfig(backend="fish_local", base_url=server.base_url))
        with pytest.raises(TtsError) as refused:
            await without.synthesize("nope")
        assert "401" in str(refused.value)
        await without.aclose()

        with_key = FishClient(
            TtsConfig(backend="fish_cloud", base_url=server.base_url, api_key="k", model="s2-pro")
        )
        assert await with_key.synthesize("yes") == b"yyy"
        assert server.recorder.headers[-1]["authorization"] == "Bearer k"
        # The model is a header, not a body field.
        assert server.recorder.headers[-1]["model"] == "s2-pro"
        assert "model" not in server.recorder.requests[-1]
        await with_key.aclose()
    finally:
        await server.stop()


async def test_an_unreachable_backend_is_a_sentence_not_a_traceback():
    dead = FishClient(TtsConfig(base_url="http://127.0.0.1:1", connect_timeout=0.2))
    health = await dead.health()
    assert not health.reachable
    assert "unreachable" in health.summary()
    with pytest.raises(TtsError):
        await dead.synthesize("anything")
    await dead.aclose()


# -- incremental text over the websocket ----------------------------


async def test_the_websocket_is_probed_and_used_when_it_is_there(client, fish):
    assert await client.supports_websocket() is True

    audio = [chunk async for chunk in client.stream_incremental(["First clause. ", "second."])]
    events = [event.get("event") for event in fish.recorder.live_events]

    assert events == ["start", "text", "flush", "text", "flush", "stop"]
    # Two flushes, two pieces of audio: the model spoke the first clause
    # while the second was still being written, which is the whole point.
    assert len(audio) == 2
    assert fish.recorder.requests[0]["text"] == "", "the start event carries no text"


async def test_a_backend_without_a_websocket_is_detected_not_assumed():
    """Probed, not assumed: the open-source server may not have it."""
    server = await FakeFish(websocket=False).start()
    try:
        opened = FishClient(TtsConfig(base_url=server.base_url, streaming="auto"))
        assert await opened.supports_websocket() is False
        # And the guaranteed path still works.
        assert await opened.synthesize("Per-sentence HTTP still works.")
        await opened.aclose()
    finally:
        await server.stop()


async def test_configured_streaming_skips_the_probe(fish):
    forced = FishClient(TtsConfig(base_url=fish.base_url, streaming="http"))
    assert await forced.supports_websocket() is False
    assert fish.recorder.live_connections == 0, "config said http; nothing should have connected"
    await forced.aclose()


# -- narration ------------------------------------------------------


async def test_sentences_are_spoken_in_order(client):
    playback = Recording()
    spoken = await Narrator(client, playback).say(["First one.", "Second one.", "Third one."])

    assert spoken.sentences == ["First one.", "Second one.", "Third one."]
    assert spoken.bytes_played > 0
    assert not spoken.errors
    assert b"".join(playback.all_writes).startswith(b"F")


async def test_exactly_one_request_runs_ahead_of_the_speaker(client, fish):
    """One request of lookahead, measured rather than asserted.

    Bounding the queue alone does not do it: a fast server empties the queue
    faster than the speaker drains it and every request goes out at once.
    """
    playback = Recording(delay=0.06)
    await Narrator(client, playback).say(["Alpha one.", "Bravo two.", "Charlie three."])

    requests = fish.recorder.request_times
    assert len(requests) == 3
    # Compared against the speaker's own timeline, not a stopwatch: ordering
    # holds however slow the machine is; a threshold in seconds does not.
    first_done = max(when for when, first in playback.finished if first == b"A")
    assert requests[1] < first_done, "the second request should be in flight while one plays"
    assert requests[2] > first_done, "the third should wait for the first to finish playing"


async def test_barge_in_stops_the_sound_and_drops_what_was_queued(client):
    """Barge-in: stop playback immediately, cancel the work in flight."""
    playback = Recording(delay=0.05)
    narrator = Narrator(client, playback)

    speaking = asyncio.create_task(narrator.say([f"Sentence number {n}." for n in range(1, 8)]))
    await _until(lambda: playback.all_writes)  # interrupt mid-sound, however long that takes
    narrator.interrupt()
    spoken = await asyncio.wait_for(speaking, timeout=5)

    assert spoken.interrupted
    assert playback.stops == 1
    assert len(spoken.sentences) < 7, "it kept talking through the interruption"


async def test_a_failing_sentence_is_recorded_and_the_rest_still_play(client, monkeypatch):
    playback = Recording()
    real = client.stream
    calls: list[str] = []

    def flaky(text: str):
        calls.append(text)
        if len(calls) == 2:
            raise TtsError("the server hiccupped")
        return real(text)

    monkeypatch.setattr(client, "stream", flaky)
    spoken = await Narrator(client, playback).say(["One here.", "Two here.", "Three here."])

    assert spoken.errors and "hiccupped" in spoken.errors[0]
    assert spoken.sentences == ["One here.", "Two here.", "Three here."]
    assert spoken.bytes_played > 0, "the sentences that worked were still spoken"


# -- the session and barge-in ---------------------------------------


async def test_speech_while_buddy_is_talking_is_a_barge_in(client):
    cancelled: list[str] = []
    playback = Recording(delay=0.05)
    session = VoiceSession(
        Narrator(client, playback),
        on_barge_in=lambda: cancelled.append("brain"),
    )

    speaking = asyncio.create_task(
        session.speak(["Something long. ", "And more of it. ", "And still more."])
    )
    await _until(lambda: playback.all_writes)
    session.heard_speech("actually, stop")
    await asyncio.wait_for(speaking, timeout=5)

    turn = session.heard.get_nowait()
    assert turn.text == "actually, stop"
    assert turn.barged_in is True
    # The brain's stream is cancelled too, so it does not keep writing
    # a reply nobody will hear.
    assert cancelled == ["brain"]


async def test_speech_while_buddy_is_silent_is_just_a_turn(client):
    cancelled: list[str] = []
    session = VoiceSession(Narrator(client, Recording()), on_barge_in=lambda: cancelled.append("x"))
    session.heard_speech("what is scout doing?")

    turn = session.heard.get_nowait()
    assert turn.barged_in is False
    assert cancelled == []


# -- configuration ---------------------------------------------------


def test_the_backend_is_resolved_from_the_voice_block(monkeypatch):
    monkeypatch.setenv("FISH_API_KEY", "from-the-environment")
    voice = VoiceSection(
        tts_backend="fish_cloud",
        fish_cloud={"base_url": "https://api.fish.audio", "model": "s2-pro", "reference_id": "abc"},
    )
    config = config_for(voice, EnvSecretResolver())

    assert config.is_cloud
    assert config.api_key == "from-the-environment"
    assert config.model == "s2-pro"
    assert config.reference_id == "abc"
    assert config.websocket_url == "wss://api.fish.audio/v1/tts/live"
    # The secret is resolved, never stored in the config file.
    assert "from-the-environment" not in str(voice.fish_cloud)


def test_the_local_backend_needs_no_key():
    config = config_for(VoiceSection(tts_backend="fish_local"), EnvSecretResolver())
    assert config.api_key is None
    assert config.headers() == {}
    assert config.websocket_url == "ws://127.0.0.1:8080/v1/tts/live"


def test_the_defaults_come_from_a_real_config(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[voice]\ntts_backend = "fish_local"\n'
        '[voice.fish_local]\nbase_url = "http://127.0.0.1:9999"\nformat = "wav"\n'
    )
    config = Config.load(home=tmp_path)
    resolved = config_for(config.voice, EnvSecretResolver())
    assert resolved.base_url == "http://127.0.0.1:9999"
    assert resolved.audio_format == "wav"
    assert resolved.request("hi")["format"] == "wav"
    # pcm is the only format that carries no header, so it is the only one
    # that needs the rate agreed up front.
    assert "sample_rate" not in resolved.request("hi")


# -- what doctor measures -------------------------------------------


async def test_measure_reports_a_real_first_byte_time(fish):
    slow = await FakeFish(first_byte_delay=0.05).start()
    try:
        opened = FishClient(TtsConfig(base_url=slow.base_url))
        health = await opened.measure("Buddy is ready.")
        assert health.reachable
        assert health.ttfb_seconds is not None and health.ttfb_seconds >= 0.05
        assert "first byte in" in health.summary()
        assert health.websocket is True
        await opened.aclose()
    finally:
        await slow.stop()


def test_health_summarises_without_a_measurement():
    unreachable = Health(reachable=False, backend="fish_local", base_url="u", detail="no server")
    assert unreachable.summary() == "fish_local at u: no server"


def test_a_machine_with_no_speaker_gets_a_silent_one(monkeypatch):
    """No speaker is a reason to be quiet, never a reason to stop."""
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", None)
    playback = open_playback()
    playback.write(b"\x00\x01")
    playback.stop()
    playback.close()
    assert isinstance(playback, NullPlayback)


# -- hands-free turn-taking ----------------------------------------


def _clip() -> Any:
    """The bundled test clip as float32 samples, at whisper's rate."""
    import wave

    import numpy

    path = pathlib.Path(__file__).parent.parent / "buddy/setup/assets/test-clip.wav"
    with wave.open(str(path)) as opened:
        raw = opened.readframes(opened.getnframes())
    return numpy.frombuffer(raw, dtype=numpy.int16).astype(numpy.float32) / 32768.0


def _windows(samples, ms: int = 200):
    from buddy.voice.stt import SAMPLE_RATE

    step = int(SAMPLE_RATE * ms / 1000)
    for start in range(0, len(samples), step):
        window = samples[start : start + step]
        if len(window) == step:
            yield window


@needs_speech
def test_a_turn_is_found_in_silence_speech_silence():
    """Real audio, not a synthetic tone: the detector has to work on a voice."""
    import numpy

    from buddy.voice.stt import SAMPLE_RATE, VadSettings, VoiceActivity

    speech = _clip()
    quiet = numpy.zeros(int(SAMPLE_RATE * 1.5), dtype="float32")
    detector = VoiceActivity(VadSettings(silence_ms=600))

    turns = []
    for window in _windows(numpy.concatenate([quiet, speech, quiet])):
        if detector.feed(window) == "ended":
            turns.append(detector.take())

    assert len(turns) == 1, "one person said one thing"
    # Longer than the speech, because the pre-roll keeps the first syllable
    # and the trailing pause is what ended the turn.
    assert len(turns[0]) / SAMPLE_RATE > len(speech) / SAMPLE_RATE


@needs_speech
def test_silence_alone_never_starts_a_turn():
    """A microphone left on must not hand the brain an empty utterance."""
    import numpy

    from buddy.voice.stt import SAMPLE_RATE, VadSettings, VoiceActivity

    detector = VoiceActivity(VadSettings(silence_ms=600))
    states = [
        detector.feed(window)
        for window in _windows(numpy.zeros(int(SAMPLE_RATE * 4), dtype="float32"))
    ]
    assert set(states) == {"quiet"}
    assert not detector.speaking


@needs_speech
def test_a_captured_turn_still_transcribes():
    """The end-to-end point of the detector: what it hands over is usable."""
    import asyncio as aio

    import numpy

    from buddy.voice import stt

    speech = _clip()
    quiet = numpy.zeros(int(stt.SAMPLE_RATE * 1.0), dtype="float32")
    detector = stt.VoiceActivity(stt.VadSettings(silence_ms=600))

    captured = None
    for window in _windows(numpy.concatenate([quiet, speech, quiet])):
        if detector.feed(window) == "ended":
            captured = detector.take()
            break
    assert captured is not None

    model = stt.model_for("none")
    if not stt.is_cached(model):
        pytest.skip(f"the {model} whisper model is not cached here")
    said = aio.run(stt.transcribe_samples(captured, model=model))
    assert stt.looks_like(said.text, "Buddy, this is a test of the microphone.")


@needs_speech
def test_the_two_capture_strategies_are_interchangeable():
    """Hands-free replaces push-to-talk rather than adding a second code path."""
    from buddy.voice.stt import Capture, HandsFree, Microphone, PushToTalk, PushToTalkCapture

    keyed = PushToTalkCapture(Microphone(), PushToTalk(" "))
    free = HandsFree(Microphone())
    for capture in (keyed, free):
        assert isinstance(capture, Capture)
        assert callable(capture.next_utterance)


def test_hands_free_is_off_unless_asked_for(tmp_path):
    """Turning a microphone on permanently is the user's call."""
    from buddy.config import Config, VoiceSection

    assert VoiceSection().hands_free is False
    (tmp_path / "config.toml").write_text("[voice]\nhands_free = true\nsilence_ms = 700\n")
    voice = Config.load(home=tmp_path).voice
    assert voice.hands_free is True
    assert voice.silence_ms == 700


# -- falling back, and failing out loud ------------------------------------


def _local_that_is_down() -> FishClient:
    # Port 9 is discard: nothing listens, so the connection is refused at once.
    return FishClient(TtsConfig(backend="fish_local", base_url="http://127.0.0.1:9"))


async def test_an_unreachable_local_server_falls_back_to_cloud_mid_session(fish):
    """`tts_fallback` was parsed, written by setup, and read by nothing: the
    promised fall back to cloud did not exist."""
    from buddy.voice.tts import FallbackClient

    cloud = FishClient(TtsConfig(backend="fish_cloud", base_url=fish.base_url, api_key="k"))
    speaker = FallbackClient(_local_that_is_down(), cloud)
    try:
        audio = b"".join([chunk async for chunk in speaker.stream("Hello there.")])
        assert audio == audio_for("Hello there.")
        assert speaker.config.backend == "fish_cloud"
        assert "fish_local" in speaker.switched_because
        # And it stays switched: the next sentence does not retry the dead one.
        again = b"".join([chunk async for chunk in speaker.stream("Still here.")])
        assert again == audio_for("Still here.")
    finally:
        await speaker.aclose()


async def test_no_fallback_across_sample_rates(fish):
    """Playback was opened at the primary's rate; audio at another one plays
    at the wrong speed, which is worse than silence."""
    from buddy.voice.tts import FallbackClient

    cloud = FishClient(
        TtsConfig(backend="fish_cloud", base_url=fish.base_url, api_key="k", sample_rate=24000)
    )
    speaker = FallbackClient(_local_that_is_down(), cloud)
    try:
        with pytest.raises(TtsError, match="fish_local"):
            [chunk async for chunk in speaker.stream("Hello there.")]
        assert speaker.switched_because == ""
    finally:
        await speaker.aclose()


async def test_startup_speaks_through_the_fallback_when_the_primary_is_down(fish, monkeypatch):
    from buddy.voice.tts import choose_client

    voice = VoiceSection.from_dict(
        {
            "tts_backend": "fish_local",
            "tts_fallback": "fish_cloud",
            "fish_local": {"base_url": "http://127.0.0.1:9"},
            "fish_cloud": {"base_url": fish.base_url},
        }
    )
    monkeypatch.setenv("FISH_API_KEY", "k")
    speaker, why = await choose_client(voice, EnvSecretResolver())
    try:
        assert speaker.config.backend == "fish_cloud"
        assert "fish_local" in why and "fish_cloud" in why
    finally:
        await speaker.aclose()


async def test_a_voice_that_cannot_speak_says_so(tmp_path):
    """Synthesis errors were recorded on the result and read
    by nothing, so a voice session went quiet with no explanation."""
    from buddy.voice.session import Speech

    problems: list[str] = []
    speech = Speech(Config(home=tmp_path), EnvSecretResolver())
    speech.on_problem = problems.append
    speech._report(Spoken(errors=["Hello there.: fish_local: ConnectError: refused"]))
    speech._report(Spoken(errors=["Again.: fish_local: ConnectError: refused"]))
    assert len(problems) == 1, "said once, not once per sentence"
    assert "could not speak" in problems[0] and "ConnectError" in problems[0]


async def test_a_microphone_error_does_not_end_listening():
    """An exception from capture escaped the listener thread
    and voice input stopped for good, silently."""
    import threading

    from buddy.voice.session import Listener

    loop = asyncio.get_running_loop()
    states: list[str] = []
    calls = {"n": 0}
    done = threading.Event()

    class FlakyMicrophone:
        def next_utterance(self, stop):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("Input overflowed")
            done.set()
            stop.wait(5)
            return None

    listener = Listener(
        VoiceSession(Narrator(None, None)),
        capture=FlakyMicrophone(),
        model="tiny.en",
        loop=loop,
        on_state=states.append,
    )
    listener.RETRY_SECONDS = 0.01
    listener._thread = threading.Thread(target=listener._run, daemon=True)
    listener._thread.start()
    try:
        assert await asyncio.to_thread(done.wait, 5), "the thread died on the first error"
        assert any("microphone" in state and "Input overflowed" in state for state in states)
    finally:
        await asyncio.to_thread(listener.stop)


async def test_a_live_stream_that_goes_quiet_is_an_error_not_a_hang():
    """The websocket frame loop had no timeout at all."""
    stalled = await FakeFish(live_stall=5).start()
    client = FishClient(TtsConfig(base_url=stalled.base_url, request_timeout=0.3))
    try:
        started = time.monotonic()
        with pytest.raises(TtsError, match="nothing for 0.3s"):
            [chunk async for chunk in client.stream_incremental(["Hello there."])]
        assert time.monotonic() - started < 3
    finally:
        await client.aclose()
        await stalled.stop()


async def test_the_cloud_is_unknown_until_measured_not_unreachable():
    health = await FishClient(
        TtsConfig(backend="fish_cloud", base_url="https://api.fish.audio", api_key="k")
    ).health()
    assert health.reachable is None
    assert "not checked" in health.detail


def test_a_tts_config_never_prints_its_key():
    """It is a dataclass, so its repr is what an error message or a debug
    line shows - and it used to include the Fish key verbatim."""
    key = "fish-" + "k" * 8 + "-live"
    config = TtsConfig(backend="fish_cloud", api_key=key)
    assert key not in repr(config)
    assert config.api_key == key
