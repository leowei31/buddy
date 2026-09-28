"""The Fish Audio client: local server or cloud, one implementation.

Every byte of Fish's wire format is in this module and nowhere else, so
`session.py` asks for audio and never knows whether it came from a container
on this machine or from api.fish.audio.

**Verified against the current API rather than typed from memory.** What
the API documents, as of 2026-09, is:

- `POST /v1/tts` takes `application/msgpack` (or JSON) with `text`,
  `reference_id`, `format` (wav | pcm | mp3 | opus, default mp3),
  `chunk_length`, `normalize`, `latency`, `temperature`, `top_p`.
  There is **no `streaming` field**: the response is chunked transfer
  encoding, so audio arrives progressively whether or not you ask for it.
  A `streaming: true` field does not exist, and sending it would be a field
  the server ignores at best.
- `WS /v1/tts/live` frames MessagePack events: `{"event": "start",
  "request": {...}}`, then `{"event": "text", "text": ...}`,
  `{"event": "flush"}`, `{"event": "stop"}`. The server answers with
  `{"event": "audio", "audio": <bytes>}` and finally
  `{"event": "finish", "reason": "stop" | "error"}`.
- Auth is `Authorization: Bearer`, and the model is an optional `model:`
  header rather than a body field. The model line moves fast, so
  Buddy sends whatever config names and `doctor` reports what answered.

`pcm` is the default format here rather than the API's `mp3`: raw samples go
straight to the speaker with nothing to decode, which is the whole latency
budget argument.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

from buddy.config import Config, SecretResolver, VoiceSection

#: Fish's own default. Buddy asks for pcm, but a server that only knows the
#: documented set still gets a format it recognises.
FORMATS = ("wav", "pcm", "mp3", "opus")

#: What `latency` accepts. "balanced" trades a little quality for the first
#: byte, which is the trade a two-second spoken turn wants.
LATENCIES = ("normal", "balanced")

HEALTH_PATH = "/v1/health"
TTS_PATH = "/v1/tts"
LIVE_PATH = "/v1/tts/live"


class TtsError(Exception):
    pass


class TtsNotInstalled(TtsError):
    def __init__(self, package: str) -> None:
        super().__init__(
            f"speech needs the {package!r} package. Install it with: "
            "uv tool install 'buddy-orchestrator[voice]' (or `uv sync --extra voice` "
            "from a source checkout)."
        )


@dataclass(frozen=True)
class TtsConfig:
    """Everything the client needs, resolved from `[voice]`."""

    backend: str = "fish_local"
    base_url: str = "http://127.0.0.1:8080"
    #: Never in `repr`: a config printed in an error message or a log line
    #: must not carry the key with it.
    api_key: str | None = field(default=None, repr=False)
    model: str = ""  # sent as the `model` header; blank lets the server choose
    reference_id: str = ""
    audio_format: str = "pcm"
    sample_rate: int = 44100
    latency: str = "balanced"
    chunk_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.7
    #: "websocket" | "http" | "auto". "auto" probes once at startup.
    streaming: str = "auto"
    connect_timeout: float = 5.0
    request_timeout: float = 60.0

    @property
    def is_cloud(self) -> bool:
        return self.backend == "fish_cloud"

    @property
    def websocket_url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[-1].rstrip("/")
        return f"{scheme}://{host}{LIVE_PATH}"

    def headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        if self.model:
            headers["model"] = self.model
        return headers

    def request(self, text: str = "") -> dict[str, Any]:
        """The `/v1/tts` body, which the WebSocket start event reuses."""
        body: dict[str, Any] = {
            "text": text,
            "format": self.audio_format,
            "chunk_length": self.chunk_length,
            "normalize": True,
            "latency": self.latency,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.reference_id:
            body["reference_id"] = self.reference_id
        if self.audio_format == "pcm" and self.sample_rate:
            # pcm carries no header, so the sample rate has to be agreed
            # rather than discovered.
            body["sample_rate"] = self.sample_rate
        return body


def config_for(
    voice: VoiceSection,
    resolver: SecretResolver,
    *,
    backend: str | None = None,
) -> TtsConfig:
    """Resolve `[voice]` into a client config.

    The API key is looked up through the resolver, so it comes from the
    environment or the OS keychain and never from `config.toml`.
    """
    chosen = backend or voice.tts_backend
    block = voice.fish_cloud if chosen == "fish_cloud" else voice.fish_local
    key_env = str(block.get("api_key_env", "FISH_API_KEY"))
    default_url = "https://api.fish.audio" if chosen == "fish_cloud" else "http://127.0.0.1:8080"
    return TtsConfig(
        backend=chosen,
        base_url=str(block.get("base_url", default_url)).rstrip("/"),
        api_key=resolver.resolve(key_env) if chosen == "fish_cloud" else None,
        model=str(block.get("model", "")),
        reference_id=str(block.get("reference_id", block.get("reference", ""))),
        audio_format=str(block.get("format", "pcm")),
        sample_rate=int(block.get("sample_rate", 44100)),
        latency=str(block.get("latency", "balanced")),
        streaming=str(block.get("streaming", "auto")),
    )


def config_from(config: Config, resolver: SecretResolver, **kwargs) -> TtsConfig:
    return config_for(config.voice, resolver, **kwargs)


@dataclass
class Health:
    """What `doctor` reports about the configured backend."""

    #: None when it cannot be known without spending a synthesis.
    reachable: bool | None
    backend: str
    base_url: str
    model: str = ""  # what actually answered, not what config asked for
    detail: str = ""
    ttfb_seconds: float | None = None
    websocket: bool | None = None  # None = not probed

    def summary(self) -> str:
        if not self.reachable:
            return f"{self.backend} at {self.base_url}: {self.detail}"
        parts = [f"{self.backend} at {self.base_url}"]
        if self.model:
            parts.append(f"model {self.model}")
        if self.ttfb_seconds is not None:
            parts.append(f"first byte in {self.ttfb_seconds * 1000:.0f} ms")
        if self.websocket is not None:
            parts.append("websocket" if self.websocket else "per-sentence HTTP")
        return ", ".join(parts)


def _packb(payload: dict[str, Any]) -> bytes:
    try:
        import msgpack
    except ImportError as exc:  # pragma: no cover - the voice extra is optional
        raise TtsNotInstalled("msgpack") from exc
    return msgpack.packb(payload, use_bin_type=True)


def _unpackb(raw: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(raw, raw=False)


class FishClient:
    """One client, two backends.

    Nothing here plays audio or splits sentences; it turns text into bytes.
    """

    def __init__(self, config: TtsConfig) -> None:
        self.config = config
        self._client: Any = None
        #: Cached answer to "does this backend do incremental text?" so the
        #: websocket probe happens once per session, not once per sentence.
        self._websocket: bool | None = (
            None if config.streaming == "auto" else (config.streaming == "websocket")
        )

    async def __aenter__(self) -> FishClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _http(self):
        if self._client is None:
            import httpx2

            self._client = httpx2.AsyncClient(
                base_url=self.config.base_url,
                headers=self.config.headers(),
                timeout=httpx2.Timeout(
                    self.config.request_timeout, connect=self.config.connect_timeout
                ),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

    # -- health -----------------------------------------------------

    async def health(self) -> Health:
        """`GET /v1/health`, and what model answered.

        The cloud has no health endpoint, so reaching it is proved by the
        same one-sentence synthesis that measures TTFB - which is a better
        test anyway, since it exercises auth and the model header too.
        """
        import httpx2

        base = Health(reachable=False, backend=self.config.backend, base_url=self.config.base_url)
        if self.config.is_cloud:
            if not self.config.api_key:
                base.detail = "no API key; set FISH_API_KEY or run `buddy setup --force keys`"
                return base
            # Unknown, not down: the cloud has no health endpoint, and saying
            # "unreachable" about a working backend is a false alarm (review
            # voice #7). `measure` settles it with a real synthesis.
            base.reachable = None
            base.detail = "not checked; the cloud has no health endpoint"
            return base

        try:
            response = await self._http().get(HEALTH_PATH)
        except (httpx2.HTTPError, OSError) as exc:
            base.detail = f"unreachable ({type(exc).__name__}: {exc})"
            return base
        if response.status_code != 200:
            base.detail = f"health returned {response.status_code}"
            return base

        model = ""
        with contextlib.suppress(Exception):
            body = response.json()
            if isinstance(body, dict):
                # Report whichever checkpoint the server is running,
                # rather than asserting one from the docs.
                model = str(body.get("model") or body.get("checkpoint") or "")
        base.reachable = True
        base.model = model
        base.detail = "ok"
        return base

    # -- 6.2, non-streaming ------------------------------------------------

    async def synthesize(self, text: str) -> bytes:
        """The whole clip in one call. The simplest thing that can work."""
        chunks = [chunk async for chunk in self.stream(text)]
        return b"".join(chunks)

    # -- per-sentence HTTP streaming, the guaranteed path -----------------

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Audio for one piece of text, yielded as it arrives.

        No `streaming` flag is sent: the endpoint answers with chunked
        transfer encoding regardless, which is what makes this the path that
        always works.
        """
        import httpx2

        request = self._http().build_request(
            "POST",
            TTS_PATH,
            content=_packb(self.config.request(text)),
            headers={"content-type": "application/msgpack"},
        )
        try:
            response = await self._http().send(request, stream=True)
        except (httpx2.HTTPError, OSError) as exc:
            raise TtsError(f"{self.config.backend}: {type(exc).__name__}: {exc}") from exc

        try:
            if response.status_code != 200:
                body = (await response.aread())[:200].decode(errors="replace")
                raise TtsError(f"{self.config.backend}: {response.status_code} {body}")
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
        finally:
            await response.aclose()

    # -- incremental text over the websocket ------------------------------

    async def supports_websocket(self) -> bool:
        """Probe once, rather than assume.

        The open-source server may or may not expose `/v1/tts/live`; the
        answer decides between incremental text and per-sentence HTTP for the
        rest of the session, and is never assumed either way.
        """
        if self._websocket is not None:
            return self._websocket
        self._websocket = await self._probe_websocket()
        return self._websocket

    async def _probe_websocket(self) -> bool:
        import websockets

        try:
            async with asyncio.timeout(self.config.connect_timeout):
                connection = await websockets.connect(
                    self.config.websocket_url, additional_headers=self.config.headers()
                )
        except Exception:  # noqa: BLE001 - any failure means "use HTTP"
            return False
        await connection.close()
        return True

    async def stream_incremental(self, chunks: Iterable[str]) -> AsyncIterator[bytes]:
        """Feed text as it is generated, get audio as it is synthesised.

        The lowest-latency path when the backend has it: the model starts
        speaking the first clause while the brain is still writing the
        second.
        """
        async for audio in self.stream_incremental_async(_as_async(chunks)):
            yield audio

    async def stream_incremental_async(self, chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        import websockets

        try:
            connection = await websockets.connect(
                self.config.websocket_url,
                additional_headers=self.config.headers(),
                # The close handshake waits on the server too; a stalled one
                # would otherwise hold the reply open long after the error.
                close_timeout=1.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise TtsError(f"{self.config.websocket_url}: {type(exc).__name__}: {exc}") from exc

        async with connection:
            await connection.send(_packb({"event": "start", "request": self.config.request("")}))
            sender = asyncio.create_task(_send_text(connection, chunks))
            try:
                while True:
                    # Bounded per frame, like the HTTP path's request timeout:
                    # a server that accepts and then goes quiet must not hold
                    # a spoken reply open forever.
                    try:
                        async with asyncio.timeout(self.config.request_timeout):
                            frame = await connection.recv()
                    except TimeoutError as exc:
                        raise TtsError(
                            f"{self.config.websocket_url}: nothing for "
                            f"{self.config.request_timeout:g}s"
                        ) from exc
                    except websockets.ConnectionClosed:
                        break
                    event = _unpackb(frame) if isinstance(frame, bytes) else {}
                    kind = event.get("event")
                    if kind == "audio" and event.get("audio"):
                        yield event["audio"]
                    elif kind == "finish":
                        if event.get("reason") == "error":
                            raise TtsError(
                                f"the server ended the stream with an error: "
                                f"{event.get('message', 'no detail given')}"
                            )
                        break
            finally:
                sender.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await sender

    # -- what doctor measures --------------------------------------

    async def measure(self, text: str = "Buddy is ready.") -> Health:
        """Health, model, real time-to-first-byte, and which path is in play.

        `doctor` reports this rather than a number from a document, so the
        two-second budget for a spoken turn is checked against the machine in
        front of you.
        """
        health = await self.health()
        if self.config.is_cloud:
            if not self.config.api_key:
                # No point spending a round trip to be told what is already
                # known, and a raw 401 body in `doctor`'s table reads like a
                # fault rather than a setting nobody chose.
                return health
            health.reachable = False  # until the synthesis below proves it

        started = time.monotonic()
        first: float | None = None
        try:
            async for chunk in self.stream(text):
                if chunk:
                    first = time.monotonic() - started
                    break
        except TtsError as exc:
            health.reachable = False
            health.detail = str(exc)
            return health

        health.reachable = True
        health.detail = "ok"
        health.ttfb_seconds = first
        health.websocket = await self.supports_websocket()
        return health


async def _as_async(chunks: Iterable[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


async def _send_text(connection: Any, chunks: AsyncIterator[str]) -> None:
    """Text in, flushed at each chunk so audio starts before the text ends."""
    async for chunk in chunks:
        if not chunk:
            continue
        await connection.send(_packb({"event": "text", "text": chunk}))
        await connection.send(_packb({"event": "flush"}))
    await connection.send(_packb({"event": "stop"}))


# --------------------------------------------------------------------------
# Falling back
# --------------------------------------------------------------------------


class FallbackClient:
    """The configured backend, and `tts_fallback` for when it is down.

    Cloud is the fallback, not the primary, and this is the piece that makes
    that sentence true: a sentence that cannot even start on the primary is
    spoken on the fallback instead, and every sentence after it goes there
    too - retrying a dead server once per sentence would add its timeout to
    every reply.

    Two cases never switch. Audio that already started is not repeated on
    another backend; the error stands. And a fallback at a different sample
    rate is refused, because playback was opened for the primary's, and
    audio at the wrong rate plays at the wrong speed.
    """

    def __init__(self, primary: FishClient, fallback: FishClient | None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.active = primary
        #: Empty until the switch happens; then, why it happened.
        self.switched_because = ""

    @property
    def config(self) -> TtsConfig:
        return self.active.config

    def use_fallback(self, why: str) -> None:
        if self.fallback is not None:
            self.active = self.fallback
            self.switched_because = why

    def can_fall_back(self) -> bool:
        return (
            self.fallback is not None
            and self.active is self.primary
            and self.fallback.config.sample_rate == self.primary.config.sample_rate
            and not (self.fallback.config.is_cloud and not self.fallback.config.api_key)
        )

    async def health(self) -> Health:
        return await self.active.health()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        started = False
        try:
            async for chunk in self.active.stream(text):
                started = True
                yield chunk
            return
        except TtsError as exc:
            if started or not self.can_fall_back():
                raise
            self.use_fallback(f"{exc}")
        async for chunk in self.active.stream(text):
            yield chunk

    async def aclose(self) -> None:
        await self.primary.aclose()
        if self.fallback is not None:
            await self.fallback.aclose()


async def choose_client(
    voice: VoiceSection, resolver: SecretResolver
) -> tuple[FallbackClient, str]:
    """The client a session speaks through, and a sentence saying which.

    A local primary is health-checked first, because it has an endpoint for
    that; if it does not answer, the session starts on the fallback rather
    than discovering the outage on its first reply.
    """
    primary = FishClient(config_for(voice, resolver))
    fallback = None
    if voice.tts_fallback and voice.tts_fallback != primary.config.backend:
        fallback = FishClient(config_for(voice, resolver, backend=voice.tts_fallback))
    client = FallbackClient(primary, fallback)
    if not primary.config.is_cloud:
        health = await primary.health()
        if not health.reachable and client.can_fall_back():
            where = f"{primary.config.backend} at {primary.config.base_url}"
            client.use_fallback(f"{where}: {health.detail}")
            return client, (
                f"{primary.config.backend} is not answering ({health.detail}), "
                f"so speaking through {client.config.backend}"
            )
    return client, f"speaking through {client.config.backend}"
