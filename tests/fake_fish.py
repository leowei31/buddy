"""A Fish Audio server that speaks the documented protocol.

Not a mock of Buddy's client - a server on a real socket, framing real
MessagePack, answering `/v1/health`, `/v1/tts` and `/v1/tts/live`. That is the
only way to test the thing that actually matters: that Buddy's bytes are
the bytes Fish expects, and that audio starts arriving before the text ends.

It is deliberately strict. A request with the wrong content type, a missing
bearer token, or an event the protocol does not define is refused, so a client
that drifts from the spec fails here rather than in front of a microphone.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import msgpack
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse


def audio_for(text: str) -> bytes:
    """What the fake "synthesises": the first byte of the text, repeated once
    per character, so a test can tell which text produced which audio."""
    return text.encode()[:1] * max(1, len(text.strip()))


@dataclass
class Recorder:
    """What the server was asked to do, for the tests to assert against."""

    requests: list[dict[str, Any]] = field(default_factory=list)
    #: When each request arrived, so a test can prove that sentence two was
    #: already in flight while sentence one was still playing.
    request_times: list[float] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)
    live_events: list[dict[str, Any]] = field(default_factory=list)
    live_connections: int = 0


def create_app(
    recorder: Recorder,
    *,
    model: str = "s2.1-pro",
    require_key: bool = False,
    websocket: bool = True,
    chunk_delay: float = 0.0,
    first_byte_delay: float = 0.0,
    live_stall: float = 0.0,
) -> FastAPI:
    app = FastAPI()

    def authorised(request_headers) -> bool:
        if not require_key:
            return True
        return str(request_headers.get("authorization", "")).startswith("Bearer ")

    @app.get("/v1/health")
    async def health() -> JSONResponse:
        # `doctor` reports whichever checkpoint the server is running,
        # so the server is the one that names it.
        return JSONResponse({"status": "ok", "model": model})

    @app.post("/v1/tts")
    async def tts(request: Request) -> Any:
        recorder.content_types.append(request.headers.get("content-type", ""))
        recorder.headers.append(dict(request.headers))
        if not authorised(request.headers):
            return JSONResponse({"error": "unauthorised"}, status_code=401)
        raw = await request.body()
        if "msgpack" not in request.headers.get("content-type", ""):
            return JSONResponse({"error": "expected application/msgpack"}, status_code=415)
        try:
            body = msgpack.unpackb(raw, raw=False)
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "undecodable msgpack"}, status_code=400)
        recorder.requests.append(body)
        recorder.request_times.append(time.monotonic())

        audio = audio_for(str(body.get("text", "")))

        async def chunks():
            if first_byte_delay:
                await asyncio.sleep(first_byte_delay)
            for index in range(0, len(audio), 4):
                if chunk_delay:
                    await asyncio.sleep(chunk_delay)
                yield audio[index : index + 4]

        return StreamingResponse(chunks(), media_type="application/octet-stream")

    if websocket:

        @app.websocket("/v1/tts/live")
        async def live(connection: WebSocket) -> None:
            if not authorised(connection.headers):
                await connection.close(code=1008, reason="unauthorised")
                return
            await connection.accept()
            recorder.live_connections += 1
            if live_stall:
                # A server that accepted and then went quiet.
                await asyncio.sleep(live_stall)
            buffered: list[str] = []
            try:
                while True:
                    event = msgpack.unpackb(await connection.receive_bytes(), raw=False)
                    recorder.live_events.append(event)
                    kind = event.get("event")
                    if kind == "start":
                        recorder.requests.append(event.get("request", {}))
                    elif kind == "text":
                        buffered.append(str(event.get("text", "")))
                    elif kind == "flush":
                        # The point of the protocol: audio for what has been
                        # said so far, without waiting for the rest.
                        pending, buffered = "".join(buffered), []
                        if pending.strip():
                            await connection.send_bytes(
                                msgpack.packb(
                                    {"event": "audio", "audio": audio_for(pending)},
                                    use_bin_type=True,
                                )
                            )
                    elif kind == "stop":
                        pending = "".join(buffered)
                        if pending.strip():
                            await connection.send_bytes(
                                msgpack.packb(
                                    {"event": "audio", "audio": audio_for(pending)},
                                    use_bin_type=True,
                                )
                            )
                        await connection.send_bytes(
                            msgpack.packb({"event": "finish", "reason": "stop"}, use_bin_type=True)
                        )
                        return
                    else:
                        await connection.send_bytes(
                            msgpack.packb(
                                {"event": "finish", "reason": "error", "message": f"?{kind}"},
                                use_bin_type=True,
                            )
                        )
                        return
            except WebSocketDisconnect:
                return

    return app


class FakeFish:
    """The server on a real port, started and stopped like the dashboard's."""

    def __init__(self, **options: Any) -> None:
        self.recorder = Recorder()
        self.app = create_app(self.recorder, **options)
        self.port = 0
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> FakeFish:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        self.port = sock.getsockname()[1]
        self._socket = sock

        config = uvicorn.Config(
            self.app, host="127.0.0.1", port=self.port, log_level="error", access_log=False
        )
        server = uvicorn.Server(config)
        server.capture_signals = contextlib.nullcontext  # type: ignore[method-assign]
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._server = server
        self._task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if self._task.done():
                await self._task
            await asyncio.sleep(0.01)
        return self

    async def stop(self) -> None:
        if self._server and self._task:
            self._server.should_exit = True
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except TimeoutError:  # pragma: no cover
                self._server.force_exit = True
                await self._task
        if self._socket:
            self._socket.close()
            self._socket = None
