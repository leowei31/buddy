"""Talking to a running session from outside it.

Control lived in voice and the CLI alone while the read path earned trust.
This is the third surface, added deliberately, and the shape of it is the
part that matters.

**It is a unix socket, not an HTTP route.** The dashboard stays read-only -
every route there is still GET or HEAD, and `test_web.py` still enforces it.
A socket at `~/.buddy/control.sock`, mode 0600, cannot be reached by a
browser at all: no origin to check, no port to scan, no CSRF, and the
filesystem does the authorisation. The overlay's *page* never touches it
either; the page asks its own Python process, and that process holds the
socket.

**It does not touch the brain directly.** An utterance goes onto the same
kind of queue the voice layer uses, and the session's own loop takes it
from there - so there is exactly one thing calling `brain.send()`, whatever
asked for it.

Anything that needs your confirmation still asks in the terminal, because
that is where the session is. With `trust_mode` on there is nothing to ask.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Newline-delimited JSON. One object per line, in both directions.
ENCODING = "utf-8"

#: A refusal rather than an out-of-memory: nothing legitimate is this long.
MAX_LINE = 64 * 1024

#: `sun_path` is 104 bytes on macOS, 108 on Linux. The smaller one is the
#: portable limit, minus room for the filename itself.
MAX_SOCKET_PATH = 100


def socket_path(home: Path) -> Path:
    return home / "control.sock"


@dataclass
class Utterance:
    """Something said to the session from outside it."""

    text: str
    reply: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)

    def send(self, kind: str, text: str = "") -> None:
        self.reply.put_nowait({"type": kind, "text": text})

    def finish(self, text: str) -> None:
        self.send("reply", text)
        self.reply.put_nowait({"type": "end"})

    def fail(self, why: str) -> None:
        self.send("error", why)
        self.reply.put_nowait({"type": "end"})


class ControlServer:
    """The session's end of the channel.

    Holds a queue of utterances for the session loop to drain. It never calls
    the brain itself, which is what keeps "one thing drives the conversation"
    true no matter how many surfaces there are.
    """

    def __init__(
        self, home: Path, *, on_status: Callable[[], dict[str, Any]] | None = None
    ) -> None:
        self.home = home
        self.path = socket_path(home)
        self.heard: asyncio.Queue[Utterance] = asyncio.Queue()
        self.on_status = on_status or (lambda: {})
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # `sun_path` is 104 bytes on macOS and 108 on Linux, and overflowing
        # it fails as a bare "AF_UNIX path too long" that says nothing about
        # what to change. A deep BUDDY_HOME is the way anyone reaches this.
        if len(str(self.path).encode()) >= MAX_SOCKET_PATH:
            raise OSError(
                f"{self.path} is too long to be a unix socket "
                f"({MAX_SOCKET_PATH} bytes at most). Set BUDDY_HOME to a "
                "shorter path."
            )
        # A socket left by a crashed session would refuse to bind. Removing a
        # stale one is safe; removing a live one is not, so it is connected to
        # first and left alone if something answers.
        if self.path.exists():
            if await self._someone_home():
                raise OSError(f"another session is already listening on {self.path}")
            self.path.unlink(missing_ok=True)

        self._server = await asyncio.start_unix_server(self._client, path=str(self.path))
        # Before anything can connect: the filesystem is the authorisation.
        os.chmod(self.path, 0o600)

    async def _someone_home(self) -> bool:
        try:
            _, writer = await asyncio.open_unix_connection(str(self.path))
        except (TimeoutError, OSError):
            return False
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        self.path.unlink(missing_ok=True)

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    return
                if len(raw) > MAX_LINE:
                    await _write(writer, {"type": "error", "text": "that is too long to be a turn"})
                    continue
                try:
                    message = json.loads(raw.decode(ENCODING))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    await _write(writer, {"type": "error", "text": "expected one JSON object"})
                    continue
                await self._handle(message, writer)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            return
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _handle(self, message: Any, writer: asyncio.StreamWriter) -> None:
        if not isinstance(message, dict):
            await _write(writer, {"type": "error", "text": "expected one JSON object"})
            return
        kind = message.get("type")

        if kind == "ping":
            await _write(writer, {"type": "pong", **self.on_status()})
            return

        if kind != "say":
            await _write(writer, {"type": "error", "text": f"unknown request {kind!r}"})
            return

        text = str(message.get("text", "")).strip()
        if not text:
            await _write(writer, {"type": "error", "text": "nothing to say"})
            return

        utterance = Utterance(text=text)
        self.heard.put_nowait(utterance)
        # Stream whatever the session produces for this turn straight back.
        while True:
            part = await utterance.reply.get()
            if part.get("type") == "end":
                await _write(writer, {"type": "end"})
                return
            await _write(writer, part)


async def _write(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    writer.write((json.dumps(payload) + "\n").encode(ENCODING))
    with contextlib.suppress(Exception):
        await writer.drain()


# --------------------------------------------------------------------------
# The other end
# --------------------------------------------------------------------------


class NotListening(RuntimeError):
    """No session is running, or it has no control channel."""


class ControlClient:
    """Used by the overlay's own process. Never by its page."""

    def __init__(self, home: Path, *, timeout: float = 300.0) -> None:
        self.path = socket_path(home)
        self.timeout = timeout

    async def _open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if not self.path.exists():
            raise NotListening(f"nothing is listening on {self.path}")
        try:
            return await asyncio.open_unix_connection(str(self.path))
        except OSError as exc:
            raise NotListening(f"could not reach the session: {exc}") from exc

    async def ping(self) -> dict[str, Any]:
        reader, writer = await self._open()
        try:
            await _write(writer, {"type": "ping"})
            async with asyncio.timeout(5):
                line = await reader.readline()
            return json.loads(line.decode(ENCODING)) if line else {}
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def say(self, text: str, *, on_delta: Callable[[str], None] | None = None) -> str:
        """Send a turn and wait for the reply."""
        reader, writer = await self._open()
        parts: list[str] = []
        try:
            await _write(writer, {"type": "say", "text": text})
            async with asyncio.timeout(self.timeout):
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    message = json.loads(line.decode(ENCODING))
                    kind = message.get("type")
                    if kind == "end":
                        break
                    if kind == "error":
                        raise NotListening(message.get("text", "the session refused"))
                    if kind == "delta" and on_delta:
                        on_delta(message.get("text", ""))
                    elif kind == "reply":
                        parts.append(message.get("text", ""))
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        return "".join(parts)


def say_sync(home: Path, text: str, *, timeout: float = 300.0) -> str:
    """Blocking, for the overlay's Cocoa thread, which has no event loop."""
    return asyncio.run(ControlClient(home, timeout=timeout).say(text))


def is_listening(home: Path) -> bool:
    path = socket_path(home)
    if not path.exists():
        return False

    async def probe() -> bool:
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
        except OSError:
            return False
        try:
            await _write(writer, {"type": "ping"})
            async with asyncio.timeout(3):
                return bool(await reader.readline())
        except (TimeoutError, OSError):
            return False
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    return asyncio.run(probe())
