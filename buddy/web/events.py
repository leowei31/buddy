"""Fan-out of manager events and conversation turns to the dashboard.

Separate from `server.py` on purpose: `cli.py` publishes to the hub on every
tick, and importing FastAPI and uvicorn to do that would put a tenth of a
second on the cold start of `buddy status`. Nothing here imports the web
stack, so the dashboard stays a subscriber rather than a dependency.

The hub is the only thing the read-only dashboard is *given*; everything else
it reads for itself, from the manager's live state or the database.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

#: The two published websocket channels. `/ws/logs` is not one of them:
#: a log is tailed from its file, so it needs no publisher.
EVENTS = "events"
CONVERSATION = "conversation"


def jsonable(value: Any) -> Any:
    """Whatever this is, as something `json.dumps` accepts.

    Buddy's event payloads carry `datetime`, `Path`, and `StrEnum` values;
    each has exactly one sensible wire form and this is where it is decided.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(item) for item in value]
    return value


def event_payload(event: Any) -> dict[str, Any]:
    """One `models.Event` as a JSON object.

    The class name is the discriminator, so a new event type in `models.py`
    reaches the browser without a second table to keep in step here.
    """
    payload: dict[str, Any] = {"type": type(event).__name__}
    for key, value in vars(event).items():
        payload[key] = jsonable(value)
    return payload


def turn_payload(turn: dict[str, Any]) -> dict[str, Any]:
    """One `conversation_log` row as a JSON object."""
    return {
        "id": turn.get("id"),
        "ts": jsonable(turn.get("ts")),
        "speaker": turn.get("speaker"),
        "text": turn.get("text"),
    }


class Hub:
    """Publish/subscribe with bounded queues.

    A subscriber that cannot keep up loses its oldest message rather than
    growing without limit, and publishing never awaits. The dashboard is a
    viewer: a browser tab that stalls, or a laptop that sleeps with the tab
    open, must not be able to slow down or wedge a running task.
    """

    def __init__(self, maxsize: int = 512) -> None:
        self._maxsize = maxsize
        self._channels: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        #: Messages dropped because a subscriber fell behind. Surfaced in
        #: tests and useful if the dashboard ever looks like it skipped one.
        self.dropped = 0

    @contextmanager
    def subscribe(self, channel: str) -> Iterator[asyncio.Queue[dict[str, Any]]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self._maxsize)
        self._channels.setdefault(channel, set()).add(queue)
        try:
            yield queue
        finally:
            self._channels.get(channel, set()).discard(queue)

    def subscribers(self, channel: str) -> int:
        return len(self._channels.get(channel, ()))

    def publish(self, channel: str, message: dict[str, Any]) -> None:
        for queue in self._channels.get(channel, ()):
            if queue.full():
                queue.get_nowait()
                self.dropped += 1
            queue.put_nowait(message)

    def publish_events(self, events: Iterable[Any]) -> None:
        for event in events:
            self.publish(EVENTS, event_payload(event))

    def publish_turn(self, turn: dict[str, Any]) -> None:
        self.publish(CONVERSATION, turn_payload(turn))
