"""FastAPI: routes, websockets, and log tailing.

Embedded in the Buddy process and running in the same asyncio loop as the
manager. Live slot state comes from `AgentManager`, history from the
database, and output from tailing the log files `pipe-pane` is already
writing - there is no second polling loop and no `capture-pane` diffing.

**Read-only by design.** There is no route here that changes
anything: no kill, no spawn, no merge. Voice and the CLI are the control
surfaces, and a third one racing them buys a class of bugs that is not worth
taking on before the read path has earned trust. `test_web.py` asserts the
absence structurally, so adding a POST is a test failure rather than a code
review someone has to catch.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from buddy.config import Config
from buddy.ideation import Brainstorm
from buddy.logs import tail
from buddy.manager import AgentManager
from buddy.models import SLOT_NAMES, AgentSlot, TaskRun, TaskSpec, TaskState
from buddy.state import Store
from buddy.web.events import CONVERSATION, EVENTS, Hub, jsonable
from buddy.workspace import Workspace, WorkspaceError, branch_name

STATIC = Path(__file__).parent / "static"

#: How often a tailing websocket looks for new bytes. The log is a local file
#: an already-running `pipe-pane` appends to, so this costs one `stat`.
LOG_POLL_SECONDS = 0.25

#: Bytes read per tail iteration, so a large backfill streams into the
#: terminal instead of arriving as one enormous frame.
LOG_CHUNK_BYTES = 256 * 1024

#: How much of an existing log a fresh connection is sent. A task that has run
#: for an hour can leave far more than a browser wants; xterm.js has a
#: scrollback limit regardless, and `buddy logs` is there for the whole file.
INITIAL_TAIL_BYTES = 1024 * 1024

#: The only names the dashboard answers to. It binds to loopback, so these
#: are the only names that legitimately reach it.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


def _seconds_since(then: datetime | None, now: datetime) -> int | None:
    return None if then is None else max(0, int((now - then).total_seconds()))


def _slot_sort_key(slot: AgentSlot) -> tuple[bool, int, int]:
    """Sorted by priority, with unassigned slots last and the week's
    order as the tie-break so cards never swap places between polls."""
    order = {name: index for index, name in enumerate(SLOT_NAMES)}
    return (slot.priority is None, slot.priority or 0, order[slot.name])


def run_payload(run: TaskRun) -> dict[str, Any]:
    log = run.log_path
    return {
        "task_id": run.task_id,
        "attempt": run.attempt,
        "slot": run.slot,
        "branch": run.branch,
        "base_ref": run.base_ref,
        "worktree": str(run.worktree),
        "log_path": str(log),
        "log_exists": log.exists(),
        "log_bytes": log.stat().st_size if log.exists() else 0,
        "started_at": jsonable(run.started_at),
        "ended_at": jsonable(run.ended_at),
        "exit_code": run.exit_code,
        "outcome": jsonable(run.outcome),
        "wip_commit": run.wip_commit,
    }


def task_payload(task: TaskSpec, state: TaskState | None) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "project": task.project,
        "harness": task.harness,
        "priority": task.priority,
        "state": jsonable(state),
        "attempt": task.attempt,
        "model": task.model,
        "depends_on": list(task.depends_on),
        "merge_required": task.merge_required,
        "created_at": jsonable(task.created_at),
        "created_from_utterance": task.created_from_utterance,
    }


# --------------------------------------------------------------------------
# Websocket plumbing
# --------------------------------------------------------------------------


def same_origin(websocket: WebSocket, port: int) -> bool:
    """Whether this handshake came from the dashboard's own page.

    Binding to 127.0.0.1 keeps other machines out, the Host check in
    `create_app` keeps rebound hostnames out, and the same-origin policy
    keeps other pages from reading the JSON routes - but **browsers do not
    apply the same-origin policy to websocket handshakes**. Without this, any page open
    in the operator's browser can `new WebSocket("ws://127.0.0.1:4321/ws/
    conversation")` and read every turn of the conversation, the whole event
    stream, and any task's live output, with ids as guessable as `t-0001`.

    An absent Origin is allowed: that is a native client - `websockets`, a
    test, `wscat` - which is on this machine already and had simpler ways in.
    A *present* Origin has to be ours.
    """
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    return origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}


async def _wait_closed(websocket: WebSocket) -> None:
    """Resolve when the browser goes away.

    Nothing is ever sent *to* the dashboard, so any receive resolving at all
    means the tab was closed, refreshed, or the network dropped.
    """
    with contextlib.suppress(Exception):
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return


async def stream(websocket: WebSocket, produce: AsyncGenerator[str, None]) -> None:
    """Send everything `produce` yields until it stops or the tab closes.

    The two are raced rather than interleaved: a tailer that is quietly
    waiting for a task to write its next line would otherwise not notice a
    closed socket until that line arrived, and would hold the subscription
    open in the meantime.
    """
    await websocket.accept()
    async with contextlib.aclosing(produce) as source:
        sender = asyncio.create_task(_send_all(websocket, source))
        closed = asyncio.create_task(_wait_closed(websocket))
        try:
            await asyncio.wait({sender, closed}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (sender, closed):
                task.cancel()
            for task in (sender, closed):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
    with contextlib.suppress(Exception):
        await websocket.close()


async def _send_all(websocket: WebSocket, produce: AsyncIterator[str]) -> None:
    async for message in produce:
        await websocket.send_text(message)


async def _from_queue(hub: Hub, channel: str) -> AsyncGenerator[str, None]:
    with hub.subscribe(channel) as queue:
        while True:
            yield json.dumps(await queue.get())


#: What the log view says when the file under it is rotated.
LOG_ROTATED_NOTE = "\x1b[2m[buddy] log rotated; the earlier part is in the .1 file\x1b[0m\r\n"


def _log_size(path: Path) -> int | None:
    """Current size, or None if the log is not there (yet)."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def _read_chunk(path: Path, offset: int, limit: int) -> tuple[bytes, int]:
    """One bounded read, run in a thread so the loop never waits on disk."""
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(limit)
    return data, offset + len(data)


async def tail_log(
    path: Path,
    *,
    poll: float = LOG_POLL_SECONDS,
    initial_bytes: int = INITIAL_TAIL_BYTES,
) -> AsyncGenerator[str, None]:
    """Existing content, then whatever gets appended.

    Yields the bytes exactly as `pipe-pane` wrote them, escapes and all, which
    is the whole reason the browser renders this with xterm.js.

    A log that does not exist yet is waited for rather than refused: a slot
    can be connected to before its `pipe-pane` has written a byte.
    """
    offset = -1  # nothing read yet
    while True:
        size = await asyncio.to_thread(_log_size, path)
        if size is not None:
            if offset < 0:
                offset = max(0, size - initial_bytes)
                if offset:
                    skipped = offset // 1024
                    yield f"\x1b[2m[buddy] skipped the first {skipped} KiB of this log\x1b[0m\r\n"
            elif size < offset:
                # Rotated past `max_log_mb`, or replaced: start the new file
                # on a fresh terminal rather than glue it onto the old one.
                offset = 0
                yield "\x1bc" + LOG_ROTATED_NOTE
            while offset < size:
                data, offset = await asyncio.to_thread(_read_chunk, path, offset, LOG_CHUNK_BYTES)
                if not data:
                    break
                yield data.decode("utf-8", errors="replace")
        await asyncio.sleep(poll)


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------


def create_app(
    *,
    config: Config,
    store: Store,
    manager: AgentManager,
    workspace: Workspace,
    hub: Hub,
) -> FastAPI:
    app = FastAPI(
        title="Buddy",
        description="Read-only view of the orchestrator.",
        version="1",
    )
    # The port the page is served from, which is the only origin its own
    # websockets may come from. Set again by `Dashboard.start` once a port 0
    # request has been given a real one.
    app.state.buddy_port = config.buddy.web_port
    # Binding to 127.0.0.1 keeps other machines out, not other web pages: a
    # page can re-point its own hostname at 127.0.0.1 (DNS rebinding), and
    # to the browser it is then same-origin with every route here. What it
    # cannot change is the Host it sends, so anything but loopback is refused
    # - measured before this, `Host: attacker.example` read the conversation.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOOPBACK_HOSTS))
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def slot_payload(slot: AgentSlot, now: datetime) -> dict[str, Any]:
        task = store.get_task(slot.task_id) if slot.task_id else None
        run = (
            store.get_run(slot.task_id, slot.run_attempt)
            if slot.task_id and slot.run_attempt
            else None
        )
        return {
            "name": slot.name,
            "status": jsonable(slot.status),
            "occupied": slot.status.is_occupied,
            "task_id": slot.task_id,
            "attempt": slot.run_attempt,
            "title": task.title if task else None,
            "project": task.project if task else None,
            "harness": slot.harness,
            "priority": slot.priority,
            "branch": run.branch if run else None,
            "started_at": jsonable(slot.started_at),
            "age_seconds": _seconds_since(slot.started_at, now),
            "last_output_at": jsonable(slot.last_output_at),
            "last_output_ago_seconds": _seconds_since(slot.last_output_at, now),
            "tail": tail(slot.last_output, 3),
        }

    def queue_row(task: TaskSpec, state: TaskState | None) -> dict[str, Any]:
        """A task row with everything the queue view draws."""
        payload = task_payload(task, state)
        run = store.latest_run(task.id)
        payload["branch"] = run.branch if run else None
        payload["outcome"] = jsonable(run.outcome) if run else None
        payload["ended_at"] = jsonable(run.ended_at) if run else None
        # The live slot while it holds the task, and the slot it last ran in
        # once it does not: a finished task saying "nowhere" is less true than
        # saying where the work happened.
        payload["slot"] = next(
            (slot.name for slot in manager.slots() if slot.task_id == task.id),
            run.slot if run else None,
        )
        if state is TaskState.QUEUED:
            blocked = manager.dependency_block(task)
            payload["waiting_for"] = blocked
            payload["queue_position"] = None if blocked else manager.queue_position(task.id)
        else:
            payload["waiting_for"] = None
            payload["queue_position"] = None
        return payload

    # -- pages -------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/overlay", include_in_schema=False)
    async def overlay() -> FileResponse:
        """The always-on-top panel's page.

        Served from the same app as the dashboard so it is read-only in the
        same way: it reads the same API and there is nothing here it could
        change.
        """
        return FileResponse(STATIC / "overlay.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")

    # -- the API -----------------------------------------------------------

    @app.get("/api/slots")
    async def slots() -> dict[str, Any]:
        now = manager.now()
        ordered = sorted(manager.slots(), key=_slot_sort_key)
        return {
            "now": jsonable(now),
            "max_concurrent": manager.max_concurrent,
            "slots": [slot_payload(slot, now) for slot in ordered],
        }

    @app.get("/api/brainstorm")
    async def brainstorm() -> dict[str, Any]:
        """Whether Buddy is brainstorming, and the drafts so far. Read from
        the file the session writes, so it is right whichever process asks."""
        return Brainstorm.load(config.home).status()

    @app.get("/api/tasks")
    async def tasks(
        state: str | None = Query(None, description="queued|running|done|error|…"),
        project: str | None = None,
        limit: int = Query(100, ge=1, le=1000),
    ) -> dict[str, Any]:
        if state:
            try:
                wanted = TaskState(state)
            except ValueError as exc:
                allowed = ", ".join(s.value for s in TaskState)
                raise HTTPException(
                    400, f"unknown state {state!r}; expected one of {allowed}"
                ) from exc
            found = store.tasks_in_state(wanted)
            if project:
                found = [task for task in found if task.project == project]
            rows = [queue_row(task, wanted) for task in found[:limit]]
        else:
            rows = [
                queue_row(task, store.get_task_state(task.id))
                for task in store.recent_tasks(project, limit)
            ]
        return {"tasks": rows}

    @app.get("/api/tasks/{task_id}")
    async def task_detail(task_id: str) -> dict[str, Any]:
        task = store.get_task(task_id)
        if task is None:
            raise HTTPException(404, f"no such task: {task_id}")
        state = store.get_task_state(task_id)
        payload: dict[str, Any] = {
            "task": {**queue_row(task, state), "brief": task.brief},
            "runs": [run_payload(run) for run in store.runs_for(task_id)],
            "result": _read_result(config, task_id),
            "diff_stat": "",
            "diff_error": None,
        }
        try:
            payload["diff_stat"] = await workspace.diff_stat(task)
        except (WorkspaceError, KeyError, OSError) as exc:
            # A branch can be gone, merged away, or never have existed if the
            # task has not run yet. That is information, not a 500.
            payload["diff_error"] = str(exc)
        payload["branch"] = branch_name(task)
        return payload

    @app.get("/api/conversation")
    async def conversation(
        before: int | None = Query(None, description="page backwards from this turn id"),
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        turns = store.recent_turns(limit, before_id=before)
        return {
            "turns": turns,
            # Only meaningful when a full page came back; otherwise the
            # transcript simply ended.
            "has_more": len(turns) == limit,
        }

    async def _refuse_other_origins(websocket: WebSocket) -> bool:
        if same_origin(websocket, app.state.buddy_port):
            return False
        # Closed before accept: nothing is streamed to a page that is not ours.
        await websocket.close(code=1008, reason="origin not allowed")
        return True

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket) -> None:
        if await _refuse_other_origins(websocket):
            return
        await stream(websocket, _from_queue(hub, EVENTS))

    @app.websocket("/ws/conversation")
    async def ws_conversation(websocket: WebSocket) -> None:
        if await _refuse_other_origins(websocket):
            return
        await stream(websocket, _from_queue(hub, CONVERSATION))

    @app.websocket("/ws/logs/{task_id}/{attempt}")
    async def ws_logs(websocket: WebSocket, task_id: str, attempt: int) -> None:
        if await _refuse_other_origins(websocket):
            return
        run = store.get_run(task_id, attempt)
        if run is None:
            # Accepted and then closed, rather than refused: a rejected
            # handshake reaches the browser as an opaque network error, while
            # a close carries the 1008 and the reason the page can show. The
            # path is never trusted as a filename either way - the log's
            # location comes from the run row.
            await websocket.accept()
            await websocket.close(code=1008, reason=f"no attempt {attempt} of {task_id}")
            return
        await stream(websocket, tail_log(run.log_path))

    return app


def _read_result(config: Config, task_id: str) -> dict[str, Any] | None:
    """`result.json` for the task, if the wrapper has written one yet."""
    path = config.paths.result_file(task_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


class Dashboard:
    """The dashboard as an asyncio task in Buddy's own loop.

    The listening socket is bound here rather than inside uvicorn so that a
    port already in use surfaces as an `OSError` the caller can turn into one
    sentence, instead of uvicorn logging it and calling `sys.exit`.
    """

    def __init__(self, app: FastAPI, *, host: str = "127.0.0.1", port: int = 4321) -> None:
        self.app = app
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError:
            sock.close()
            raise
        sock.listen(128)
        # Port 0 means "any free port", which is how the tests get one.
        self.port = sock.getsockname()[1]
        self._socket = sock
        self.app.state.buddy_port = self.port

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            timeout_graceful_shutdown=2,
        )
        server = uvicorn.Server(config)
        # Buddy owns Ctrl-C: the session has tasks running in tmux and decides
        # for itself what a signal means. Left alone, uvicorn replaces the
        # process-wide SIGINT and SIGTERM handlers for the life of `serve`,
        # and re-raises anything it caught on the way out.
        #
        # Which method does that moved: `install_signal_handlers` up to
        # uvicorn 0.30, `capture_signals` since. Both are neutralised, so this
        # holds across the range `pyproject.toml` allows rather than only the
        # version that happens to be installed. `test_web.py` checks the
        # handler really is untouched, which is what caught the rename.
        # setattr: which of the two exists depends on the installed uvicorn.
        setattr(server, "install_signal_handlers", lambda: None)  # noqa: B010
        setattr(server, "capture_signals", contextlib.nullcontext)  # noqa: B010
        self._server = server
        self._task = asyncio.create_task(server.serve(sockets=[sock]))

        while not server.started:
            if self._task.done():
                await self._task  # re-raise whatever stopped it
                raise RuntimeError("the dashboard stopped before it started")
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        if self._server is None or self._task is None:
            return
        self._server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=5)
        except TimeoutError:
            self._server.force_exit = True
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=5)
        except asyncio.CancelledError:
            pass
        finally:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            self._server = None
            self._task = None


async def start_dashboard(
    *,
    config: Config,
    store: Store,
    manager: AgentManager,
    workspace: Workspace,
    hub: Hub,
    port: int | None = None,
    on_error: Callable[[str], None] | None = None,
) -> Dashboard | None:
    """Build and start the dashboard, or explain why it is not running.

    A dashboard that cannot bind its port is never a reason to stop: the
    session, the queue, and every running task are unaffected by whether a
    browser tab exists.
    """
    app = create_app(config=config, store=store, manager=manager, workspace=workspace, hub=hub)
    dashboard = Dashboard(app, port=config.buddy.web_port if port is None else port)
    try:
        await dashboard.start()
    except OSError as exc:
        if on_error:
            on_error(
                f"the dashboard could not start on port {dashboard.port} ({exc.strerror or exc}). "
                "Set [buddy] web_port in config.toml, or run with --no-web."
            )
        return None
    return dashboard


__all__ = ["Dashboard", "create_app", "start_dashboard", "stream", "tail_log"]
