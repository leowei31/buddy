"""The read-only dashboard.

Against a real uvicorn server on a real socket, not an in-process transport:
what matters is an embedded server sharing Buddy's asyncio loop, and
half the things that can go wrong there - the signal handlers uvicorn wants to
install, a port already taken, a websocket that outlives its tab - only exist
when there is an actual server. The port is 0, so the suite never collides
with a dashboard the user has open.
"""

from __future__ import annotations

import asyncio
import json
import re
import signal
import subprocess
from datetime import timedelta
from pathlib import Path

import httpx2 as httpx
import pytest
import websockets

from buddy.config import Config
from buddy.harnesses.claude_code import ClaudeCodeAdapter
from buddy.manager import AgentManager
from buddy.models import (
    Agent,
    AgentStatus,
    RunOutcome,
    TaskFinished,
    TaskRun,
    TaskSpec,
    TaskStarted,
    TaskState,
    utcnow,
)
from buddy.state import Store
from buddy.web.events import EVENTS, Hub, event_payload, jsonable
from buddy.web.server import Dashboard, create_app, start_dashboard, tail_log
from buddy.workspace import Workspace, branch_name
from tests.helpers import wait_for


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "code" / "webapp"
    path.mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.name", "Test")
    git(path, "config", "user.email", "test@example.com")
    (path / "README.md").write_text("hello\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "initial")
    return path


@pytest.fixture
def config(tmp_path: Path, repo: Path) -> Config:
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{repo}"\nbase_branch = "main"\n'
        "[harness.claude_code]\n"
        'command = "claude -p < {prompt_path}"\n'
    )
    return Config.load(home=tmp_path)


@pytest.fixture
def store(config: Config) -> Store:
    with Store(config.paths.db) as opened:
        yield opened


@pytest.fixture
def workspace(config: Config) -> Workspace:
    return Workspace(config)


class NullRunner:
    """The dashboard never touches tmux, so neither does this."""

    async def panes(self) -> dict:
        return {}


@pytest.fixture
def manager(config: Config, store: Store, workspace: Workspace) -> AgentManager:
    return AgentManager(
        config,
        store,
        NullRunner(),
        workspace,
        adapter_for=lambda name: ClaudeCodeAdapter(config.harness(name)),
    )


@pytest.fixture
def hub() -> Hub:
    return Hub()


@pytest.fixture
def app(config, store, manager, workspace, hub):
    return create_app(config=config, store=store, manager=manager, workspace=workspace, hub=hub)


@pytest.fixture
async def server(app):
    """A real dashboard on a free port, stopped afterwards."""
    dashboard = Dashboard(app, port=0)
    await dashboard.start()
    try:
        yield dashboard
    finally:
        await dashboard.stop()


@pytest.fixture
async def client(server: Dashboard):
    async with httpx.AsyncClient(base_url=server.url, timeout=10) as opened:
        yield opened


def make_task(store: Store, *, state: TaskState = TaskState.QUEUED, **overrides) -> TaskSpec:
    defaults = {
        "id": store.next_task_id(),
        "title": "Fix onboarding flow",
        "brief": "# Goal\nMake it work\n",
        "harness": "claude_code",
        "project": "webapp",
        "priority": 2,
    }
    task = TaskSpec(**(defaults | overrides))
    store.create_task(task, state)
    return task


def make_run(config: Config, task: TaskSpec, *, attempt: int = 1) -> TaskRun:
    log = config.paths.log_file(task.id, attempt)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    return TaskRun(
        task_id=task.id,
        attempt=attempt,
        agent=task.agent,
        worktree=Path("/tmp/unused"),
        branch=branch_name(task),
        base_ref="HEAD",
        log_path=log,
    )


# -- the read-only guarantee ---------------------------------


def test_no_route_can_change_anything(app):
    """Read-only, asserted structurally.

    Adding a kill button one day should be a deliberate act that fails this
    test, not something that slips in behind a review.
    """
    for route in app.routes:
        methods = getattr(route, "methods", None) or {"GET"}
        assert methods <= {"GET", "HEAD"}, f"{route.path} exposes {methods}"


def test_websocket_routes_match_the_design(app):
    paths = {getattr(route, "path", None) for route in app.routes}
    assert {"/ws/events", "/ws/conversation", "/ws/logs/{task_id}/{attempt}"} <= paths
    assert {"/api/agents", "/api/tasks", "/api/tasks/{task_id}", "/api/conversation"} <= paths


# -- pages -----------------------------------------------------------------


async def test_index_and_static_assets_are_served(client):
    index = await client.get("/")
    assert index.status_code == 200
    assert "<title>Buddy</title>" in index.text
    # xterm.js is vendored, not fetched from a CDN: the dashboard has to work
    # on a machine with no network.
    for asset in ("/static/app.js", "/static/style.css", "/static/vendor/xterm.js"):
        response = await client.get(asset)
        assert response.status_code == 200, asset
        assert response.content


def test_the_page_never_reaches_for_a_cdn():
    """Everything the page loads is served from this process."""
    page = (Path(__file__).parent.parent / "buddy/web/static/index.html").read_text()
    referenced = re.findall(r'(?:src|href)="([^"]+)"', page)
    assert referenced, "the page loads nothing at all, which cannot be right"
    external = [url for url in referenced if not url.startswith(("/static/", "#"))]
    assert not external, external
    assert any(url.endswith("vendor/xterm.js") for url in referenced)


# -- GET /api/brainstorm ---------------------------------------------------


async def test_brainstorm_reads_what_the_session_wrote(client, config):
    """The dashboard is a separate reader of the file the session writes, so
    what it shows is what the brain has, whichever process asks."""
    from buddy.ideation import Brainstorm, Draft

    assert (await client.get("/api/brainstorm")).json() == {"active": False, "drafts": []}

    storm = Brainstorm.load(config.home)
    storm.start()
    storm.upsert(Draft(id="d1", project="webapp", title="Add limits", goal="429", after=["t-0001"]))

    body = (await client.get("/api/brainstorm")).json()
    assert body["active"] is True
    assert [(d["id"], d["title"], d["after"]) for d in body["drafts"]] == [
        ("d1", "Add limits", ["t-0001"])
    ]


async def test_a_brainstorm_change_reaches_the_event_socket_as_its_own_type():
    from buddy.models import BrainstormChanged
    from buddy.web.events import event_payload

    payload = event_payload(BrainstormChanged(active=True, drafts=2))
    assert payload["type"] == "BrainstormChanged"
    assert payload["active"] is True and payload["drafts"] == 2


# -- GET /api/agents -------------------------------------------------------


async def test_agents_lists_the_running_ones_most_urgent_first(client, store, config):
    task = make_task(store, priority=1, title="Urgent thing", agent="firefighter")
    store.create_run(make_run(config, task))
    store.save_agent(
        Agent(
            name="firefighter",
            task_id=task.id,
            run_attempt=1,
            status=AgentStatus.RUNNING,
            harness="claude_code",
            priority=1,
            started_at=utcnow() - timedelta(minutes=3),
            last_output_at=utcnow() - timedelta(seconds=20),
            last_output="line one\nline two\nline three\nline four",
        )
    )
    other = make_task(store, priority=4, title="Later", agent="tidier")
    store.save_agent(Agent("tidier", other.id, 1, priority=4, started_at=utcnow()))

    body = (await client.get("/api/agents")).json()
    assert [a["name"] for a in body["agents"]] == ["firefighter", "tidier"]
    assert body["max_concurrent"] is None, "no limit unless one is set"

    firefighter = body["agents"][0]
    assert firefighter["status"] == "running"
    assert firefighter["title"] == "Urgent thing"
    assert firefighter["branch"] == branch_name(task)
    assert firefighter["project"] == "webapp"
    assert 170 <= firefighter["age_seconds"] <= 200
    assert firefighter["last_output_ago_seconds"] >= 19
    # Three lines, the "last-output" summary on the card.
    assert firefighter["tail"].splitlines() == ["line two", "line three", "line four"]


async def test_with_nothing_running_there_are_no_agents(client):
    assert (await client.get("/api/agents")).json()["agents"] == []


# -- GET /api/tasks --------------------------------------------------------


async def test_tasks_filtered_by_state_and_ordered_as_the_queue_runs(client, store):
    low = make_task(store, priority=5, title="Whenever")
    high = make_task(store, priority=1, title="Urgent")
    done = make_task(store, priority=3, title="Finished")
    store.set_task_state(done.id, TaskState.DONE)

    queued = (await client.get("/api/tasks?state=queued")).json()["tasks"]
    assert [t["id"] for t in queued] == [high.id, low.id]
    assert [t["queue_position"] for t in queued] == [0, 1]
    assert all(t["state"] == "queued" for t in queued)

    finished = (await client.get("/api/tasks?state=done")).json()["tasks"]
    assert [t["id"] for t in finished] == [done.id]


async def test_a_dependency_becomes_a_badge_not_a_position(client, store):
    first = make_task(store, title="First")
    second = make_task(store, title="Second", depends_on=[first.id], priority=1)

    queued = (await client.get("/api/tasks?state=queued")).json()["tasks"]
    rows = {task["id"]: task for task in queued}
    assert rows[second.id]["depends_on"] == [first.id]
    assert first.id in rows[second.id]["waiting_for"]
    # Blocked work has no place in the queue order, so no number.
    assert rows[second.id]["queue_position"] is None
    assert rows[first.id]["queue_position"] == 0


async def test_unknown_state_is_a_sentence_not_a_stack_trace(client):
    response = await client.get("/api/tasks?state=nonsense")
    assert response.status_code == 400
    assert "queued" in response.json()["detail"]


async def test_tasks_can_be_filtered_by_project(client, store):
    make_task(store, title="Webapp work")
    body = (await client.get("/api/tasks?project=webapp")).json()
    assert [t["title"] for t in body["tasks"]] == ["Webapp work"]
    assert (await client.get("/api/tasks?project=nothing")).json()["tasks"] == []


# -- GET /api/tasks/{id} ---------------------------------------------------


async def test_task_detail_carries_spec_runs_result_and_diff(client, store, config, workspace):
    task = make_task(store, title="Add a greeting")
    checkout = await workspace.create(task)
    (checkout.path / "greeting.txt").write_text("hello\n")
    git(checkout.path, "add", "-A")
    git(checkout.path, "commit", "-q", "-m", "add greeting")

    run = make_run(config, task)
    store.create_run(run)
    run.ended_at = utcnow()
    run.exit_code = 0
    run.outcome = RunOutcome.DONE
    run.wip_commit = "abc1234"
    store.save_run(run)
    store.set_task_state(task.id, TaskState.DONE)
    config.paths.result_file(task.id).write_text(
        json.dumps({"task_id": task.id, "ok": True, "summary": "done"})
    )

    body = (await client.get(f"/api/tasks/{task.id}")).json()
    assert body["task"]["title"] == "Add a greeting"
    assert body["task"]["brief"].startswith("# Goal")
    assert body["task"]["state"] == "done"
    assert body["branch"] == branch_name(task)
    assert body["result"]["summary"] == "done"
    assert body["diff_error"] is None
    assert "greeting.txt" in body["diff_stat"]

    (attempt,) = body["runs"]
    assert attempt["attempt"] == 1
    assert attempt["outcome"] == "done"
    assert attempt["exit_code"] == 0
    assert attempt["wip_commit"] == "abc1234"
    assert attempt["log_exists"] is True


async def test_task_with_no_branch_reports_why_rather_than_failing(client, store):
    task = make_task(store, title="Never ran")
    body = (await client.get(f"/api/tasks/{task.id}")).json()
    assert body["runs"] == []
    assert body["result"] is None
    assert body["diff_stat"] == ""
    assert body["diff_error"]  # a git message, not a 500


async def test_a_task_names_its_agent_running_or_not(client, store, config):
    task = make_task(store, state=TaskState.DONE, agent="scout")
    store.create_run(make_run(config, task))

    body = (await client.get(f"/api/tasks/{task.id}")).json()
    assert body["task"]["agent"] == "scout"
    assert body["task"]["running"] is False
    assert body["runs"][0]["agent"] == "scout"


async def test_unknown_task_is_a_404(client):
    response = await client.get("/api/tasks/t-9999")
    assert response.status_code == 404


# -- GET /api/conversation -------------------------------------------------


async def test_conversation_backfills_oldest_first_and_pages_backwards(client, store):
    for index in range(10):
        store.log_turn("user" if index % 2 == 0 else "buddy", f"turn {index}")

    page = (await client.get("/api/conversation?limit=4")).json()
    assert [t["text"] for t in page["turns"]] == ["turn 6", "turn 7", "turn 8", "turn 9"]
    assert page["has_more"] is True

    older = (await client.get(f"/api/conversation?before={page['turns'][0]['id']}&limit=4")).json()
    assert [t["text"] for t in older["turns"]] == ["turn 2", "turn 3", "turn 4", "turn 5"]

    first_id = older["turns"][0]["id"]
    oldest = (await client.get(f"/api/conversation?before={first_id}&limit=4")).json()
    assert [t["text"] for t in oldest["turns"]] == ["turn 0", "turn 1"]
    assert oldest["has_more"] is False


# -- WS /ws/events ---------------------------------------------------------


async def test_events_reach_an_open_socket(server: Dashboard, hub: Hub):
    async with websockets.connect(f"ws://127.0.0.1:{server.port}/ws/events") as ws:
        await wait_for(lambda: hub.subscribers(EVENTS) == 1)
        hub.publish_events([TaskStarted(task_id="t-0001", agent="scout", attempt=2)])
        payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))

    assert payload["type"] == "TaskStarted"
    assert payload["task_id"] == "t-0001"
    assert payload["agent"] == "scout"
    assert payload["attempt"] == 2
    assert payload["at"].endswith("+00:00")


async def test_a_closed_tab_releases_its_subscription(server: Dashboard, hub: Hub):
    """A viewer that goes away must not leave a queue behind filling forever."""
    async with websockets.connect(f"ws://127.0.0.1:{server.port}/ws/events"):
        await wait_for(lambda: hub.subscribers(EVENTS) == 1)
    await wait_for(lambda: hub.subscribers(EVENTS) == 0)


async def test_conversation_turns_are_pushed_as_the_brain_writes_them(
    server: Dashboard, store: Store, hub: Hub
):
    store.on_turn(hub.publish_turn)
    async with websockets.connect(f"ws://127.0.0.1:{server.port}/ws/conversation") as ws:
        await wait_for(lambda: hub.subscribers("conversation") == 1)
        store.log_turn("user", "spawn scout on webapp")
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        store.log_turn("buddy", "scout is on it")
        second = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))

    assert (first["speaker"], first["text"]) == ("user", "spawn scout on webapp")
    assert (second["speaker"], second["text"]) == ("buddy", "scout is on it")
    assert second["id"] > first["id"]


# -- WS /ws/logs/{task_id}/{attempt} --------------------------------------


async def test_log_socket_sends_what_exists_then_follows(server, store, config):
    task = make_task(store)
    run = make_run(config, task)
    store.create_run(run)
    # Raw, escapes and all: that is why the browser renders it in xterm.js.
    run.log_path.write_text("\x1b[32malready written\x1b[0m\r\n")

    url = f"ws://127.0.0.1:{server.port}/ws/logs/{task.id}/1"
    async with websockets.connect(url) as ws:
        backfill = await asyncio.wait_for(ws.recv(), timeout=5)
        assert backfill == "\x1b[32malready written\x1b[0m\r\n"

        with run.log_path.open("a") as handle:
            handle.write("and then this\r\n")
        assert await asyncio.wait_for(ws.recv(), timeout=5) == "and then this\r\n"


async def test_log_socket_says_why_when_an_attempt_never_ran(server, store):
    task = make_task(store)
    url = f"ws://127.0.0.1:{server.port}/ws/logs/{task.id}/7"
    async with websockets.connect(url) as ws:
        with pytest.raises(websockets.exceptions.ConnectionClosed) as caught:
            await asyncio.wait_for(ws.recv(), timeout=5)
    assert caught.value.rcvd.code == 1008
    assert caught.value.rcvd.reason == f"no attempt 7 of {task.id}"


async def test_tailing_a_log_never_writes_to_it(tmp_path: Path):
    """The dashboard reads; it does not touch what the task is writing."""
    log = tmp_path / "attempt-1.log"
    log.write_text("one\n")
    before = log.stat().st_mtime_ns

    tailer = tail_log(log, poll=0.01, initial_bytes=1024)
    assert await asyncio.wait_for(tailer.__anext__(), timeout=5) == "one\n"
    await tailer.aclose()

    assert log.stat().st_mtime_ns == before
    assert log.read_text() == "one\n"


async def test_a_truncated_log_resets_the_terminal(tmp_path: Path):
    log = tmp_path / "attempt-1.log"
    log.write_text("first run output\n")
    tailer = tail_log(log, poll=0.01, initial_bytes=1024)
    assert await asyncio.wait_for(tailer.__anext__(), timeout=5) == "first run output\n"

    log.write_text("x\n")  # shorter than what was already sent
    chunks = [await asyncio.wait_for(tailer.__anext__(), timeout=5) for _ in range(2)]
    await tailer.aclose()
    from buddy.web.server import LOG_ROTATED_NOTE

    assert chunks == ["\x1bc" + LOG_ROTATED_NOTE, "x\n"]


async def test_a_huge_log_is_tailed_not_replayed(tmp_path: Path):
    log = tmp_path / "attempt-1.log"
    log.write_text("A" * 5000 + "the end\n")

    tailer = tail_log(log, poll=0.01, initial_bytes=1024)
    notice = await asyncio.wait_for(tailer.__anext__(), timeout=5)
    body = await asyncio.wait_for(tailer.__anext__(), timeout=5)
    await tailer.aclose()

    assert "skipped" in notice
    assert body.endswith("the end\n")
    assert len(body) <= 1024


async def test_a_log_that_does_not_exist_yet_is_waited_for(tmp_path: Path):
    log = tmp_path / "attempt-1.log"
    tailer = tail_log(log, poll=0.01, initial_bytes=1024)
    pending = asyncio.ensure_future(tailer.__anext__())
    await asyncio.sleep(0.05)
    assert not pending.done()

    log.write_text("here at last\n")
    assert await asyncio.wait_for(pending, timeout=5) == "here at last\n"
    await tailer.aclose()


# -- the hub ---------------------------------------------------------------


async def test_a_stalled_subscriber_drops_the_oldest_and_never_blocks():
    """A sleeping laptop with the tab open cannot back up onto the manager."""
    hub = Hub(maxsize=3)
    with hub.subscribe(EVENTS) as queue:
        for index in range(6):
            hub.publish(EVENTS, {"n": index})
        assert queue.qsize() == 3
        assert [queue.get_nowait()["n"] for _ in range(3)] == [3, 4, 5]
    assert hub.dropped == 3


def test_publishing_with_nobody_listening_is_free():
    hub = Hub()
    hub.publish_events([TaskStarted(task_id="t-1", agent="scout")])
    assert hub.subscribers(EVENTS) == 0


def test_event_payload_keeps_every_field_and_names_the_type():
    event = TaskFinished(
        task_id="t-0007",
        agent="scout",
        attempt=2,
        outcome=RunOutcome.TIMEOUT,
        exit_code=124,
        branch="buddy/t-0007-thing",
        summary="ran out of time",
    )
    payload = event_payload(event)
    assert payload["type"] == "TaskFinished"
    assert payload["outcome"] == "timeout"  # the enum's wire value, not its repr
    assert payload["exit_code"] == 124
    assert json.dumps(payload)  # every value survives serialization


def test_jsonable_handles_the_types_events_actually_carry():
    assert jsonable(Path("/tmp/x.log")) == "/tmp/x.log"
    assert jsonable(AgentStatus.WAITING_INPUT) == "waiting_input"
    assert jsonable([RunOutcome.DONE, None]) == ["done", None]
    assert jsonable({"outcome": RunOutcome.ERROR}) == {"outcome": "error"}


# -- lifecycle -------------------------------------------------------------


async def test_a_taken_port_is_explained_rather_than_fatal(
    config, store, manager, workspace, hub, server: Dashboard
):
    """The dashboard is a viewer. It failing must cost nothing else."""
    messages: list[str] = []
    second = await start_dashboard(
        config=config,
        store=store,
        manager=manager,
        workspace=workspace,
        hub=hub,
        port=server.port,
        on_error=messages.append,
    )
    assert second is None
    assert "--no-web" in messages[0]
    assert str(server.port) in messages[0]

    # The one that was already there is untouched.
    async with httpx.AsyncClient(base_url=server.url, timeout=10) as client:
        assert (await client.get("/api/agents")).status_code == 200


async def test_stopping_twice_is_harmless(app):
    dashboard = Dashboard(app, port=0)
    await dashboard.start()
    await dashboard.stop()
    await dashboard.stop()


async def test_the_dashboard_does_not_steal_the_sessions_signal_handlers(app):
    """Buddy owns Ctrl-C: it has tasks in tmux to decide about.

    uvicorn installs its own SIGINT handler by default, which would make the
    dashboard - the one component whose failure should cost nothing - the
    thing that decides what a Ctrl-C means.
    """
    before = signal.getsignal(signal.SIGINT)
    dashboard = Dashboard(app, port=0)
    await dashboard.start()
    try:
        assert signal.getsignal(signal.SIGINT) is before
    finally:
        await dashboard.stop()
    assert signal.getsignal(signal.SIGINT) is before


# -- who may open a websocket --------------------------------------


async def test_a_page_from_another_origin_cannot_read_the_conversation(server):
    """Binding to 127.0.0.1 keeps other machines out and the same-origin
    policy keeps other pages off the JSON routes - but browsers do not apply
    it to websocket handshakes. Without this check, any page open in the
    operator's browser could read every turn, every event, and any task's
    live output, with ids as guessable as `t-0001`.
    """
    for path in ("/ws/conversation", "/ws/events", "/ws/logs/t-0001/1"):
        url = f"ws://127.0.0.1:{server.port}{path}"
        with pytest.raises(websockets.exceptions.InvalidStatus) as refused:
            await websockets.connect(url, additional_headers={"Origin": "https://evil.example"})
        # Refused at the handshake, so the socket is never accepted and not one
        # byte is streamed - a better answer than accepting and then closing.
        assert refused.value.response.status_code == 403, path


async def test_a_rebound_hostname_reads_nothing(server):
    """DNS rebinding: a page re-points its own hostname at 127.0.0.1, and the
    browser then treats every route here as same-origin with it. The one
    thing it cannot change is the Host it sends. Measured before the check:
    `Host: attacker.example` read `/api/conversation` with a 200."""
    rebound = {"Host": f"attacker.example:{server.port}"}
    async with httpx.AsyncClient(base_url=server.url, timeout=10) as client:
        for path in ("/api/conversation", "/api/agents", "/api/tasks", "/"):
            assert (await client.get(path, headers=rebound)).status_code == 400, path
        assert (await client.get("/api/agents")).status_code == 200
        localhost = {"Host": f"localhost:{server.port}"}
        assert (await client.get("/api/agents", headers=localhost)).status_code == 200

    url = f"ws://127.0.0.1:{server.port}/ws/conversation"
    with pytest.raises(websockets.exceptions.InvalidStatus) as refused:
        await websockets.connect(url, additional_headers={"Host": rebound["Host"]})
    assert refused.value.response.status_code == 400


async def test_the_dashboards_own_page_still_connects(server, hub):
    origin = f"http://127.0.0.1:{server.port}"
    url = f"ws://127.0.0.1:{server.port}/ws/events"
    async with websockets.connect(url, additional_headers={"Origin": origin}) as ws:
        await wait_for(lambda: hub.subscribers(EVENTS) == 1)
        hub.publish_events([TaskStarted(task_id="t-0001", agent="scout")])
        assert json.loads(await asyncio.wait_for(ws.recv(), timeout=5))["task_id"] == "t-0001"


async def test_localhost_is_the_same_origin_as_127_0_0_1(server, hub):
    """People type one and browsers send the other."""
    url = f"ws://127.0.0.1:{server.port}/ws/events"
    origin = f"http://localhost:{server.port}"
    async with websockets.connect(url, additional_headers={"Origin": origin}):
        await wait_for(lambda: hub.subscribers(EVENTS) == 1)


async def test_a_native_client_with_no_origin_is_allowed(server, hub):
    """`websockets`, `wscat`, a test: already on this machine, and with
    simpler ways in than a websocket."""
    async with websockets.connect(f"ws://127.0.0.1:{server.port}/ws/events"):
        await wait_for(lambda: hub.subscribers(EVENTS) == 1)
