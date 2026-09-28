"""The always-on-top panel.

The window itself needs a screen, so what is tested here is everything
around it: that it reads the same read-only API the dashboard serves, that
only one runs at a time, that a stale pid file does not confuse it, and that
the page reaches for nothing outside the process serving it.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

import httpx2 as httpx
import pytest

from buddy import overlay
from buddy.config import Config
from buddy.harnesses.claude_code import ClaudeCodeAdapter
from buddy.manager import AgentManager
from buddy.models import SlotStatus
from buddy.state import Store
from buddy.web.events import Hub
from buddy.web.server import Dashboard, create_app
from buddy.workspace import Workspace


class NullRunner:
    async def ensure_session(self, slots=None) -> None: ...

    async def panes(self) -> dict:
        return {}


@pytest.fixture
def panel_home(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("BUDDY_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        f'[projects.webapp]\npath = "{tmp_path / "webapp"}"\n'
        '[harness.claude_code]\ncommand = "claude -p < {prompt_path}"\n'
    )
    return tmp_path


@pytest.fixture
async def panel_server(panel_home: Path):
    """A dashboard for the overlay to read, on a free port."""
    config = Config.load(home=panel_home)
    config.paths.ensure()
    store = Store(config.paths.db)
    store.ensure_slots()
    workspace = Workspace(config)
    app = create_app(
        config=config,
        store=store,
        manager=AgentManager(
            config,
            store,
            NullRunner(),
            workspace,
            adapter_for=lambda name: ClaudeCodeAdapter(config.harness(name)),
        ),
        workspace=workspace,
        hub=Hub(),
    )
    dashboard = Dashboard(app, port=0)
    await dashboard.start()
    try:
        yield dashboard, store
    finally:
        await dashboard.stop()
        store.close()


OVERLAY_PAGE = Path(__file__).parent.parent / "buddy/web/static/overlay.html"


# -- the page --------------------------------------------------------------


async def test_the_overlay_page_is_served(panel_server):
    dashboard, _ = panel_server
    async with httpx.AsyncClient(base_url=dashboard.url, timeout=10) as client:
        response = await client.get("/overlay")
    assert response.status_code == 200
    assert "<title>Buddy</title>" in response.text


def test_the_page_fetches_only_the_read_only_api():
    """It is a viewer, like the dashboard, and inherits that by
    reading the same routes rather than by being trusted not to."""
    page = OVERLAY_PAGE.read_text()
    fetched = set(re.findall(r'fetch\("([^"]+)"', page))
    assert fetched == {"/api/slots", "/api/tasks?state=queued&limit=100", "/api/brainstorm"}
    for verb in ("POST", "PUT", "DELETE", "PATCH"):
        assert verb not in page


def test_the_page_reaches_for_nothing_off_the_machine():
    page = OVERLAY_PAGE.read_text()
    assert "http://" not in page.replace("http://127.0.0.1", "")
    assert "https://" not in page


def test_the_page_never_builds_markup_from_data():
    """Task titles and agent output land in this page too."""
    assert "innerHTML" not in OVERLAY_PAGE.read_text()


# -- the summary -----------------------------------------------------------


async def test_the_summary_reads_the_live_slots(panel_server):
    dashboard, store = panel_server
    for name, status in (("Monday", SlotStatus.RUNNING), ("Friday", SlotStatus.WAITING_INPUT)):
        slot = next(s for s in store.load_slots() if s.name == name)
        slot.status, slot.task_id = status, "t-0001"
        store.save_slot(slot)

    line = await asyncio.to_thread(overlay.summarise, dashboard.url)
    assert "2 running" in line
    # Singular, because one slot is not "need you".
    assert "Friday needs you" in line


async def test_an_unreachable_buddy_is_a_sentence():
    line = await asyncio.to_thread(overlay.summarise, "http://127.0.0.1:1", timeout=0.3)
    assert "not reachable" in line


async def test_dashboard_is_up_answers_honestly(panel_server):
    dashboard, _ = panel_server
    assert await asyncio.to_thread(overlay.dashboard_is_up, dashboard.url)
    assert not await asyncio.to_thread(overlay.dashboard_is_up, "http://127.0.0.1:1", timeout=0.3)


# -- one at a time ---------------------------------------------------------


def test_no_pid_file_means_nothing_is_running(tmp_path):
    assert overlay.running_pid(tmp_path) is None
    assert overlay.stop(tmp_path) is False


def test_a_stale_pid_file_is_cleaned_up_rather_than_believed(tmp_path):
    """A machine that was rebooted, or an overlay that was killed -9."""
    dead = 999_999
    overlay.pid_file(tmp_path).write_text(f"{dead}\n")
    assert overlay.running_pid(tmp_path) is None
    assert not overlay.pid_file(tmp_path).exists()


def test_a_corrupt_pid_file_is_not_believed_either(tmp_path):
    overlay.pid_file(tmp_path).write_text("not a pid\n")
    assert overlay.running_pid(tmp_path) is None


def test_a_live_pid_is_reported(tmp_path):
    overlay.pid_file(tmp_path).write_text(f"{os.getpid()}\n")
    assert overlay.running_pid(tmp_path) == os.getpid()


def test_the_panel_is_placed_somewhere_on_screen():
    corner = overlay.top_right()
    assert corner.x > 0 and corner.y >= 0


def test_the_window_height_is_clamped():
    """The page measures itself; a bad measurement must not produce a window
    taller than the screen or shorter than its own header."""
    api = overlay._Api()
    api.set_height(5)
    assert api.height == overlay.FOLDED_HEIGHT
    api.set_height(99_999)
    assert api.height == overlay.MAX_HEIGHT
    api.set_height(310)
    assert api.height == 310


# -- typing at it ----------------------------------------------------------


@pytest.fixture
def short_home():
    """A unix socket must fit in `sun_path`; pytest's `tmp_path` does not."""
    import shutil
    import tempfile

    made = Path(tempfile.mkdtemp(prefix="bud-"))
    try:
        yield made
    finally:
        shutil.rmtree(made, ignore_errors=True)


def test_the_page_hides_the_composer_until_a_session_is_listening():
    """A text box that silently does nothing is worse than no text box."""
    page = OVERLAY_PAGE.read_text()
    assert "#composer { display: none; }" in page
    assert "body.can-say #composer { display: block; }" in page
    assert "can_say()" in page


def test_the_page_still_talks_only_to_its_own_process():
    """The composer must not become a second way to reach the dashboard.

    Read-only is kept structurally: typing goes through `pywebview.api`, which is
    this window's own Python, not through a route.
    """
    page = OVERLAY_PAGE.read_text()
    assert "pywebview.api.say" in page
    fetched = set(re.findall(r'fetch\("([^"]+)"', page))
    assert fetched == {"/api/slots", "/api/tasks?state=queued&limit=100", "/api/brainstorm"}


def test_saying_nothing_is_refused_without_opening_anything(tmp_path):
    assert overlay._Api(tmp_path).say("   ") == {"ok": False, "error": "nothing to say"}


def test_no_session_listening_is_a_value_not_an_exception(tmp_path):
    """An exception here reaches the page as an unhandled rejection with
    nothing useful in it, so every failure comes back as a value."""
    api = overlay._Api(tmp_path)
    assert api.can_say() is False
    result = api.say("what have you got?")
    assert result["ok"] is False
    assert result["error"]


async def test_a_socket_path_too_long_says_what_to_change(tmp_path):
    """`sun_path` overflows as a bare "AF_UNIX path too long", which names
    nothing you could act on."""
    from buddy.control import ControlServer

    deep = tmp_path / ("d" * 60) / ("e" * 60)
    with pytest.raises(OSError, match="BUDDY_HOME"):
        await ControlServer(deep).start()


async def test_a_turn_typed_at_the_panel_reaches_the_session(short_home):
    """The seam that matters: page -> this process -> socket -> session.

    The session side is the real `ControlServer`, drained the way `cli.py`
    drains it, so this covers the whole path rather than one end of it.
    """
    from buddy.control import ControlServer

    tmp_path = short_home
    server = ControlServer(tmp_path)
    await server.start()

    async def be_the_session() -> None:
        utterance = await server.heard.get()
        utterance.finish(f"heard: {utterance.text}")

    drainer = asyncio.create_task(be_the_session())
    api = overlay._Api(tmp_path)
    try:
        assert await asyncio.to_thread(api.can_say) is True
        result = await asyncio.to_thread(api.say, "what have you got?")
        assert result == {"ok": True, "text": "heard: what have you got?"}
    finally:
        drainer.cancel()
        await server.stop()


# -- the command -----------------------------------------------------------


def test_overlay_says_what_to_do_when_nothing_is_running(panel_home, monkeypatch):
    from typer.testing import CliRunner

    from buddy.cli import app

    monkeypatch.setattr(overlay, "dashboard_is_up", lambda *a, **k: False)
    result = CliRunner().invoke(app, ["overlay"])
    assert result.exit_code == 1
    assert "start a session first" in result.stdout


def test_overlay_status_prints_instead_of_drawing(panel_home, monkeypatch):
    from typer.testing import CliRunner

    from buddy.cli import app

    monkeypatch.setattr(overlay, "summarise", lambda *a, **k: "4 running")
    result = CliRunner().invoke(app, ["overlay", "--status"])
    assert result.exit_code == 0
    assert "4 running" in result.stdout
