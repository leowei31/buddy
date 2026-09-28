"""The always-on-top glance panel.

The dashboard is where you *look* at Buddy; this is what stays on screen
while you look at something else. A small frameless window, above other
windows, showing the seven slots and what each agent is doing right now.

It is a separate process on purpose, and that is not an implementation
detail:

- macOS requires a Cocoa event loop on the main thread, and Buddy's main
  thread already belongs to asyncio. A window inside the session would mean
  inverting the whole program around Tk or Cocoa.
- It reads the dashboard's own API, so it is read-only exactly as that is: it is
  another read-only viewer, and closing, opening or dragging it cannot affect
  a running task.
- It can be started, killed and restarted without touching the session.

Reading is over HTTP; **typing is not**. A turn typed here goes down the
unix socket in `buddy.control` to the running session, which puts it on the
same queue voice uses, so there is still exactly one thing calling the
brain. The dashboard stays read-only and the page itself never touches the
socket - the page asks this process, and this process holds it.

If no session is listening, the composer is simply not shown: the panel is
still a perfectly good read-only viewer.
"""

from __future__ import annotations

import json
import os
import signal
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Small enough to leave in a corner, wide enough for a task title. The
#: height is a starting point only: the page measures itself and asks the
#: window to fit, because the number of busy slots changes all day.
DEFAULT_WIDTH = 330
DEFAULT_HEIGHT = 240
MAX_HEIGHT = 620

#: What it shrinks to when folded: the header alone.
FOLDED_HEIGHT = 38


class OverlayNotInstalled(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "the overlay needs the 'pywebview' package. Install it with: "
            "uv tool install 'buddy-orchestrator[overlay]' (or `uv sync --extra overlay` "
            "from a source checkout)."
        )


@dataclass(frozen=True)
class Corner:
    """Where the panel sits. Screen coordinates, top-left origin."""

    x: int
    y: int


def top_right(width: int = DEFAULT_WIDTH, margin: int = 24) -> Corner:
    """Out of the way of almost everything, on any screen size."""
    try:
        from AppKit import NSScreen

        frame = NSScreen.mainScreen().visibleFrame()
        return Corner(int(frame.origin.x + frame.size.width - width - margin), margin + 8)
    except Exception:  # noqa: BLE001 - a sensible guess beats not opening
        return Corner(1000, 40)


def dashboard_is_up(url: str, *, timeout: float = 1.5) -> bool:
    """Whether a session is running with its dashboard on."""
    try:
        with urllib.request.urlopen(f"{url}/api/slots", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def summarise(url: str, *, timeout: float = 2.0) -> str:
    """One line about the slots, for a terminal that cannot draw a window."""
    try:
        with urllib.request.urlopen(f"{url}/api/slots", timeout=timeout) as response:
            slots = json.load(response)["slots"]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        return f"Buddy is not reachable at {url} ({type(exc).__name__})"
    busy = [s for s in slots if s["occupied"]]
    attention = [s["name"] for s in slots if s["status"] in ("waiting_input", "stalled")]
    line = f"{len(busy)} running"
    if attention:
        verb = "needs" if len(attention) == 1 else "need"
        line += f", and {', '.join(attention)} {verb} you"
    return line


# --------------------------------------------------------------------------
# Only one at a time
# --------------------------------------------------------------------------


def pid_file(home: Path) -> Path:
    return home / "overlay.pid"


def running_pid(home: Path) -> int | None:
    """The live overlay's pid, or None. A stale file is cleaned up."""
    path = pid_file(home)
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)  # signal 0 asks "is it there?" without touching it
    except (ProcessLookupError, ValueError):
        path.unlink(missing_ok=True)
        return None
    except PermissionError:
        return pid
    return pid


def stop(home: Path, *, timeout: float = 3.0) -> bool:
    """Close a running overlay. Returns whether there was one."""
    pid = running_pid(home)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pid_file(home).unlink(missing_ok=True)
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if running_pid(home) is None:
            return True
        time.sleep(0.05)
    with _suppressed():
        os.kill(pid, signal.SIGKILL)
    pid_file(home).unlink(missing_ok=True)
    return True


class _suppressed:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------


class _Api:
    """What the page may ask this process to do.

    Three window things and one conversation thing. `say` is the only one
    that leaves the process, and it leaves over the unix socket, never over
    HTTP: the dashboard's routes stay read-only.
    """

    def __init__(self, home: Path | None = None) -> None:
        #: pywebview's Window, once `show` has made it.
        self.window: Any = None
        self.height = DEFAULT_HEIGHT
        self.home = home or Path.home() / ".buddy"

    def can_say(self) -> bool:
        """Whether a session is listening, so the page can hide the composer."""
        try:
            from buddy.control import is_listening
        except ImportError:
            return False
        with _suppressed():
            return is_listening(self.home)
        return False

    def say(self, text: str) -> dict[str, object]:
        """Send one turn to the session and wait for its reply.

        pywebview calls js_api off the Cocoa thread, so blocking here blocks
        the call, not the window. Every failure comes back as a value: an
        exception raised in here reaches the page as an unhandled rejection
        with nothing useful in it.
        """
        text = str(text or "").strip()
        if not text:
            return {"ok": False, "error": "nothing to say"}
        try:
            from buddy.control import NotListening, say_sync
        except ImportError:
            return {"ok": False, "error": "control channel unavailable"}
        try:
            return {"ok": True, "text": say_sync(self.home, text)}
        except NotListening as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # a dead session must not take the panel with it
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def set_folded(self, folded: bool) -> None:
        if self.window is None:
            return
        with _suppressed():
            self.window.resize(DEFAULT_WIDTH, FOLDED_HEIGHT if folded else self.height)

    def set_height(self, height: int) -> None:
        """The page has measured itself; make the window that tall.

        Clamped before anything else: a bad measurement must not produce a
        window taller than the screen or shorter than its own header, and
        that has to hold whether or not a window is attached yet.
        """
        self.height = max(FOLDED_HEIGHT, min(int(height), MAX_HEIGHT))
        if self.window is None:
            return
        with _suppressed():
            self.window.resize(DEFAULT_WIDTH, self.height)

    def close(self) -> None:
        if self.window is not None:
            with _suppressed():
                self.window.destroy()


def show(url: str, home: Path, *, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> int:
    """Open the panel and block until it is closed.

    Runs the Cocoa loop on this process's main thread, which is why this is
    its own process rather than a thread inside the session.
    """
    try:
        import webview
    except ImportError as exc:
        raise OverlayNotInstalled() from exc

    corner = top_right(width)
    api = _Api(home)
    window = webview.create_window(
        "Buddy",
        f"{url}/overlay",
        width=width,
        height=height,
        x=corner.x,
        y=corner.y,
        frameless=True,
        easy_drag=False,  # the page marks its own drag region, so text stays selectable
        on_top=True,
        resizable=False,
        transparent=True,
        background_color="#0b0e14",
        js_api=api,
    )
    api.window = window

    path = pid_file(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n")

    def leave(*_args: object) -> None:
        api.close()

    for received in (signal.SIGTERM, signal.SIGINT):
        with _suppressed():
            signal.signal(received, leave)

    try:
        webview.start()
    finally:
        path.unlink(missing_ok=True)
    return 0
