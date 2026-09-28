"""Shared fixtures.

The tmux-facing tests each get their own tmux server, so they can never
disturb the user's real `buddy` session, and the socket file is removed
afterwards: `kill-server` leaves it behind, and a suite that grows a new file
in /tmp on every run is a leak whether or not anything is still listening.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

from buddy.tmux_runner import TmuxRunner

#: How much of a failure's traceback an annotation carries: the end, where
#: the assertion and the values it compared are.
ANNOTATION_LINES = 40


def _escape(text: str, *, property: bool = False) -> str:
    """GitHub's workflow-command escaping, for a message or a property."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if property else text


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """On GitHub Actions, every failure becomes an annotation on the commit.

    A run's logs are readable only when signed in to GitHub; its annotations
    are public, and shown next to the commit. So which test failed, and why,
    is visible to anyone looking at a red check - which is the first thing a
    contributor, or the maintainer away from a browser session, needs.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true" or not report.failed:
        return
    path, line, _ = report.location
    tail = "\n".join(report.longreprtext.splitlines()[-ANNOTATION_LINES:])
    sys.__stdout__.write(
        f"\n::error file={_escape(path, property=True)},line={(line or 0) + 1},"
        f"title={_escape(f'{report.when} failed: {report.nodeid}', property=True)}"
        f"::{_escape(tail)}\n"
    )
    sys.__stdout__.flush()


def _socket_dir() -> Path:
    return Path(os.environ.get("TMUX_TMPDIR", "/tmp")) / f"tmux-{os.getuid()}"


@pytest.fixture
async def tmux_socket():
    """A private tmux server name, torn down and unlinked afterwards."""
    name = f"buddytest-{uuid.uuid4().hex[:8]}"
    try:
        yield name
    finally:
        await TmuxRunner(socket_name=name)._call("kill-server")
        (_socket_dir() / name).unlink(missing_ok=True)
