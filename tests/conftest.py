"""Shared fixtures.

The tmux-facing tests each get their own tmux server, so they can never
disturb the user's real `buddy` session, and the socket file is removed
afterwards: `kill-server` leaves it behind, and a suite that grows a new file
in /tmp on every run is a leak whether or not anything is still listening.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from buddy.tmux_runner import TmuxRunner


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
