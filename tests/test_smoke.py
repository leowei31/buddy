"""The package imports, the CLI runs, and async tests work."""

import asyncio

from typer.testing import CliRunner

import buddy
from buddy.cli import app


def test_version_is_exposed():
    assert buddy.__version__


def test_cli_reports_its_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert buddy.__version__ in result.stdout


async def test_async_tests_need_no_decorator():
    """pytest-asyncio runs in auto mode, so every async test in the
    suite is written like this one."""
    await asyncio.sleep(0)
    assert asyncio.get_running_loop() is not None
