"""Shared test helpers."""

from __future__ import annotations

import asyncio
import inspect


async def wait_for(predicate, seconds: float = 15.0, interval: float = 0.05):
    """Poll until the predicate is truthy, then return its value.

    The predicate may be sync or return an awaitable, so a plain
    `lambda: some_coroutine(...)` works and is actually awaited.
    """
    async with asyncio.timeout(seconds):
        while True:
            value = predicate()
            if inspect.isawaitable(value):
                value = await value
            if value:
                return value
            await asyncio.sleep(interval)
