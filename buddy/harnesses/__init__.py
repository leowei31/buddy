"""Harness adapters, and the one registry every other module reads.

Adding a harness is an adapter module and a line here. The CLI, `buddy
setup`, `buddy doctor` and the brain all look harnesses up through this
dict, so none of them can drift out of step with the others.
"""

from __future__ import annotations

from buddy.harnesses.antigravity import AntigravityAdapter
from buddy.harnesses.base import BaseAdapter, HarnessError
from buddy.harnesses.claude_code import ClaudeCodeAdapter
from buddy.harnesses.codex import CodexAdapter
from buddy.harnesses.opencode import OpenCodeAdapter

ADAPTERS: dict[str, type[BaseAdapter]] = {
    adapter.name: adapter
    for adapter in (ClaudeCodeAdapter, CodexAdapter, OpenCodeAdapter, AntigravityAdapter)
}


class UnknownHarness(HarnessError):
    """A name with no adapter behind it."""


def adapter_class(name: str) -> type[BaseAdapter]:
    try:
        return ADAPTERS[name]
    except KeyError:
        known = ", ".join(sorted(ADAPTERS))
        raise UnknownHarness(f"no adapter named {name!r} (known: {known})") from None


__all__ = ["ADAPTERS", "UnknownHarness", "adapter_class"]
