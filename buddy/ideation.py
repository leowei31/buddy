"""Brainstorming before anything starts.

Buddy's default is to act: when an intent is clear, it writes a brief and
spawns it. That is the wrong default while an idea is still being shaped - the
user wants to think out loud, weigh options, read the code together, and
only then hand the work over.

So brainstorming is a *mode*, and the guarantee that nothing starts during it
is structural rather than a request in a prompt:

* The brain is not offered the tools that start or change work. A model
  cannot call a tool it was never given, and a hallucinated call to one is
  refused at dispatch anyway.
* Ideas become **drafts** - the same fields as a spawn, recorded and revisable,
  but inert.
* Only the user ends it. Typing `/go` launches the drafts; saying "go ahead"
  lets the brain ask, and that one question is always asked, whatever
  `trust_mode` says, because opting into brainstorming *is* asking for a
  checkpoint before hand-off.
* What launches is exactly the drafts the user reviewed - spawned by Buddy
  from their stored fields, not re-written by the model on the way out.

Drafts live in `~/.buddy/brainstorm.json`, so a brainstorm survives a restart.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Tools that start or change work. Withheld for the whole brainstorm.
ACTING_TOOLS = frozenset(
    {
        "spawn_agent",
        "kill_agent",
        "reprioritize",
        "accept_preemption",
        "decline_preemption",
        "propose_merge",
        "discard_task",
        "resolve_conflict",
    }
)

#: A draft's id: `d` and a number, so it cannot be mistaken for a task id.
DRAFT_ID = re.compile(r"^d\d+$")

FILENAME = "brainstorm.json"


@dataclass
class Draft:
    """A task that has been thought through and not started."""

    id: str
    project: str
    title: str
    goal: str
    context: str = ""
    constraints: str = ""
    definition_of_done: str = ""
    harness: str = ""
    model: str = ""
    priority: int = 0
    #: Draft ids (`d1`) or existing task ids (`t-0042`) this one builds on.
    after: list[str] = field(default_factory=list)
    #: What its agent will be called; empty to name it from the title.
    name: str = ""

    def line(self) -> str:
        named = f"{self.name}: " if self.name else ""
        parts = [f"{self.id}  {named}{self.title}  ({self.project}"]
        if self.harness:
            parts.append(f", {self.harness}")
        if self.priority:
            parts.append(f", p{self.priority}")
        if self.after:
            parts.append(f", after {', '.join(self.after)}")
        return "".join(parts) + ")"


@dataclass
class Brainstorm:
    """Whether a brainstorm is on, and what it has produced so far."""

    path: Path
    active: bool = False
    drafts: dict[str, Draft] = field(default_factory=dict)
    #: Told after every save, so the session can let the dashboard know.
    on_change: Callable[[Brainstorm], None] | None = field(default=None, repr=False)

    @classmethod
    def load(
        cls, home: Path, *, on_change: Callable[[Brainstorm], None] | None = None
    ) -> Brainstorm:
        path = home / FILENAME
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls(path=path, on_change=on_change)
        if not isinstance(data, dict):
            return cls(path=path, on_change=on_change)
        drafts: dict[str, Draft] = {}
        for raw in data.get("drafts") or []:
            try:
                draft = Draft(**{k: v for k, v in raw.items() if k in Draft.__annotations__})
            except TypeError:
                continue  # a draft from a different version; not worth failing a session
            drafts[draft.id] = draft
        return cls(path=path, active=bool(data.get("active")), drafts=drafts, on_change=on_change)

    def save(self) -> None:
        """Atomically: a crash mid-write must not cost the brainstorm."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"active": self.active, "drafts": [asdict(d) for d in self.drafts.values()]}
        fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix=".brainstorm-")
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp, self.path)
        if self.on_change is not None:
            self.on_change(self)

    # -- the mode ---------------------------------------------------------

    def start(self) -> None:
        self.active = True
        self.save()

    def stop(self) -> None:
        """Leave without launching. The drafts stay for next time."""
        self.active = False
        self.save()

    # -- drafts -----------------------------------------------------------

    def next_id(self) -> str:
        numbers = [int(key[1:]) for key in self.drafts if DRAFT_ID.match(key)]
        return f"d{max(numbers, default=0) + 1}"

    def upsert(self, draft: Draft) -> Draft:
        self.drafts[draft.id] = draft
        self.save()
        return draft

    def drop(self, draft_id: str) -> Draft | None:
        dropped = self.drafts.pop(draft_id, None)
        if dropped is not None:
            # Anything that waited on it no longer can.
            for other in self.drafts.values():
                other.after = [ref for ref in other.after if ref != draft_id]
            self.save()
        return dropped

    def ordered(self) -> list[Draft]:
        """Drafts with every draft dependency before its dependents.

        A cycle cannot be launched in any order, so it is reported rather
        than broken arbitrarily.
        """
        placed: list[Draft] = []
        done: set[str] = set()
        visiting: set[str] = set()

        def visit(draft: Draft) -> None:
            if draft.id in done:
                return
            if draft.id in visiting:
                raise ValueError(f"drafts depend on each other in a circle through {draft.id}")
            visiting.add(draft.id)
            for ref in draft.after:
                if ref in self.drafts:
                    visit(self.drafts[ref])
            visiting.discard(draft.id)
            done.add(draft.id)
            placed.append(draft)

        for draft in self.drafts.values():
            visit(draft)
        return placed

    def describe(self) -> str:
        if not self.drafts:
            return "no drafts yet"
        return "\n".join(draft.line() for draft in self.drafts.values())

    def status(self) -> dict[str, Any]:
        """For the dashboard and the overlay, which only ever read it."""
        return {
            "active": self.active,
            "drafts": [asdict(draft) for draft in self.drafts.values()],
        }
