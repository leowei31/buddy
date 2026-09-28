"""Slot, task, run, and event types.

Pure data. Nothing here touches the filesystem, the database, tmux, or git.
The string-valued enums exist so the vocabularies stored in SQLite and sent
as JSON - task states, run outcomes - cannot drift. `StrEnum` gives identical
wire values and no divergence between `str()` and `format()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

#: The seven slots. A name is a stable handle the user says out loud,
#: never a priority rank. Ordered as the week runs, not as work is ranked.
SLOT_NAMES: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def utcnow() -> datetime:
    """Timezone-aware now. Every timestamp Buddy stores is UTC."""
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# Agent slot
# --------------------------------------------------------------------------


class SlotStatus(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"  # harness is asking a question
    STALLED = "stalled"  # no output for stall_timeout
    DONE = "done"  # exit 0, awaiting next assignment
    ERROR = "error"  # non-zero exit
    KILLED = "killed"  # by you or by preemption
    INTERRUPTED = "interrupted"  # Buddy/tmux died underneath it

    @property
    def is_occupied(self) -> bool:
        """A slot holding a live task. DONE and ERROR keep their pane and log
        for inspection but the slot is reusable, so they are not."""
        return self in _OCCUPIED_STATUSES


_OCCUPIED_STATUSES = frozenset(
    {
        SlotStatus.RUNNING,
        SlotStatus.WAITING_INPUT,
        SlotStatus.STALLED,
    }
)


@dataclass
class AgentSlot:
    name: str  # "Monday".."Sunday" - a stable handle, not a rank
    status: SlotStatus = SlotStatus.IDLE
    task_id: str | None = None
    run_attempt: int | None = None
    harness: str | None = None
    priority: int | None = None
    tmux_window: str = ""  # always == name
    started_at: datetime | None = None
    last_output_at: datetime | None = None  # last time the log file grew
    last_output: str = ""  # last ~20 lines, for status queries and voice summaries

    def __post_init__(self) -> None:
        if not self.tmux_window:
            self.tmux_window = self.name


# --------------------------------------------------------------------------
# Task spec
# --------------------------------------------------------------------------


class TaskState(StrEnum):
    """The `state` column on `tasks`."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    KILLED = "killed"
    MERGED = "merged"
    DISCARDED = "discarded"


@dataclass
class TaskSpec:
    id: str  # short, unique, filesystem- and branch-safe: "t-0142"
    title: str  # human label: "Fix onboarding flow"
    brief: str  # the prompt sent to the harness: goal, context, constraints, done
    harness: str  # "claude_code" | "opencode" | "codex" | "antigravity"
    project: str  # key into config.projects
    priority: int  # 1 (urgent) .. 5 (whenever); lower runs first
    model: str | None = None  # provider/model for harnesses with per-run selection
    depends_on: list[str] = field(default_factory=list)
    merge_required: bool = False  # dependents wait for the merge, not just DONE
    max_runtime: timedelta = timedelta(hours=2)
    stall_timeout: timedelta = timedelta(minutes=10)
    attempt: int = 1  # incremented on requeue after preempt/interrupt
    created_from_utterance: str = ""
    created_at: datetime = field(default_factory=utcnow)
    #: The task whose merge conflict this one resolves, if any.
    resolves: str | None = None
    #: Where this task's branch starts, when not the project's base branch: a
    #: conflict fix starts from the branch that conflicted, so its merge of
    #: the base is made on top of that work rather than beside it.
    start_from: str | None = None


# --------------------------------------------------------------------------
# Task run (one per attempt)
# --------------------------------------------------------------------------


class RunOutcome(StrEnum):
    """How an attempt ended."""

    DONE = "done"
    ERROR = "error"
    KILLED = "killed"
    PREEMPTED = "preempted"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"

    @property
    def is_resumable(self) -> bool:
        """Outcomes that requeue the task with a resume note and reuse the same
        worktree and branch. KILLED is deliberately absent:
        kill means kill."""
        return self in _RESUMABLE_OUTCOMES


_RESUMABLE_OUTCOMES = frozenset(
    {
        RunOutcome.PREEMPTED,
        RunOutcome.INTERRUPTED,
        RunOutcome.TIMEOUT,
    }
)


@dataclass
class TaskRun:
    task_id: str
    attempt: int
    slot: str
    worktree: Path
    branch: str
    base_ref: str  # commit the branch was cut from
    log_path: Path
    started_at: datetime = field(default_factory=utcnow)
    ended_at: datetime | None = None
    exit_code: int | None = None
    outcome: RunOutcome | None = None
    wip_commit: str | None = None  # sha of the auto-commit made at run end


# --------------------------------------------------------------------------
# Events (`tick` returns these; the brain narrates them)
# --------------------------------------------------------------------------


@dataclass
class Event:
    """Base for everything the manager reports upward. The brain turns these
    into one sentence each; nothing else subscribes."""

    at: datetime = field(default_factory=utcnow)


@dataclass
class BranchDeleted(Event):
    """A discarded task's branch, deleted once its grace period ran out."""

    task_id: str = ""
    branch: str = ""


@dataclass
class BrainstormChanged(Event):
    """Brainstorming started, stopped, or its drafts changed (buddy.ideation).

    Carries nothing but the fact: the dashboard re-reads `/api/brainstorm`,
    so there is one source of truth rather than a copy in every event.
    """

    active: bool = False
    drafts: int = 0


@dataclass
class TaskStarted(Event):
    task_id: str = ""
    slot: str = ""
    attempt: int = 1


@dataclass
class TaskFinished(Event):
    """A run reached a terminal state."""

    task_id: str = ""
    slot: str = ""
    attempt: int = 1
    outcome: RunOutcome = RunOutcome.DONE
    exit_code: int | None = None
    branch: str = ""
    summary: str = ""  # from adapter.parse_result


@dataclass
class TaskRequeued(Event):
    """Checkpointed and put back on the queue with a resume note."""

    task_id: str = ""
    next_attempt: int = 2
    reason: RunOutcome = RunOutcome.PREEMPTED


@dataclass
class TaskBlocked(Event):
    """A queued task whose dependency ended in a state it can never recover
    from. Leaving it queued silently forever is the one answer that helps
    nobody."""

    task_id: str = ""
    reason: str = ""


@dataclass
class SlotHealthChanged(Event):
    """WAITING_INPUT or STALLED. Buddy notifies and never acts alone."""

    slot: str = ""
    task_id: str = ""
    status: SlotStatus = SlotStatus.RUNNING
    detail: str = ""


@dataclass
class PreemptionProposal(Event):
    """Raised when the top ready task outranks a running one. The
     manager never preempts on its own; the brain asks and the user answers
    ."""

    proposal_id: str = ""
    victim_slot: str = ""
    victim_task_id: str = ""
    incoming_task_id: str = ""
