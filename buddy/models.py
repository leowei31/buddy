"""Agent, task, run, and event types.

Pure data. Nothing here touches the filesystem, the database, tmux, or git.
The string-valued enums exist so the vocabularies stored in SQLite and sent
as JSON - task states, run outcomes - cannot drift. `StrEnum` gives identical
wire values and no divergence between `str()` and `format()`.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path


def utcnow() -> datetime:
    """Timezone-aware now. Every timestamp Buddy stores is UTC."""
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------

#: The longest name an agent may have. Long enough for a description, short
#: enough to say out loud and to fit a card and a tmux status line.
AGENT_NAME_MAX = 32

#: Letters and digits in any script, `-` and `_`, starting with a letter or
#: digit. The first character matters: a name that starts with `-` reads as
#: an option on every command line that takes one. Nothing here means
#: anything to tmux's target syntax, which splits on `:` and `.`.
_AGENT_NAME = re.compile(r"[^\W_][\w-]*")

#: What a task id looks like. A name that looks like one would make
#: `buddy logs t-0007` mean two different things.
_TASK_ID = re.compile(r"t-\d+", re.IGNORECASE)


class AgentNameError(ValueError):
    """A name an agent cannot have, with the reason and what would work."""


def normalize_agent_name(name: str) -> str:
    """The name as it is stored: trimmed, with runs of spaces made `-`.

    Spaces are the one thing people reliably say that a name cannot hold, so
    "code reviewer" becomes `code-reviewer` rather than a refusal.
    """
    return re.sub(r"\s+", "-", name.strip())


def check_agent_name(name: str) -> str:
    """`name`, normalized, or an `AgentNameError` saying why it cannot be one."""
    wanted = normalize_agent_name(name)
    if not wanted:
        raise AgentNameError("an agent's name cannot be empty")
    if len(wanted) > AGENT_NAME_MAX:
        raise AgentNameError(
            f"{wanted!r} is {len(wanted)} characters; names are {AGENT_NAME_MAX} at most"
        )
    if not _AGENT_NAME.fullmatch(wanted):
        raise AgentNameError(
            f"{wanted!r} cannot be an agent's name: use letters, digits, '-' and '_', "
            "starting with a letter or a digit"
        )
    if _TASK_ID.fullmatch(wanted):
        raise AgentNameError(f"{wanted!r} looks like a task id; pick a name that does not")
    return wanted


def default_agent_name(title: str, taken: Collection[str] = ()) -> str:
    """A name for an agent nobody named: the first words of its title.

    `Add rate limiting to the API` becomes `add-rate-limiting` - short enough
    to say, specific enough to recognise. A name already in use by a live
    agent gets the first free `-2`, `-3`, ... after it.
    """
    words = re.findall(r"[^\W_]+", title.lower())
    base = ""
    for word in words[:3]:
        candidate = f"{base}-{word}" if base else word
        if len(candidate) > AGENT_NAME_MAX - 3:  # room for a suffix
            break
        base = candidate
    if not base or _TASK_ID.fullmatch(base):
        base = "agent"
    lowered = {name.lower() for name in taken}
    if base not in lowered:
        return base
    number = 2
    while f"{base}-{number}" in lowered:
        number += 1
    return f"{base}-{number}"


class AgentStatus(StrEnum):
    """What a running agent is doing. An agent exists only while its task
    runs, so there is no idle or finished status: the task's own state and
    its runs record how it ended."""

    RUNNING = "running"
    WAITING_INPUT = "waiting_input"  # harness is asking a question
    STALLED = "stalled"  # no output for stall_timeout


@dataclass
class Agent:
    """A running task, under the name the user calls it by.

    Also the name of its tmux window. Created when the task starts and
    removed when that attempt ends; a requeued task keeps its name and gets
    a new window when it starts again.
    """

    name: str
    task_id: str
    run_attempt: int
    status: AgentStatus = AgentStatus.RUNNING
    harness: str | None = None
    priority: int | None = None
    started_at: datetime | None = None
    last_output_at: datetime | None = None  # last time the log file grew
    last_output: str = ""  # last ~20 lines, for status queries and voice summaries


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
    #: What the user calls this task's agent, and its tmux window's name.
    #: Unique among queued and running tasks; given at spawn, or derived
    #: from the title when nobody named it.
    agent: str = ""


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
    agent: str  # the name it ran under
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
    agent: str = ""
    attempt: int = 1


@dataclass
class TaskFinished(Event):
    """A run reached a terminal state."""

    task_id: str = ""
    agent: str = ""
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
class AgentHealthChanged(Event):
    """WAITING_INPUT or STALLED, or back to RUNNING. Buddy notifies and never
    acts alone."""

    agent: str = ""
    task_id: str = ""
    status: AgentStatus = AgentStatus.RUNNING
    detail: str = ""


@dataclass
class PreemptionProposal(Event):
    """Raised when the top ready task outranks a running one. The manager
    never preempts on its own; the brain asks and the user answers."""

    proposal_id: str = ""
    victim_agent: str = ""
    victim_task_id: str = ""
    incoming_task_id: str = ""
