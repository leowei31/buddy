"""SQLite (WAL) persistence.

The DB is the source of truth for *what* happened. The log files are the
source of truth for *the output*. The worktrees are the source of truth for
*the code*. Buddy's in-memory state is a cache of all three, rebuilt on
startup.

This module is synchronous on purpose. Buddy requires `create_subprocess_exec`
for tmux and git because those are processes that can block for minutes; a
local SQLite write in WAL mode is sub-millisecond, and wrapping it in a
thread pool would add a failure mode to buy nothing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from buddy.models import (
    Agent,
    AgentNameError,
    AgentStatus,
    RunOutcome,
    TaskRun,
    TaskSpec,
    TaskState,
    default_agent_name,
    utcnow,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3


class StateError(Exception):
    pass


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

_SCHEMA_V1 = """
CREATE TABLE slots (
    name            TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    task_id         TEXT,
    run_attempt     INTEGER,
    harness         TEXT,
    priority        INTEGER,
    tmux_window     TEXT NOT NULL,
    started_at      TEXT,
    last_output_at  TEXT,
    last_output     TEXT NOT NULL DEFAULT ''
);

CREATE TABLE tasks (
    id                     TEXT PRIMARY KEY,
    title                  TEXT NOT NULL,
    brief                  TEXT NOT NULL,
    harness                TEXT NOT NULL,
    model                  TEXT,
    project                TEXT NOT NULL,
    priority               INTEGER NOT NULL,
    depends_on             TEXT NOT NULL DEFAULT '[]',
    merge_required         INTEGER NOT NULL DEFAULT 0,
    max_runtime_s          REAL NOT NULL,
    stall_timeout_s        REAL NOT NULL,
    attempt                INTEGER NOT NULL DEFAULT 1,
    created_from_utterance TEXT NOT NULL DEFAULT '',
    created_at             TEXT NOT NULL,
    state                  TEXT NOT NULL,
    -- When the task was discarded, so the sweep can delete its branch once the
    -- grace period has passed. Not part of TaskSpec: it is a fact about the
    -- task's lifecycle, not about the work.
    discarded_at           TEXT
);
CREATE INDEX tasks_state_priority ON tasks (state, priority, created_at);
CREATE INDEX tasks_project ON tasks (project);

CREATE TABLE task_runs (
    task_id     TEXT NOT NULL,
    attempt     INTEGER NOT NULL,
    slot        TEXT NOT NULL,
    worktree    TEXT NOT NULL,
    branch      TEXT NOT NULL,
    base_ref    TEXT NOT NULL,
    log_path    TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    exit_code   INTEGER,
    outcome     TEXT,
    wip_commit  TEXT,
    PRIMARY KEY (task_id, attempt),
    FOREIGN KEY (task_id) REFERENCES tasks (id)
);

CREATE TABLE conversation_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    speaker TEXT NOT NULL CHECK (speaker IN ('user', 'buddy')),
    text    TEXT NOT NULL
);

CREATE TABLE conversation_summaries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    summary        TEXT NOT NULL,
    tokens_before  INTEGER,
    tokens_after   INTEGER,
    strategy       TEXT NOT NULL,
    through_turn   INTEGER
);

CREATE TABLE memory (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    fact        TEXT NOT NULL,
    source_turn INTEGER
);

-- Backs the brain's `recall` tool (context layer 3). External-content
-- table plus triggers, so the index can never drift from the log.
CREATE VIRTUAL TABLE conversation_log_fts USING fts5 (
    text,
    content='conversation_log',
    content_rowid='id'
);
CREATE TRIGGER conversation_log_ai AFTER INSERT ON conversation_log BEGIN
    INSERT INTO conversation_log_fts (rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER conversation_log_ad AFTER DELETE ON conversation_log BEGIN
    INSERT INTO conversation_log_fts (conversation_log_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER conversation_log_au AFTER UPDATE ON conversation_log BEGIN
    INSERT INTO conversation_log_fts (conversation_log_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
    INSERT INTO conversation_log_fts (rowid, text) VALUES (new.id, new.text);
END;
"""

#: Ordered migrations. Index i takes the schema from version i to i+1, so a
#: new one is appended and never edited in place.
#: v2: the conflict-resolution task - what it resolves, and where it starts.
_SCHEMA_V2 = """
ALTER TABLE tasks ADD COLUMN resolves TEXT;
ALTER TABLE tasks ADD COLUMN start_from TEXT;
"""

#: v3: named agents instead of seven fixed slots. A task carries the name of
#: its agent; `agents` holds only the ones running now. A task that was
#: running in a slot keeps that slot's name, which is also its tmux window's,
#: so upgrading mid-run loses nothing. Every other task is named for its id.
_SCHEMA_V3 = """
ALTER TABLE tasks ADD COLUMN agent TEXT NOT NULL DEFAULT '';
UPDATE tasks SET agent = (SELECT s.name FROM slots s WHERE s.task_id = tasks.id)
    WHERE id IN (
        SELECT task_id FROM slots
        WHERE task_id IS NOT NULL AND status IN ('running', 'waiting_input', 'stalled')
    );
UPDATE tasks SET agent = id WHERE agent = '';
-- A name is a handle: two live agents can never answer to the same one.
CREATE UNIQUE INDEX tasks_live_agent ON tasks (agent COLLATE NOCASE)
    WHERE state IN ('queued', 'running');
ALTER TABLE task_runs RENAME COLUMN slot TO agent;
CREATE TABLE agents (
    name            TEXT PRIMARY KEY COLLATE NOCASE,
    task_id         TEXT NOT NULL,
    run_attempt     INTEGER NOT NULL,
    status          TEXT NOT NULL,
    harness         TEXT,
    priority        INTEGER,
    started_at      TEXT,
    last_output_at  TEXT,
    last_output     TEXT NOT NULL DEFAULT ''
);
INSERT INTO agents (name, task_id, run_attempt, status, harness, priority,
                    started_at, last_output_at, last_output)
    SELECT name, task_id, COALESCE(run_attempt, 1), status, harness, priority,
           started_at, last_output_at, last_output
    FROM slots
    WHERE task_id IS NOT NULL AND status IN ('running', 'waiting_input', 'stalled');
DROP TABLE slots;
"""

MIGRATIONS: tuple[str, ...] = (_SCHEMA_V1, _SCHEMA_V2, _SCHEMA_V3)


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _agent_from_row(row: sqlite3.Row) -> Agent:
    return Agent(
        name=row["name"],
        task_id=row["task_id"],
        run_attempt=row["run_attempt"],
        status=AgentStatus(row["status"]),
        harness=row["harness"],
        priority=row["priority"],
        started_at=_dt(row["started_at"]),
        last_output_at=_dt(row["last_output_at"]),
        last_output=row["last_output"],
    )


def _task_from_row(row: sqlite3.Row) -> TaskSpec:
    return TaskSpec(
        id=row["id"],
        title=row["title"],
        brief=row["brief"],
        harness=row["harness"],
        model=row["model"],
        project=row["project"],
        priority=row["priority"],
        depends_on=json.loads(row["depends_on"]),
        merge_required=bool(row["merge_required"]),
        max_runtime=timedelta(seconds=row["max_runtime_s"]),
        stall_timeout=timedelta(seconds=row["stall_timeout_s"]),
        attempt=row["attempt"],
        created_from_utterance=row["created_from_utterance"],
        created_at=_dt(row["created_at"]),  # type: ignore[arg-type]
        resolves=row["resolves"],
        start_from=row["start_from"],
        agent=row["agent"],
    )


def _run_from_row(row: sqlite3.Row) -> TaskRun:
    return TaskRun(
        task_id=row["task_id"],
        attempt=row["attempt"],
        agent=row["agent"],
        worktree=Path(row["worktree"]),
        branch=row["branch"],
        base_ref=row["base_ref"],
        log_path=Path(row["log_path"]),
        started_at=_dt(row["started_at"]),  # type: ignore[arg-type]
        ended_at=_dt(row["ended_at"]),
        exit_code=row["exit_code"],
        outcome=RunOutcome(row["outcome"]) if row["outcome"] else None,
        wip_commit=row["wip_commit"],
    )


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class Store:
    """Every read and write of `~/.buddy/state.db`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Before connecting, so SQLite gives the -wal and -shm files it
        # creates the same mode: it copies the database file's.
        path.touch(mode=0o600, exist_ok=True)
        with contextlib.suppress(OSError):
            for existing in (path, *path.parent.glob(f"{path.name}-*")):
                existing.chmod(0o600)
        self._conn = sqlite3.connect(path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # WAL so the dashboard's reads never contend with Buddy's writes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        #: Called with each conversation turn as it is written. The
        #: dashboard's `/ws/conversation` wants new turns "as the brain
        #: writes them", and `log_turn` is the one place they are
        #: written, so a listener here beats anyone polling the table.
        self._turn_listeners: list[Callable[[dict[str, Any]], None]] = []
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- schema ------------------------------------------------------------

    def _migrate(self) -> None:
        current = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if current > SCHEMA_VERSION:
            raise StateError(
                f"{self.path} is schema version {current}, but this Buddy understands "
                f"up to {SCHEMA_VERSION}. Upgrade Buddy."
            )
        for version in range(current, SCHEMA_VERSION):
            # `executescript` commits any open transaction before it runs, so
            # the migration carries its own BEGIN/COMMIT: a failed migration
            # leaves the schema untouched rather than half-applied.
            self._conn.executescript(
                f"BEGIN;\n{MIGRATIONS[version]}\nPRAGMA user_version={version + 1};\nCOMMIT;"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # -- schema, for setup ------------------------------------------

    def schema_version(self) -> int:
        """`PRAGMA user_version`, which the migrations advance.

        `buddy setup` step 4 verifies against it rather than trusting that
        opening the store did the right thing.
        """
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def table_names(self) -> set[str]:
        """Every table and virtual table in the database.

        Step 4 checks for `conversation_log_fts` by name: an FTS index that
        quietly failed to build costs the brain its `recall`,
        and nothing else would notice.
        """
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        ).fetchall()
        return {row["name"] for row in rows}

    # -- agents -----------------------------------------------------

    def load_agents(self) -> list[Agent]:
        """Every running agent, most urgent first, then oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM agents ORDER BY priority IS NULL, priority ASC, started_at ASC, name ASC"
        ).fetchall()
        return [_agent_from_row(row) for row in rows]

    def get_agent(self, name: str) -> Agent | None:
        """The running agent with this name, whatever its case."""
        row = self._conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        return _agent_from_row(row) if row else None

    def save_agent(self, agent: Agent) -> None:
        self._conn.execute(
            """
            INSERT INTO agents (name, task_id, run_attempt, status, harness, priority,
                                started_at, last_output_at, last_output)
            VALUES (:name, :task_id, :run_attempt, :status, :harness, :priority,
                    :started_at, :last_output_at, :last_output)
            ON CONFLICT (name) DO UPDATE SET
                task_id=excluded.task_id, run_attempt=excluded.run_attempt,
                status=excluded.status, harness=excluded.harness,
                priority=excluded.priority, started_at=excluded.started_at,
                last_output_at=excluded.last_output_at, last_output=excluded.last_output
            """,
            {
                "name": agent.name,
                "task_id": agent.task_id,
                "run_attempt": agent.run_attempt,
                "status": agent.status.value,
                "harness": agent.harness,
                "priority": agent.priority,
                "started_at": _iso(agent.started_at),
                "last_output_at": _iso(agent.last_output_at),
                "last_output": agent.last_output,
            },
        )

    def remove_agent(self, name: str) -> None:
        self._conn.execute("DELETE FROM agents WHERE name = ?", (name,))

    def live_agent_names(self) -> dict[str, str]:
        """Names taken by queued or running tasks, lowercased, to their task ids."""
        rows = self._conn.execute(
            "SELECT agent, id FROM tasks WHERE state IN (?, ?)",
            (TaskState.QUEUED.value, TaskState.RUNNING.value),
        ).fetchall()
        return {row["agent"].lower(): row["id"] for row in rows}

    def task_for_agent(self, name: str) -> TaskSpec | None:
        """The task an agent name refers to: the live one if there is one,
        otherwise the most recent task that ran under that name."""
        row = self._conn.execute(
            """
            SELECT * FROM tasks WHERE agent = ? COLLATE NOCASE
            ORDER BY state IN (?, ?) DESC, created_at DESC, id DESC LIMIT 1
            """,
            (name, TaskState.QUEUED.value, TaskState.RUNNING.value),
        ).fetchone()
        return _task_from_row(row) if row else None

    def find_task(self, target: str) -> TaskSpec | None:
        """A task by id, or by the name of its agent."""
        return self.get_task(target) or self.task_for_agent(target)

    # -- tasks ------------------------------------------------------

    def next_task_id(self) -> str:
        """`t-0142`. Derived from the highest id ever issued; `tasks`
        holds every task ever created, so numbers are never reused."""
        row = self._conn.execute(
            "SELECT MAX(CAST(SUBSTR(id, 3) AS INTEGER)) AS n FROM tasks WHERE id LIKE 't-%'"
        ).fetchone()
        return f"t-{(row['n'] or 0) + 1:04d}"

    def create_task(self, task: TaskSpec, state: TaskState = TaskState.QUEUED) -> None:
        """Save a new task. One with no agent name is given one from its title.

        Raises `AgentNameError` when the name belongs to another queued or
        running task. `AgentManager.submit` checks that first, with a better
        message; this is what holds when two processes race for one name.
        """
        if not task.agent:
            task.agent = default_agent_name(task.title, self.live_agent_names())
        try:
            self._insert_task(task, state)
        except sqlite3.IntegrityError as exc:
            holder = self.live_agent_names().get(task.agent.lower())
            if holder is None or holder == task.id:
                raise
            raise AgentNameError(f"{task.agent} is already {holder}; pick another name") from exc

    def _insert_task(self, task: TaskSpec, state: TaskState) -> None:
        self._conn.execute(
            """
            INSERT INTO tasks (id, title, brief, harness, model, project, priority,
                               depends_on, merge_required, max_runtime_s, stall_timeout_s,
                               attempt, created_from_utterance, created_at, state,
                               resolves, start_from, agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.id,
                task.title,
                task.brief,
                task.harness,
                task.model,
                task.project,
                task.priority,
                json.dumps(task.depends_on),
                int(task.merge_required),
                task.max_runtime.total_seconds(),
                task.stall_timeout.total_seconds(),
                task.attempt,
                task.created_from_utterance,
                _iso(task.created_at),
                state.value,
                task.resolves,
                task.start_from,
                task.agent,
            ),
        )

    def get_task(self, task_id: str) -> TaskSpec | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _task_from_row(row) if row else None

    def get_task_state(self, task_id: str) -> TaskState | None:
        row = self._conn.execute("SELECT state FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return TaskState(row["state"]) if row else None

    def set_task_state(self, task_id: str, state: TaskState) -> None:
        """Discarding stamps the time, so the grace period has something to
        measure from; any other transition clears it."""
        discarded_at = _iso(utcnow()) if state is TaskState.DISCARDED else None
        cursor = self._conn.execute(
            "UPDATE tasks SET state = ?, discarded_at = ? WHERE id = ?",
            (state.value, discarded_at, task_id),
        )
        if cursor.rowcount == 0:
            raise StateError(f"no such task: {task_id}")

    def discarded_at(self, task_id: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT discarded_at FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        return _dt(row["discarded_at"]) if row else None

    def discarded_tasks(self) -> list[tuple[TaskSpec, datetime | None]]:
        """Every discarded task with the time it was discarded."""
        rows = self._conn.execute(
            "SELECT * FROM tasks WHERE state = ? ORDER BY discarded_at ASC",
            (TaskState.DISCARDED.value,),
        ).fetchall()
        return [(_task_from_row(row), _dt(row["discarded_at"])) for row in rows]

    def update_task(self, task: TaskSpec) -> None:
        """Persist the mutable fields: priority (reprioritize), attempt
        (requeue), and the agent's name for a task created without one."""
        self._conn.execute(
            "UPDATE tasks SET priority = ?, attempt = ?, brief = ?, depends_on = ?, agent = ? "
            "WHERE id = ?",
            (
                task.priority,
                task.attempt,
                task.brief,
                json.dumps(task.depends_on),
                task.agent,
                task.id,
            ),
        )

    def tasks_in_state(self, *states: TaskState) -> list[TaskSpec]:
        """Ordered as the scheduler wants them: lowest priority number first,
        FIFO within equal priority."""
        placeholders = ", ".join("?" for _ in states)
        rows = self._conn.execute(
            f"SELECT * FROM tasks WHERE state IN ({placeholders}) "
            "ORDER BY priority ASC, created_at ASC, id ASC",
            tuple(s.value for s in states),
        ).fetchall()
        return [_task_from_row(row) for row in rows]

    def recent_tasks(self, project: str | None = None, limit: int = 20) -> list[TaskSpec]:
        if project:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    # -- runs -------------------------------------------------------

    def create_run(self, run: TaskRun) -> None:
        self._conn.execute(
            """
            INSERT INTO task_runs (task_id, attempt, agent, worktree, branch, base_ref,
                                   log_path, started_at, ended_at, exit_code, outcome, wip_commit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.task_id,
                run.attempt,
                run.agent,
                str(run.worktree),
                run.branch,
                run.base_ref,
                str(run.log_path),
                _iso(run.started_at),
                _iso(run.ended_at),
                run.exit_code,
                run.outcome.value if run.outcome else None,
                run.wip_commit,
            ),
        )

    def save_run(self, run: TaskRun) -> None:
        """Update the mutable tail of a run: how and when it ended."""
        cursor = self._conn.execute(
            """
            UPDATE task_runs SET agent = ?, ended_at = ?, exit_code = ?, outcome = ?,
                                 wip_commit = ?
            WHERE task_id = ? AND attempt = ?
            """,
            (
                run.agent,
                _iso(run.ended_at),
                run.exit_code,
                run.outcome.value if run.outcome else None,
                run.wip_commit,
                run.task_id,
                run.attempt,
            ),
        )
        if cursor.rowcount == 0:
            raise StateError(f"no such run: {run.task_id} attempt {run.attempt}")

    def get_run(self, task_id: str, attempt: int) -> TaskRun | None:
        row = self._conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? AND attempt = ?", (task_id, attempt)
        ).fetchone()
        return _run_from_row(row) if row else None

    def runs_for(self, task_id: str) -> list[TaskRun]:
        rows = self._conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY attempt ASC", (task_id,)
        ).fetchall()
        return [_run_from_row(row) for row in rows]

    def latest_run(self, task_id: str) -> TaskRun | None:
        row = self._conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY attempt DESC LIMIT 1", (task_id,)
        ).fetchone()
        return _run_from_row(row) if row else None

    def unfinished_runs(self) -> list[TaskRun]:
        """Runs with no outcome. Restart reconciliation starts here."""
        rows = self._conn.execute(
            "SELECT * FROM task_runs WHERE outcome IS NULL ORDER BY started_at ASC"
        ).fetchall()
        return [_run_from_row(row) for row in rows]

    # -- conversation ----------------------------------------

    def on_turn(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Be told about conversation turns as they are written.

        Returns the unsubscribe callable. A listener that raises is logged and
        dropped from consideration for that turn: a viewer must never be able
        to fail a write.
        """
        self._turn_listeners.append(callback)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._turn_listeners.remove(callback)

        return unsubscribe

    def log_turn(self, speaker: str, text: str, *, ts: datetime | None = None) -> int:
        if speaker not in ("user", "buddy"):
            raise StateError(f"speaker must be 'user' or 'buddy', not {speaker!r}")
        stamp = _iso(ts or utcnow())
        cursor = self._conn.execute(
            "INSERT INTO conversation_log (ts, speaker, text) VALUES (?, ?, ?)",
            (stamp, speaker, text),
        )
        turn_id = int(cursor.lastrowid or 0)
        row = {"id": turn_id, "ts": stamp, "speaker": speaker, "text": text}
        for callback in list(self._turn_listeners):
            try:
                callback(row)
            except Exception:  # noqa: BLE001 - a viewer cannot fail a write
                logger.exception("conversation listener failed")
        return turn_id

    def recent_turns(
        self, limit: int = 20, *, before_id: int | None = None
    ) -> list[dict[str, Any]]:
        """The last `limit` turns, oldest first.

        `before_id` pages backwards through the transcript, which is what the
        dashboard's `GET /api/conversation?before=…` is for.
        """
        if before_id is None:
            rows = self._conn.execute(
                "SELECT * FROM (SELECT * FROM conversation_log ORDER BY id DESC LIMIT ?) "
                "ORDER BY id ASC",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM (SELECT * FROM conversation_log WHERE id < ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (before_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_turns(
        self, query: str, *, since: datetime | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Backs `recall`. Returns matching turns verbatim, newest first.

        `recall` takes a `project` argument, but `conversation_log` is
        (id, ts, speaker, text) with no project column, so filtering by
        project is the brain's job, over the text it gets back.
        """
        sql = [
            "SELECT c.id, c.ts, c.speaker, c.text FROM conversation_log_fts f",
            "JOIN conversation_log c ON c.id = f.rowid",
            "WHERE conversation_log_fts MATCH ?",
        ]
        params: list[Any] = [query]
        if since:
            sql.append("AND c.ts >= ?")
            params.append(_iso(since))
        sql.append("ORDER BY c.id DESC LIMIT ?")
        params.append(limit)
        rows = self._conn.execute(" ".join(sql), params).fetchall()
        return [dict(row) for row in rows]

    def save_summary(
        self,
        summary: str,
        *,
        strategy: str,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
        through_turn: int | None = None,
    ) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO conversation_summaries
                (ts, summary, tokens_before, tokens_after, strategy, through_turn)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_iso(utcnow()), summary, tokens_before, tokens_after, strategy, through_turn),
        )
        return int(cursor.lastrowid or 0)

    def latest_summary(self) -> dict[str, Any] | None:
        """What a fresh session starts from instead of amnesia."""
        row = self._conn.execute(
            "SELECT * FROM conversation_summaries ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # -- memory (context layer 4) -----------------------------------------

    def remember(self, fact: str, *, source_turn: int | None = None) -> int:
        cursor = self._conn.execute(
            "INSERT INTO memory (ts, fact, source_turn) VALUES (?, ?, ?)",
            (_iso(utcnow()), fact, source_turn),
        )
        return int(cursor.lastrowid or 0)

    def memories(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM memory ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]

    def forget(self, memory_id: int) -> bool:
        return self._conn.execute("DELETE FROM memory WHERE id = ?", (memory_id,)).rowcount > 0
