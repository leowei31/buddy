"""SQLite persistence: schema, round-trips, FTS, migrations."""

from datetime import timedelta
from pathlib import Path

import pytest

from buddy.models import (
    SLOT_NAMES,
    AgentSlot,
    RunOutcome,
    SlotStatus,
    TaskRun,
    TaskSpec,
    TaskState,
    utcnow,
)
from buddy.state import SCHEMA_VERSION, StateError, Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "state.db") as s:
        yield s


def make_task(store: Store, **overrides) -> TaskSpec:
    defaults = {
        "id": store.next_task_id(),
        "title": "Fix onboarding flow",
        "brief": "# Task\n...",
        "harness": "claude_code",
        "project": "webapp",
        "priority": 2,
    }
    spec = TaskSpec(**(defaults | overrides))
    store.create_task(spec)
    return spec


def make_run(task: TaskSpec, attempt: int = 1) -> TaskRun:
    return TaskRun(
        task_id=task.id,
        attempt=attempt,
        slot="Tuesday",
        worktree=Path("/tmp/wt") / task.id,
        branch=f"buddy/{task.id}-fix-onboarding",
        base_ref="abc1234",
        log_path=Path("/tmp/logs") / f"attempt-{attempt}.log",
    )


# -- schema ----------------------------------------------------------------


def test_wal_and_schema_version(store: Store, tmp_path: Path):
    conn = store._conn
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_all_seven_tables_from_section_3_5_exist(store: Store):
    names = {
        row[0]
        for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")
    }
    assert {
        "slots",
        "tasks",
        "task_runs",
        "conversation_log",
        "conversation_summaries",
        "conversation_log_fts",
        "memory",
    } <= names


def test_reopening_is_idempotent(tmp_path: Path):
    path = tmp_path / "state.db"
    with Store(path) as first:
        first.ensure_slots()
        make_task(first)
    with Store(path) as second:
        assert len(second.load_slots()) == 7
        assert second.next_task_id() == "t-0002"


def test_future_schema_is_refused(tmp_path: Path):
    path = tmp_path / "state.db"
    Store(path).close()
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION + 5}")
    conn.close()
    with pytest.raises(StateError, match="Upgrade Buddy"):
        Store(path)


# -- slots -----------------------------------------------------------------


def test_ensure_slots_creates_seven_idle_slots_once(store: Store):
    slots = store.ensure_slots()
    assert [s.name for s in slots] == list(SLOT_NAMES)
    assert all(s.status is SlotStatus.IDLE for s in slots)
    assert all(s.tmux_window == s.name for s in slots)

    # Idempotent, and it never clobbers live state: the name is a handle.
    slots[1].status = SlotStatus.RUNNING
    slots[1].task_id = "t-0001"
    store.save_slot(slots[1])
    again = store.ensure_slots()
    assert again[1].status is SlotStatus.RUNNING
    assert again[1].task_id == "t-0001"


def test_slot_round_trip(store: Store):
    store.ensure_slots()
    now = utcnow()
    slot = AgentSlot(
        name="Thursday",
        status=SlotStatus.WAITING_INPUT,
        task_id="t-0007",
        run_attempt=2,
        harness="opencode",
        priority=1,
        started_at=now,
        last_output_at=now,
        last_output="waiting on y/n",
    )
    store.save_slot(slot)
    loaded = {s.name: s for s in store.load_slots()}["Thursday"]
    assert loaded == slot


# -- tasks -----------------------------------------------------------------


def test_task_ids_are_sequential_and_never_reused(store: Store):
    assert store.next_task_id() == "t-0001"
    first = make_task(store)
    assert first.id == "t-0001"
    assert store.next_task_id() == "t-0002"
    second = make_task(store)
    store.set_task_state(second.id, TaskState.DISCARDED)
    assert store.next_task_id() == "t-0003"


def test_task_round_trip_preserves_every_field(store: Store):
    spec = make_task(
        store,
        model="google-vertex/gemini",
        depends_on=["t-0099"],
        merge_required=True,
        max_runtime=timedelta(minutes=45),
        stall_timeout=timedelta(minutes=3),
        created_from_utterance="fix the onboarding thing",
    )
    loaded = store.get_task(spec.id)
    assert loaded == spec


def test_queue_ordering_is_priority_then_fifo(store: Store):
    low = make_task(store, priority=5)
    urgent = make_task(store, priority=1)
    also_low = make_task(store, priority=5)
    queued = store.tasks_in_state(TaskState.QUEUED)
    assert [t.id for t in queued] == [urgent.id, low.id, also_low.id]


def test_set_state_and_unknown_task(store: Store):
    task = make_task(store)
    store.set_task_state(task.id, TaskState.MERGED)
    assert store.get_task_state(task.id) is TaskState.MERGED
    assert store.get_task_state("t-9999") is None
    with pytest.raises(StateError, match="no such task"):
        store.set_task_state("t-9999", TaskState.DONE)


def test_update_task_persists_priority_and_attempt(store: Store):
    task = make_task(store, priority=4)
    task.priority = 1
    task.attempt = 2
    store.update_task(task)
    assert store.get_task(task.id).priority == 1
    assert store.get_task(task.id).attempt == 2


def test_recent_tasks_filters_by_project(store: Store):
    make_task(store)
    other = make_task(store, project="cli")
    assert [t.id for t in store.recent_tasks(project="cli")] == [other.id]


# -- runs ------------------------------------------------------------------


def test_run_round_trip_and_finalization(store: Store):
    task = make_task(store)
    run = make_run(task)
    store.create_run(run)
    assert store.get_run(task.id, 1) == run
    assert [r.attempt for r in store.unfinished_runs()] == [1]

    run.ended_at = utcnow()
    run.exit_code = 0
    run.outcome = RunOutcome.DONE
    run.wip_commit = "def5678"
    store.save_run(run)
    assert store.get_run(task.id, 1) == run
    assert store.unfinished_runs() == []


def test_multiple_attempts_and_latest(store: Store):
    task = make_task(store)
    store.create_run(make_run(task, attempt=1))
    store.create_run(make_run(task, attempt=2))
    assert [r.attempt for r in store.runs_for(task.id)] == [1, 2]
    assert store.latest_run(task.id).attempt == 2


def test_run_requires_an_existing_task(store: Store):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        store.create_run(
            make_run(
                TaskSpec(id="t-9999", title="", brief="", harness="c", project="p", priority=3)
            )
        )


def test_saving_an_unknown_run_is_an_error(store: Store):
    task = make_task(store)
    with pytest.raises(StateError, match="no such run"):
        store.save_run(make_run(task, attempt=9))


# -- conversation, summaries, memory ---------------------------------------


def test_conversation_log_and_speaker_validation(store: Store):
    store.log_turn("user", "can you fix onboarding")
    store.log_turn("buddy", "on it")
    turns = store.recent_turns()
    assert [t["speaker"] for t in turns] == ["user", "buddy"]
    assert turns[0]["text"] == "can you fix onboarding"
    with pytest.raises(StateError, match="must be 'user' or 'buddy'"):
        store.log_turn("monday", "hello")


def test_recent_turns_returns_the_newest_in_chronological_order(store: Store):
    for i in range(10):
        store.log_turn("user", f"turn {i}")
    turns = store.recent_turns(limit=3)
    assert [t["text"] for t in turns] == ["turn 7", "turn 8", "turn 9"]


def test_fts_recall_finds_turns_verbatim(store: Store):
    store.log_turn("user", "the auth refactor should keep the retry logic")
    store.log_turn("user", "unrelated chatter about lunch")
    hits = store.search_turns("refactor")
    assert len(hits) == 1
    assert "retry logic" in hits[0]["text"]
    assert store.search_turns("nonexistent") == []


def test_fts_index_tracks_updates_and_deletes(store: Store):
    turn_id = store.log_turn("user", "findable sentinel")
    assert store.search_turns("sentinel")
    store._conn.execute("UPDATE conversation_log SET text = ? WHERE id = ?", ("changed", turn_id))
    assert store.search_turns("sentinel") == []
    assert store.search_turns("changed")
    store._conn.execute("DELETE FROM conversation_log WHERE id = ?", (turn_id,))
    assert store.search_turns("changed") == []


def test_summaries_round_trip(store: Store):
    store.save_summary("first", strategy="server", tokens_before=100, tokens_after=10)
    store.save_summary("second", strategy="client", tokens_before=200, tokens_after=20)
    latest = store.latest_summary()
    assert latest["summary"] == "second"
    assert latest["strategy"] == "client"
    assert latest["tokens_before"] == 200


def test_no_summary_yet(store: Store):
    assert store.latest_summary() is None


def test_memory_add_list_forget(store: Store):
    first = store.remember("prefers Codex for frontend")
    store.remember("never touch payments/ without asking")
    assert [m["fact"] for m in store.memories()] == [
        "prefers Codex for frontend",
        "never touch payments/ without asking",
    ]
    assert store.forget(first) is True
    assert store.forget(first) is False
    assert len(store.memories()) == 1


def test_the_database_is_private_even_when_it_already_existed(tmp_path):
    """state.db holds the whole conversation - and anything pasted
    into it - and was created 0644 inside a 0755 directory, readable by every
    account on the machine. Found that way on a real install."""
    import stat as st

    from buddy.config import Config

    home = tmp_path / "home"
    home.mkdir(mode=0o755)
    db = home / "state.db"
    db.write_bytes(b"")
    db.chmod(0o644)

    config = Config(home=home)
    config.paths.ensure()
    with Store(config.paths.db) as store:
        store.ensure_slots()
        store.log_turn("user", "my key is sk-not-really")

    assert st.S_IMODE(home.stat().st_mode) == 0o700
    for path in home.glob("state.db*"):
        assert st.S_IMODE(path.stat().st_mode) == 0o600, path.name


def test_a_version_one_database_upgrades_and_keeps_its_tasks(tmp_path):
    """Adding the conflict-resolution fields must not cost anyone their
    history: a database from before them is migrated in place."""
    import sqlite3

    from buddy.state import _SCHEMA_V1

    path = tmp_path / "state.db"
    old = sqlite3.connect(path)
    old.executescript(_SCHEMA_V1 + "\nPRAGMA user_version=1;")
    old.execute(
        "INSERT INTO tasks (id, title, brief, harness, project, priority, depends_on, "
        "merge_required, max_runtime_s, stall_timeout_s, attempt, created_from_utterance, "
        "created_at, state) VALUES ('t-0001','old','b','claude_code','webapp',3,'[]',0,7200,"
        "600,1,'','2026-09-01T00:00:00+00:00','done')"
    )
    old.commit()
    old.close()

    with Store(path) as store:
        [task] = store.recent_tasks()
        assert task.id == "t-0001" and task.resolves is None and task.start_from is None


def test_a_conflict_fix_round_trips_what_it_resolves_and_where_it_starts(store):
    make_task(store, id="t-0002", resolves="t-0001", start_from="buddy/t-0001-add-limits")
    again = store.get_task("t-0002")
    assert again.resolves == "t-0001"
    assert again.start_from == "buddy/t-0001-add-limits"
