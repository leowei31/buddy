"""SQLite persistence: schema, round-trips, FTS, migrations."""

from datetime import timedelta
from pathlib import Path

import pytest

from buddy.models import (
    Agent,
    AgentNameError,
    AgentStatus,
    RunOutcome,
    TaskRun,
    TaskSpec,
    TaskState,
    utcnow,
)
from buddy.state import MIGRATIONS, SCHEMA_VERSION, StateError, Store


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
        agent=task.agent or "scout",
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


def test_every_table_exists(store: Store):
    names = {
        row[0]
        for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table')")
    }
    assert {
        "agents",
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
        make_task(first)
    with Store(path) as second:
        assert second.schema_version() == SCHEMA_VERSION
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


# -- agents ----------------------------------------------------------------


def test_an_agent_round_trips_and_is_found_whatever_its_case(store: Store):
    now = utcnow()
    agent = Agent(
        name="Scout",
        task_id="t-0007",
        run_attempt=2,
        status=AgentStatus.WAITING_INPUT,
        harness="opencode",
        priority=1,
        started_at=now,
        last_output_at=now,
        last_output="waiting on y/n",
    )
    store.save_agent(agent)

    assert store.get_agent("scout") == agent
    assert store.load_agents() == [agent]
    store.remove_agent("SCOUT")
    assert store.load_agents() == []


def test_running_agents_come_most_urgent_first_then_oldest(store: Store):
    now = utcnow()
    for name, priority, age in (("late", 2, 1), ("early", 2, 5), ("urgent", 1, 0)):
        store.save_agent(
            Agent(name, f"t-{name}", 1, priority=priority, started_at=now - timedelta(minutes=age))
        )
    assert [a.name for a in store.load_agents()] == ["urgent", "early", "late"]


def test_a_task_saved_without_a_name_is_given_one_from_its_title(store: Store):
    first = make_task(store, title="Add rate limiting to the API")
    second = make_task(store, title="Add rate limiting to the API")
    assert first.agent == "add-rate-limiting"
    assert second.agent == "add-rate-limiting-2"
    assert store.get_task(second.id).agent == "add-rate-limiting-2"


def test_two_live_tasks_can_never_share_a_name(store: Store):
    """Enforced by the database, not only by the manager: two processes
    racing for one name must not both win."""
    make_task(store, agent="scout")
    with pytest.raises(AgentNameError, match="Scout is already t-0001"):
        make_task(store, agent="Scout")


def test_a_finished_agents_name_is_free_again(store: Store):
    first = make_task(store, agent="scout")
    store.set_task_state(first.id, TaskState.DONE)
    second = make_task(store, agent="scout")
    assert store.live_agent_names() == {"scout": second.id}


def test_a_name_refers_to_the_live_task_and_otherwise_to_the_latest(store: Store):
    old = make_task(store, agent="scout")
    store.set_task_state(old.id, TaskState.DONE)
    assert store.task_for_agent("SCOUT").id == old.id

    live = make_task(store, agent="scout")
    assert store.task_for_agent("scout").id == live.id
    assert store.find_task(old.id).id == old.id, "an id still finds exactly its task"
    assert store.find_task("nobody") is None


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


def test_a_running_slot_becomes_an_agent_of_the_same_name_on_upgrade(tmp_path):
    """Upgrading from seven fixed slots, mid-run. A slot's name was also its
    tmux window's, so keeping it as the agent's name means the window that
    is running right now is still found - and nothing is interrupted."""
    import sqlite3

    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.executescript(MIGRATIONS[0] + MIGRATIONS[1] + "PRAGMA user_version=2;")
    insert = (
        "INSERT INTO tasks (id, title, brief, harness, project, priority, max_runtime_s, "
        "stall_timeout_s, created_at, state) VALUES (?, 't', 'b', 'claude_code', 'webapp', 3, "
        "7200, 600, ?, ?)"
    )
    conn.execute(insert, ("t-0001", "2026-01-01T00:00:00+00:00", "running"))
    conn.execute(insert, ("t-0002", "2026-01-02T00:00:00+00:00", "queued"))
    conn.execute(
        "INSERT INTO task_runs (task_id, attempt, slot, worktree, branch, base_ref, log_path, "
        "started_at) VALUES ('t-0001', 1, 'Tuesday', '/w', 'b', 'r', '/l', "
        "'2026-01-01T00:00:00+00:00')"
    )
    for name in ("Monday", "Tuesday"):
        conn.execute(
            "INSERT INTO slots (name, status, tmux_window) VALUES (?, 'idle', ?)", (name, name)
        )
    conn.execute(
        "UPDATE slots SET status = 'running', task_id = 't-0001', run_attempt = 1, "
        "last_output = 'working' WHERE name = 'Tuesday'"
    )
    conn.commit()
    conn.close()

    with Store(path) as store:
        assert store.schema_version() == SCHEMA_VERSION
        assert [(a.name, a.task_id, a.last_output) for a in store.load_agents()] == [
            ("Tuesday", "t-0001", "working")
        ]
        assert store.get_task("t-0001").agent == "Tuesday"
        assert store.get_run("t-0001", 1).agent == "Tuesday"
        assert store.get_task("t-0002").agent == "t-0002", "every other task is named for its id"
        assert "slots" not in store.table_names()
