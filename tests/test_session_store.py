"""SessionStore — real SQLite against a tmp_path file (the DB *is* the unit
under test here, so mocking it would assert nothing)."""

import sqlite3
from contextlib import closing

import pytest

from omni.session_store import SessionStore, _now


@pytest.fixture
def store(tmp_path):
    return SessionStore(str(tmp_path / "s.db"))


def test_now_is_iso_utc():
    assert _now().endswith("+00:00")


def test_schema_is_created_and_reopening_is_idempotent(tmp_path):
    path = str(tmp_path / "s.db")
    SessionStore(path)
    sid = SessionStore(path).create_session("/p", "m", "t")  # second open must not wipe/fail
    assert SessionStore(path).session_exists(sid)


def test_create_session_returns_short_unique_ids(store):
    a = store.create_session("/p", "model", "task a")
    b = store.create_session("/p", "model", "task b")
    assert a != b and len(a) == 8


def test_create_session_starts_running(store):
    sid = store.create_session("/p", "m", "t")
    assert store.list_sessions()[0]["status"] == "running"


def test_session_exists(store):
    sid = store.create_session("/p", "m", "t")
    assert store.session_exists(sid) is True
    assert store.session_exists("nope") is False


def test_resolve_session_id_by_id_and_name(store):
    sid = store.create_session("/p", "m", "t", name="my-name")
    assert store.resolve_session_id(sid) == sid
    assert store.resolve_session_id("my-name") == sid
    assert store.resolve_session_id("missing") is None


def test_session_names_are_unique(store):
    """The DB's UNIQUE index is surfaced as an actionable ValueError, not a
    raw sqlite3.IntegrityError."""
    store.create_session("/p", "m", "t", name="dup")
    with pytest.raises(ValueError, match="already in use"):
        store.create_session("/p", "m", "t2", name="dup")


def test_unnamed_sessions_do_not_collide_on_null_name(store):
    """A UNIQUE index on name must still allow many NULL names."""
    store.create_session("/p", "m", "a")
    store.create_session("/p", "m", "b")
    assert len(store.list_sessions()) == 2


def test_append_and_load_messages_roundtrip_in_order(store):
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, {"role": "system", "content": "sys"})
    store.append_message(sid, 1, {"role": "user", "content": "hi"})
    msgs = store.load_messages(sid)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "hi"


def test_tool_calls_are_json_roundtripped(store):
    sid = store.create_session("/p", "m", "t")
    calls = [{"id": "1", "type": "function",
              "function": {"name": "read_file", "arguments": '{"path": "x"}'}}]
    store.append_message(sid, 0, {"role": "assistant", "content": None, "tool_calls": calls})
    loaded = store.load_messages(sid)[0]
    assert loaded["tool_calls"] == calls


def test_message_without_tool_calls_omits_the_key(store):
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, {"role": "user", "content": "x"})
    assert "tool_calls" not in store.load_messages(sid)[0]


def test_load_messages_is_scoped_per_session(store):
    a = store.create_session("/p", "m", "a")
    b = store.create_session("/p", "m", "b")
    store.append_message(a, 0, {"role": "user", "content": "for-a"})
    store.append_message(b, 0, {"role": "user", "content": "for-b"})
    assert store.load_messages(a)[0]["content"] == "for-a"
    assert len(store.load_messages(b)) == 1


def test_load_messages_for_unknown_session_is_empty(store):
    assert store.load_messages("nope") == []


def test_replace_messages_overwrites_with_fresh_seq(store):
    sid = store.create_session("/p", "m", "t")
    for i in range(5):
        store.append_message(sid, i, {"role": "user", "content": f"m{i}"})
    store.replace_messages(sid, [
        {"role": "system", "content": "summary"},
        {"role": "user", "content": "recent"},
    ])
    msgs = store.load_messages(sid)
    assert [m["content"] for m in msgs] == ["summary", "recent"]


def test_replace_messages_preserves_tool_calls(store):
    sid = store.create_session("/p", "m", "t")
    calls = [{"function": {"name": "x", "arguments": "{}"}}]
    store.replace_messages(sid, [{"role": "assistant", "content": None, "tool_calls": calls}])
    assert store.load_messages(sid)[0]["tool_calls"] == calls


def test_replace_messages_only_touches_its_own_session(store):
    a = store.create_session("/p", "m", "a")
    b = store.create_session("/p", "m", "b")
    store.append_message(b, 0, {"role": "user", "content": "keep-me"})
    store.replace_messages(a, [{"role": "user", "content": "new"}])
    assert store.load_messages(b)[0]["content"] == "keep-me"


def test_finish_session_records_status_and_summary(store):
    sid = store.create_session("/p", "m", "t")
    store.finish_session(sid, "done", "all good")
    row = store.list_sessions()[0]
    assert row["status"] == "done" and row["summary"] == "all good"


@pytest.mark.parametrize("status", ["done", "error", "max_steps", "interrupted"])
def test_finish_session_accepts_each_status(store, status):
    sid = store.create_session("/p", "m", "t")
    store.finish_session(sid, status, "s")
    assert store.list_sessions()[0]["status"] == status


def test_delete_session_by_id_and_name(store):
    a = store.create_session("/p", "m", "t", name="named")
    store.append_message(a, 0, {"role": "user", "content": "x"})
    assert store.delete_session("named") is True
    assert store.session_exists(a) is False
    assert store.load_messages(a) == []  # history went with it
    assert store.delete_session("named") is False  # already gone


def test_delete_session_unknown_returns_false(store):
    assert store.delete_session("nope") is False


def test_list_sessions_orders_by_recency_and_honors_limit(store):
    ids = [store.create_session("/p", "m", f"t{i}") for i in range(3)]
    store.finish_session(ids[0], "done", "s")  # bumps updated_at -> most recent
    assert store.list_sessions()[0]["id"] == ids[0]
    assert len(store.list_sessions(limit=2)) == 2


def test_list_sessions_empty(store):
    assert store.list_sessions() == []


def test_list_sessions_exposes_expected_columns(store):
    store.create_session("/proj", "qwen", "the task", name="nm")
    row = store.list_sessions()[0]
    assert {"id", "created_at", "updated_at", "project_root",
            "model", "task", "status", "summary", "name"} <= set(row)
    assert row["project_root"] == "/proj" and row["model"] == "qwen"


# ---------------- tool_call_id ----------------

def test_tool_call_id_round_trips(store):
    """A tool result has to name the call it answers, or replaying a resumed
    history sends role="tool" messages a strict OpenAI-compatible server
    rejects with a 400."""
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1_0", "type": "function",
         "function": {"name": "read_file", "arguments": "{}"}}]})
    store.append_message(sid, 1, {"role": "tool", "content": "file body", "tool_call_id": "call_1_0"})

    loaded = store.load_messages(sid)
    assert loaded[0]["tool_calls"][0]["id"] == "call_1_0"
    assert loaded[1]["tool_call_id"] == "call_1_0"


def test_messages_without_a_tool_call_id_stay_clean(store):
    sid = store.create_session("/p", "m", "t")
    store.append_message(sid, 0, {"role": "user", "content": "hi"})
    assert "tool_call_id" not in store.load_messages(sid)[0]


def test_replace_messages_preserves_tool_call_ids(store):
    sid = store.create_session("/p", "m", "t")
    store.replace_messages(sid, [
        {"role": "system", "content": "s"},
        {"role": "tool", "content": "out", "tool_call_id": "abc"},
    ])
    assert store.load_messages(sid)[1]["tool_call_id"] == "abc"


def test_pre_existing_db_without_the_column_is_migrated(tmp_path):
    """Opening a database written by an older version must add the column
    rather than fail every read — same contract as the earlier `name` column."""
    path = str(tmp_path / "old.db")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                     "updated_at TEXT NOT NULL, project_root TEXT NOT NULL, model TEXT NOT NULL, "
                     "task TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'running', summary TEXT)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "session_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL, "
                     "content TEXT, tool_calls TEXT, created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO sessions VALUES ('old1', 'x', 'x', '/p', 'm', 't', 'done', 's')")
        conn.execute("INSERT INTO messages VALUES (1, 'old1', 0, 'user', 'legacy', NULL, 'x')")
        conn.commit()

    store = SessionStore(path)
    assert store.load_messages("old1") == [{"role": "user", "content": "legacy"}]
    store.append_message("old1", 1, {"role": "tool", "content": "new", "tool_call_id": "zzz"})
    assert store.load_messages("old1")[1]["tool_call_id"] == "zzz"


def test_writes_are_committed_not_just_left_open(tmp_path):
    """_connect() closes rather than commits, so every writer commits itself —
    a missed commit would silently discard the write on close."""
    path = str(tmp_path / "s.db")
    sid = SessionStore(path).create_session("/p", "m", "t", name="nm")
    store = SessionStore(path)
    store.append_message(sid, 0, {"role": "user", "content": "persisted?"})
    store.finish_session(sid, "done", "summary")

    fresh = SessionStore(path)   # separate connection: only committed data is visible
    assert fresh.resolve_session_id("nm") == sid
    assert fresh.load_messages(sid) == [{"role": "user", "content": "persisted?"}]
    assert fresh.list_sessions()[0]["status"] == "done"
    assert fresh.delete_session(sid) is True
    assert SessionStore(path).session_exists(sid) is False
