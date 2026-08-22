"""SQLite-backed persistence for agent sessions and their full message
history, so a run can be resumed later (`--resume <id>`) or browsed
(`--list-sessions`) instead of always starting from a blank conversation.
"""

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: str):
    """`with _connect(path) as conn:` closes the connection on the way out.

    sqlite3's own connection context manager commits but never closes, so
    every call here used to leave the handle for the garbage collector (a
    pile of "unclosed database" ResourceWarnings under -W default). Closing
    instead of committing means each writer below commits explicitly."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return closing(conn)


class SessionStore:
    """One instance per agent process. Every message appended to the
    in-memory conversation is mirrored here in the same order, so
    `load_messages` reconstructs exactly what the model saw."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        with _connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    project_root TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    summary TEXT,
                    name TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    seq INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "name" not in existing_cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN name TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name)")
            message_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
            if "tool_call_id" not in message_cols:
                # Added after the fact: a tool result has to name the call it
                # answers when the history is replayed to the model, or a
                # resumed session sends role="tool" messages a strict
                # OpenAI-compatible server rejects.
                conn.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")
            conn.commit()

    def create_session(self, project_root: str, model: str, task: str, name: str = None) -> str:
        session_id = uuid.uuid4().hex[:8]
        now = _now()
        try:
            with _connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO sessions (id, created_at, updated_at, project_root, model, task, status, name) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
                    (session_id, now, now, project_root, model, task, name),
                )
                conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Session name {name!r} is already in use — pick another with --session-name.")
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def resolve_session_id(self, id_or_name: str) -> str:
        """Accept either a session id or a session --session-name and return
        the underlying id, or None if neither matches."""
        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE id = ? OR name = ?", (id_or_name, id_or_name),
            ).fetchone()
        return row["id"] if row else None

    def load_messages(self, session_id: str) -> list:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls, tool_call_id FROM messages "
                "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        messages = []
        for row in rows:
            msg = {"role": row["role"], "content": row["content"]}
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            messages.append(msg)
        return messages

    def append_message(self, session_id: str, seq: int, message: dict) -> None:
        tool_calls = message.get("tool_calls")
        now = _now()
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content, tool_calls, tool_call_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, seq, message.get("role", ""), message.get("content"),
                    json.dumps(tool_calls) if tool_calls else None,
                    message.get("tool_call_id"), now,
                ),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            conn.commit()

    def replace_messages(self, session_id: str, messages: list) -> None:
        """Overwrite a session's full message history (used by /compact):
        deletes the existing rows and re-inserts `messages` with fresh
        sequence numbers, so a later load_messages/resume sees the
        compacted version instead of the original."""
        now = _now()
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            for seq, message in enumerate(messages):
                tool_calls = message.get("tool_calls")
                conn.execute(
                    "INSERT INTO messages (session_id, seq, role, content, tool_calls, tool_call_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id, seq, message.get("role", ""), message.get("content"),
                        json.dumps(tool_calls) if tool_calls else None,
                        message.get("tool_call_id"), now,
                    ),
                )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            conn.commit()

    def finish_session(self, session_id: str, status: str, summary: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, summary = ?, updated_at = ? WHERE id = ?",
                (status, summary, _now(), session_id),
            )
            conn.commit()

    def delete_session(self, id_or_name: str) -> bool:
        """Delete a session and its full message history. Returns False if
        no session matches `id_or_name` (id or --session-name), True if
        deleted."""
        session_id = self.resolve_session_id(id_or_name)
        if session_id is None:
            return False
        with _connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        return True

    def list_sessions(self, limit: int = 20) -> list:
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, created_at, updated_at, project_root, model, task, status, summary, name "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
