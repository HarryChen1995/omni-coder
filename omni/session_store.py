"""SQLite-backed persistence for agent sessions and their full message
history, so a run can be resumed later (`--resume <id>`) or browsed
(`--list-sessions`) instead of always starting from a blank conversation.
"""

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone

from .llm_client import message_text


def _split_content(content):
    """(text, parts_json) for storage.

    Structured content — the multimodal shape, text plus image parts — is
    stored twice over: as JSON so a resume can restore it exactly, and
    flattened to its text so anything that just wants to read the
    conversation still can."""
    if isinstance(content, list):
        return message_text({"content": content}), json.dumps(content)
    return content, None


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
                    content_parts TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "name" not in existing_cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN name TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name)")
            message_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}
            if "content_parts" not in message_cols:
                # A message with an image is a *list* of content parts, which
                # doesn't fit a TEXT column. The parts go here as JSON and
                # `content` keeps the readable text, so /sessions and the
                # resumed-history panel stay legible while a resume restores
                # the message the model actually saw.
                conn.execute("ALTER TABLE messages ADD COLUMN content_parts TEXT")
            if "tool_call_id" not in message_cols:
                # Added after the fact: a tool result has to name the call it
                # answers when the history is replayed to the model, or a
                # resumed session sends role="tool" messages a strict
                # OpenAI-compatible server rejects.
                conn.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")
            if "meta" not in message_cols:
                # Everything the transcript showed beside a message that isn't
                # part of the message itself: how long the reply took and what
                # it cost, how long a tool call ran, whether it succeeded. The
                # model never sees this — load_messages drops it — but a resume
                # replays the old turns through the live renderers, and without
                # it those turns come back missing their timings.
                conn.execute("ALTER TABLE messages ADD COLUMN meta TEXT")
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
        """The conversation exactly as the model saw it — nothing else, since
        this is what gets sent straight back to the server on a resume."""
        return [message for message, _ in self.load_records(session_id)]

    def load_records(self, session_id: str) -> list:
        """(message, meta) for every stored message, in order.

        `meta` is whatever the transcript showed alongside the message — reply
        time and token counts, a tool call's duration and outcome — and is
        `{}` for messages stored before that was recorded."""
        with _connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls, tool_call_id, content_parts, meta FROM messages "
                "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        records = []
        for row in rows:
            msg = {"role": row["role"], "content": row["content"]}
            if row["content_parts"]:
                msg["content"] = json.loads(row["content_parts"])
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            records.append((msg, json.loads(row["meta"]) if row["meta"] else {}))
        return records

    def append_message(self, session_id: str, seq: int, message: dict, meta: dict = None) -> None:
        tool_calls = message.get("tool_calls")
        content, content_parts = _split_content(message.get("content"))
        now = _now()
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (session_id, seq, role, content, tool_calls, tool_call_id, "
                "content_parts, meta, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, seq, message.get("role", ""), content,
                    json.dumps(tool_calls) if tool_calls else None,
                    message.get("tool_call_id"), content_parts,
                    json.dumps(meta) if meta else None, now,
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
                content, content_parts = _split_content(message.get("content"))
                conn.execute(
                    "INSERT INTO messages (session_id, seq, role, content, tool_calls, tool_call_id, "
                    "content_parts, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id, seq, message.get("role", ""), content,
                        json.dumps(tool_calls) if tool_calls else None,
                        message.get("tool_call_id"), content_parts, now,
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
