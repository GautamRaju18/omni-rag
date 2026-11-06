"""Chat history and query traces in SQLite (Phases 10 and 13, ADR-0019).

One row per turn holds the whole query lifecycle as JSON (question, strategy, rewritten
queries, every ranked list, rerank scores, the selected context, the answer, its citations
and per-component latency), plus a few flat columns for listing and analytics.

SQLite (stdlib ``sqlite3``) instead of PostgreSQL: the API is one process by design (the job
manager, ADR-0007), the data is small, and the 8 GB VM is shared with Neo4j and Ollama's host
process. A connection per call keeps it thread-safe for worker threads; WAL lets the
dashboard read while a turn is written. The SQL is plain enough to port to Postgres when the
API runs as several processes.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    title TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    answerable INTEGER NOT NULL,
    grounded INTEGER NOT NULL,
    refused_by TEXT,
    verdict TEXT,
    latency_ms REAL NOT NULL,
    record TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_by_conversation ON turns (conversation_id, created_at);
CREATE INDEX IF NOT EXISTS turns_by_time ON turns (created_at);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class ChatStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:  # commit / rollback
                yield db
        finally:
            db.close()

    # ---------------------------------------------------------------- writes
    def ensure_conversation(self, conversation_id: str | None, title: str) -> str:
        with self._connect() as db:
            if conversation_id:
                row = db.execute("SELECT id FROM conversations WHERE id = ?",
                                 (conversation_id,)).fetchone()  # fmt: skip
                if row is None:
                    raise LookupError(f"conversation {conversation_id} not found")
                return conversation_id
            new_id = f"conv_{uuid.uuid4().hex[:12]}"
            db.execute("INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
                       (new_id, _now(), title[:120]))  # fmt: skip
            return new_id

    def add_turn(self, conversation_id: str, record: dict[str, Any]) -> dict[str, Any]:
        answer = record["answer"]
        turn = {"id": f"turn_{uuid.uuid4().hex[:12]}", "conversation_id": conversation_id,
                "created_at": _now(), **record}  # fmt: skip
        with self._connect() as db:
            db.execute(
                "INSERT INTO turns (id, conversation_id, created_at, question, answer_text, "
                "answerable, grounded, refused_by, verdict, latency_ms, record) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (turn["id"], conversation_id, turn["created_at"], record["question"],
                 answer["text"], int(answer["answerable"]), int(answer["grounded"]),
                 answer.get("refused_by"), record.get("retrieval", {}).get("verdict"),
                 record["timings"]["total_ms"], json.dumps(turn, ensure_ascii=False)),
            )  # fmt: skip
        return turn

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as db:
            return db.execute("DELETE FROM conversations WHERE id = ?",
                              (conversation_id,)).rowcount > 0  # fmt: skip

    # ---------------------------------------------------------------- reads
    def conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT c.id, c.created_at, c.title, COUNT(t.id) AS turns, "
                "MAX(t.created_at) AS last_turn_at FROM conversations c "
                "LEFT JOIN turns t ON t.conversation_id = c.id GROUP BY c.id "
                "ORDER BY COALESCE(MAX(t.created_at), c.created_at) DESC LIMIT ?", (limit,),
            ).fetchall()  # fmt: skip
        return [dict(r) for r in rows]

    def conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            conv = db.execute("SELECT id, created_at, title FROM conversations WHERE id = ?",
                              (conversation_id,)).fetchone()  # fmt: skip
            if conv is None:
                return None
            turns = db.execute("SELECT record FROM turns WHERE conversation_id = ? "
                               "ORDER BY created_at", (conversation_id,)).fetchall()  # fmt: skip
        return {**dict(conv), "turns": [json.loads(t["record"]) for t in turns]}

    def turn(self, turn_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT record FROM turns WHERE id = ?", (turn_id,)).fetchone()
        return json.loads(row["record"]) if row else None

    def recent_turns(self, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT record FROM turns ORDER BY created_at DESC LIMIT ?",
                              (limit,)).fetchall()  # fmt: skip
        return [json.loads(r["record"]) for r in rows]
