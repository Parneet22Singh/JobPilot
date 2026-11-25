"""SQLite persistence: jobs (+embeddings), chat memory, parsed profiles."""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


def job_text(job: dict) -> str:
    tags = " ".join(job.get("tags") or [])
    return f"{job['title']} at {job.get('company', '')}. {job.get('location', '')}. {tags}. {job.get('description', '')}"[
        :4000
    ]


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY, title TEXT, company TEXT, location TEXT,
          remote INTEGER DEFAULT 0, url TEXT, tags TEXT, description TEXT,
          content_hash TEXT, embedding BLOB, updated_at REAL,
          source TEXT DEFAULT '', posted_at REAL, last_seen REAL);
        CREATE TABLE IF NOT EXISTS messages(
          id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, ts REAL);
        CREATE TABLE IF NOT EXISTS profiles(session_id TEXT PRIMARY KEY, profile TEXT);
        """)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(jobs)")}
        for name, typ in (
            ("source", "TEXT DEFAULT ''"),
            ("posted_at", "REAL"),
            ("last_seen", "REAL"),
        ):
            if name not in cols:
                self.db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {typ}")
        self.db.commit()
        self.version = 0

    def upsert_job(self, job: dict) -> bool:
        """Insert/update a job. Always refreshes last_seen. Embedding is cleared only if content changed.
        Returns True if the job is new or its content changed."""
        h = hashlib.sha256(job_text(job).encode()).hexdigest()
        now = time.time()
        with self.lock:
            row = self.db.execute(
                "SELECT content_hash FROM jobs WHERE id=?", (job["id"],)
            ).fetchone()
            if row and row["content_hash"] == h:
                self.db.execute(
                    "UPDATE jobs SET last_seen=? WHERE id=?", (now, job["id"])
                )
                self.db.commit()
                return False
            self.db.execute(
                """INSERT INTO jobs(id,title,company,location,remote,url,tags,description,content_hash,embedding,
                                    updated_at,source,posted_at,last_seen)
                   VALUES(?,?,?,?,?,?,?,?,?,NULL,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET title=excluded.title, company=excluded.company,
                   location=excluded.location, remote=excluded.remote, url=excluded.url, tags=excluded.tags,
                   description=excluded.description, content_hash=excluded.content_hash,
                   embedding=NULL, updated_at=excluded.updated_at, source=excluded.source,
                   posted_at=excluded.posted_at, last_seen=excluded.last_seen""",
                (
                    job["id"],
                    job["title"],
                    job.get("company", ""),
                    job.get("location", ""),
                    int(job.get("remote", False)),
                    job.get("url", ""),
                    json.dumps(job.get("tags") or []),
                    job.get("description", ""),
                    h,
                    now,
                    job.get("source", ""),
                    job.get("posted_at"),
                    now,
                ),
            )
            self.db.commit()
            self.version += 1
            return True

    def expire_missing(self, source: str, seen_ids: set[str]) -> int:
        """Delete jobs of `source` that are no longer in its latest full listing (i.e. taken down)."""
        with self.lock:
            ids = [
                r["id"]
                for r in self.db.execute(
                    "SELECT id FROM jobs WHERE source=?", (source,)
                )
            ]
            gone = [i for i in ids if i not in seen_ids]
            for i in gone:
                self.db.execute("DELETE FROM jobs WHERE id=?", (i,))
            self.db.commit()
            if gone:
                self.version += 1
            return len(gone)

    def prune_stale(self, days: float) -> int:
        """Delete jobs not seen in any refresh for `days` days."""
        with self.lock:
            cur = self.db.execute(
                "DELETE FROM jobs WHERE last_seen IS NULL OR last_seen < ?",
                (time.time() - days * 86400,),
            )
            self.db.commit()
            if cur.rowcount:
                self.version += 1
            return cur.rowcount

    def jobs_missing_embeddings(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM jobs WHERE embedding IS NULL").fetchall()
        return [self._row(r) for r in rows]

    def set_embeddings(self, ids: list[str], vecs: np.ndarray):
        with self.lock:
            for i, v in zip(ids, vecs):
                self.db.execute(
                    "UPDATE jobs SET embedding=? WHERE id=?",
                    (v.astype(np.float32).tobytes(), i),
                )
            self.db.commit()
            self.version += 1

    def all_jobs(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM jobs ORDER BY id").fetchall()
        return [self._row(r, with_emb=True) for r in rows]

    def get_job(self, job_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(r) if r else None

    @staticmethod
    def _row(r, with_emb=False) -> dict:
        d = {
            k: r[k]
            for k in ("id", "title", "company", "location", "url", "description")
        }
        d["remote"] = bool(r["remote"])
        d["tags"] = json.loads(r["tags"] or "[]")
        d["posted_at"] = r["posted_at"]
        if with_emb:
            d["embedding"] = (
                np.frombuffer(r["embedding"], dtype=np.float32)
                if r["embedding"]
                else None
            )
        return d

    def add_message(self, session_id: str, role: str, content: str):
        with self.lock:
            self.db.execute(
                "INSERT INTO messages(session_id,role,content,ts) VALUES(?,?,?,?)",
                (session_id, role, content, time.time()),
            )
            self.db.commit()

    def get_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self.db.execute(
            "SELECT role,content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def set_profile(self, session_id: str, profile: dict):
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO profiles(session_id,profile) VALUES(?,?)",
                (session_id, json.dumps(profile)),
            )
            self.db.commit()

    def get_profile(self, session_id: str) -> dict | None:
        r = self.db.execute(
            "SELECT profile FROM profiles WHERE session_id=?", (session_id,)
        ).fetchone()
        return json.loads(r["profile"]) if r else None
