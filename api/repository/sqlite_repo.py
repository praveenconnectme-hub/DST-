"""
SQLiteRepository — concrete implementation of AbstractRepository.

ONLY this file may use sqlite3.  All other modules call the repo interface.
"""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .abstract import AbstractRepository


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRepository(AbstractRepository):
    """Thread-safe SQLite backend.

    Uses a per-thread connection (check_same_thread=False + a threading.local
    approach) so that FastAPI async workers and the background worker can each
    hold their own connection without races.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()

    def _in_transaction(self) -> bool:
        """True when we are inside an explicit transaction() context."""
        return getattr(self._local, "txn_depth", 0) > 0

    def _maybe_commit(self) -> None:
        """Commit only when NOT inside an explicit transaction() block."""
        if not self._in_transaction():
            self._conn().commit()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            # Retry for up to 5 s on lock contention (api + worker share the DB).
            # Without this, concurrent access returns SQLITE_BUSY immediately.
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return self._local.conn

    def _cursor(self):
        return self._conn().cursor()

    @staticmethod
    def _row_to_dict(row) -> dict:
        if row is None:
            return None
        return dict(row)

    def _build_where(self, filters: dict) -> tuple[str, list]:
        """Return (WHERE clause, params list) for equality filters."""
        if not filters:
            return "", []
        clauses = [f'"{k}" = ?' for k in filters]
        return "WHERE " + " AND ".join(clauses), list(filters.values())

    def _append_audit(self, cursor, actor: str, action: str,
                      entity: str, detail: dict) -> None:
        """Insert an audit_log row using an already-open cursor (same transaction)."""
        cursor.execute(
            """INSERT INTO audit_log (timestamp, actor, action, entity, detail_json)
               VALUES (?, ?, ?, ?, ?)""",
            (_now_iso(), actor, action, entity, json.dumps(detail)),
        )

    # ── Generic CRUD ────────────────────────────────────────────────────────

    def get(self, entity: str, key: dict) -> dict | None:
        where, params = self._build_where(key)
        cur = self._cursor()
        cur.execute(f'SELECT * FROM "{entity}" {where} LIMIT 1', params)
        return self._row_to_dict(cur.fetchone())

    def query(self, entity: str, filters: dict = None,
               order_by: list[str] = None) -> list[dict]:
        where, params = self._build_where(filters)
        order = ""
        if order_by:
            order = "ORDER BY " + ", ".join(f'"{c}"' for c in order_by)
        cur = self._cursor()
        cur.execute(f'SELECT * FROM "{entity}" {where} {order}', params)
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def upsert(self, entity: str, rows: list[dict]) -> int:
        if not rows:
            return 0
        conn = self._conn()
        columns = list(rows[0].keys())
        placeholders = ", ".join("?" * len(columns))
        col_names = ", ".join(f'"{c}"' for c in columns)
        sql = f'INSERT OR REPLACE INTO "{entity}" ({col_names}) VALUES ({placeholders})'
        data = [tuple(r.get(c) for c in columns) for r in rows]
        cur = conn.cursor()
        cur.executemany(sql, data)
        self._maybe_commit()
        return cur.rowcount

    def delete(self, entity: str, filters: dict) -> int:
        where, params = self._build_where(filters)
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(f'DELETE FROM "{entity}" {where}', params)
        self._maybe_commit()
        return cur.rowcount

    # ── Bulk frame I/O ──────────────────────────────────────────────────────

    def read_frame(self, entity: str, filters: dict = None) -> pd.DataFrame:
        where, params = self._build_where(filters)
        sql = f'SELECT * FROM "{entity}" {where}'
        return pd.read_sql_query(sql, self._conn(), params=params if params else None)

    def write_frame(self, entity: str, df: pd.DataFrame, mode: str = "upsert") -> int:
        if df.empty:
            return 0
        conn = self._conn()
        if mode == "replace":
            df.to_sql(entity, conn, if_exists="replace", index=False)
        else:
            # upsert: INSERT OR REPLACE row by row via executemany
            columns = list(df.columns)
            col_names = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join("?" * len(columns))
            sql = f'INSERT OR REPLACE INTO "{entity}" ({col_names}) VALUES ({placeholders})'
            data = [tuple(row) for row in df.itertuples(index=False, name=None)]
            cur = conn.cursor()
            cur.executemany(sql, data)
            self._maybe_commit()
            return cur.rowcount
        self._maybe_commit()
        return len(df)

    # ── Transaction ─────────────────────────────────────────────────────────

    @contextmanager
    def transaction(self):
        conn = self._conn()
        depth = getattr(self._local, "txn_depth", 0)
        self._local.txn_depth = depth + 1
        try:
            yield conn
            if self._local.txn_depth == 1:
                conn.commit()
        except Exception:
            if self._local.txn_depth == 1:
                conn.rollback()
            raise
        finally:
            self._local.txn_depth = max(0, self._local.txn_depth - 1)

    # ── Gate operations ─────────────────────────────────────────────────────

    def get_gate_status(self, gate_id: str, cycle_id: str) -> str:
        row = self.get("gate_status", {"gate_id": gate_id, "cycle_id": cycle_id})
        return row["status"] if row else "pending"

    def set_gate_status(self, gate_id: str, cycle_id: str,
                        status: str, actor: str) -> None:
        conn = self._conn()
        with self.transaction():
            cur = conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO gate_status
                   (gate_id, cycle_id, status, approved_by, approved_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (gate_id, cycle_id, status, actor, _now_iso()),
            )
            self._append_audit(cur, actor, "set_gate_status", "gate_status",
                                {"gate_id": gate_id, "cycle_id": cycle_id, "status": status})

    # ── Pipeline state ───────────────────────────────────────────────────────

    def get_pipeline_state(self, cycle_id: str) -> dict | None:
        return self.get("pipeline_state", {"cycle_id": cycle_id})

    def set_pipeline_state(self, cycle_id: str, state: str,
                           meta: dict, actor: str) -> None:
        conn = self._conn()
        with self.transaction():
            cur = conn.cursor()
            cur.execute(
                """INSERT OR REPLACE INTO pipeline_state
                   (cycle_id, current_state, state_meta_json, updated_at, updated_by)
                   VALUES (?, ?, ?, ?, ?)""",
                (cycle_id, state, json.dumps(meta), _now_iso(), actor),
            )
            self._append_audit(cur, actor, "set_pipeline_state", "pipeline_state",
                                {"cycle_id": cycle_id, "state": state})

    # ── Human-facing import/export ───────────────────────────────────────────

    def import_csv(self, entity: str, path: str) -> int:
        """Load CSV into entity using pandas.  File I/O lives ONLY here."""
        df = pd.read_csv(path)
        return self.write_frame(entity, df, mode="upsert")

    def export_excel(self, entity: str, path: str, filters: dict = None) -> str:
        """Export entity to Excel.  File I/O lives ONLY here."""
        df = self.read_frame(entity, filters=filters)
        df.to_excel(path, index=False)
        return path

    def read_csv_raw(self, path: str) -> pd.DataFrame:
        """Read CSV as raw strings without persisting. File I/O lives ONLY here."""
        return pd.read_csv(path, dtype=str)
