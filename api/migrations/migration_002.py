"""
Migration 002 — Add 'blocked' to job_queue.status CHECK constraint.

migration_001 defined:
  status TEXT CHECK(status IN ('queued','running','done','error'))

'blocked' was omitted, which caused an IntegrityError when the worker
tried to park a G1-blocked cycle.  The job was left as 'running' and
never re-picked by the poll loop (which queries only 'queued' and
'blocked').

SQLite does not support ALTER TABLE...MODIFY COLUMN, so we use the
standard rename pattern: create new table → copy data → drop old → rename.

This migration is idempotent: it checks whether 'blocked' is already in
the constraint before doing any work.

IMPORTANT: Only this file (and SQLiteRepository / migration_001) may
use sqlite3 directly.
"""
import sqlite3
from pathlib import Path


def run(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        # Check whether the table exists and whether it already has 'blocked'
        cur = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='job_queue'"
        )
        row = cur.fetchone()
        if row is None:
            # migration_001 not yet applied — nothing to patch
            print("[migration-002] job_queue table not found — skipping")
            return

        if "'blocked'" in row[0]:
            print("[migration-002] job_queue already has 'blocked' — no-op")
            return

        # SQLite requires PRAGMA foreign_keys=OFF while we swap tables
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN")

        conn.execute("""
            CREATE TABLE job_queue_new (
                job_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id    TEXT NOT NULL,
                action      TEXT NOT NULL,
                status      TEXT CHECK(status IN ('queued','running','done','blocked','error'))
                            DEFAULT 'queued',
                created_at  TEXT,
                started_at  TEXT,
                finished_at TEXT,
                result_json TEXT
            )
        """)
        conn.execute("""
            INSERT INTO job_queue_new
                (job_id, cycle_id, action, status,
                 created_at, started_at, finished_at, result_json)
            SELECT
                job_id, cycle_id, action, status,
                created_at, started_at, finished_at, result_json
            FROM job_queue
        """)
        conn.execute("DROP TABLE job_queue")
        conn.execute("ALTER TABLE job_queue_new RENAME TO job_queue")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_status ON job_queue(status)"
        )

        conn.execute("COMMIT")
        conn.execute("PRAGMA foreign_keys=ON")
        print(f"[migration-002] job_queue.status CHECK updated in {db_path}")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/data/dst.db"
    run(path)
