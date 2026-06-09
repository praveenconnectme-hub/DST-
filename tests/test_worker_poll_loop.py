"""
Integration tests for the worker poll-loop path — exercises the full
job_queue write sequence that bypassed tests previously caught.

The regression these tests guard:

  migration_001 defined:
    job_queue.status CHECK(status IN ('queued','running','done','error'))

  'blocked' was absent.  When run_job() returned status='blocked', the
  poll loop called repo.upsert("job_queue", [{..., "status": "blocked"}]),
  which raised an IntegrityError caught silently by the outer except clause.
  The job was left as 'running' and never re-picked — pipeline stuck at
  G1_PROMOTIONS_BLOCKED forever.

  migration_002 adds 'blocked' to the constraint.  These tests confirm that:
    1. A queued job gets picked up, run, and its status written as 'blocked'
       (not 'running', not an IntegrityError-hidden 'running').
    2. After G1 approval, the blocked job is re-picked, resumes, and ends
       as 'done' with pipeline_state = CYCLE_COMPLETE.
    3. The full path from job insertion → status='done' is covered.

All tests use real job_queue rows — no direct run_job() calls that would
skip the status-write path.

Run:  python -m pytest tests/test_worker_poll_loop.py -v
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import numpy as np
import pandas as pd
import pytest

from migrations.migration_001 import run as apply_migration_001
from migrations.migration_002 import run as apply_migration_002
from repository.factory import RepositoryFactory
from main import _process_job


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_repo(tmp_path):
    db = str(tmp_path / "poll_test.db")
    apply_migration_001(db)
    apply_migration_002(db)
    return RepositoryFactory.create({"type": "sqlite", "db_path": db})


def _write_fixture_csvs(tmp_path, n_weeks=40):
    """Minimal CSV fixtures for a full pipeline run (D-009 carve-out)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    SKUS = [
        {"sku_id": "E1", "sku_name": "E1", "product_tier": "entry",
         "base_cost_inr": 1000, "is_active": 1},
        {"sku_id": "M1", "sku_name": "M1", "product_tier": "mid",
         "base_cost_inr": 2000, "is_active": 1},
    ]
    STATES = [
        {"state_code": "MH", "state_name": "Maharashtra",
         "commercial_zone": "West", "is_reporting": 1},
        {"state_code": "DL", "state_name": "Delhi",
         "commercial_zone": "North", "is_reporting": 1},
    ]
    pd.DataFrame(SKUS).to_csv(data_dir / "sku_master.csv", index=False)
    pd.DataFrame(STATES).to_csv(data_dir / "geo_master.csv", index=False)

    weeks = [f"2023-W{w:02d}" for w in range(1, n_weeks + 1)]
    rng   = np.random.default_rng(seed=7)
    holdout_start = n_weeks - 12

    sales_rows, actuals_rows = [], []
    for sku in SKUS:
        for state in STATES:
            for w_idx, wk in enumerate(weeks):
                qty = max(1, int(200 + 10 * w_idx + rng.normal(0, 10)))
                sales_rows.append({
                    "sku_id": sku["sku_id"], "state_code": state["state_code"],
                    "week_index": wk, "quantity_actual": qty,
                })
                if w_idx >= holdout_start:
                    actuals_rows.append({
                        "sku_id": sku["sku_id"], "state_code": state["state_code"],
                        "week_index": wk, "quantity_actual": qty,
                        "loaded_at": "2023-10-01T00:00:00+00:00",
                    })

    pd.DataFrame(sales_rows).to_csv(data_dir / "sales_history.csv", index=False)
    pd.DataFrame(actuals_rows).to_csv(data_dir / "actuals_holdout.csv", index=False)

    sig_rows = []
    for state in STATES:
        for w_idx, wk in enumerate(weeks):
            sig_rows.append({
                "state_code": state["state_code"], "week_index": wk,
                "temp_deviation": float(w_idx + 1),
                "competitor_price_index": 1.0 + (w_idx + 1) * 0.01,
                "search_trend_index": 50.0 + (w_idx + 1),
            })
    sig_df = pd.DataFrame(sig_rows)
    sig_df[["state_code", "week_index", "temp_deviation"]].to_csv(
        data_dir / "weather_data.csv", index=False)
    sig_df[["state_code", "week_index", "competitor_price_index"]].to_csv(
        data_dir / "competitor_scrapes.csv", index=False)
    sig_df[["state_code", "week_index", "search_trend_index"]].to_csv(
        data_dir / "google_trends_export.csv", index=False)

    return str(data_dir)


def _insert_queued_job(repo, cycle_id: str) -> dict:
    """Insert a job_queue row with status='queued' and return it."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    repo.upsert("job_queue", [{
        "cycle_id":   cycle_id,
        "action":     "run_pipeline",
        "status":     "queued",
        "created_at": now,
    }])
    rows = repo.query("job_queue", filters={"cycle_id": cycle_id},
                      order_by=["job_id"])
    return rows[-1]


def _get_job(repo, job_id: int) -> dict:
    rows = repo.query("job_queue", filters={"job_id": job_id})
    assert rows, f"job_id={job_id} not found in job_queue"
    return rows[0]


# ── Constraint tests (the regression guard) ───────────────────────────────────

def test_migration_002_allows_blocked_status(tmp_path):
    """
    Core regression guard: after migration_002, inserting status='blocked'
    must not raise an IntegrityError.
    """
    repo = _make_repo(tmp_path)
    # This would raise IntegrityError on the pre-migration_002 schema
    repo.upsert("job_queue", [{
        "cycle_id": "2023-W01", "action": "run_pipeline",
        "status": "blocked", "created_at": "2023-01-01T00:00:00+00:00",
    }])
    rows = repo.query("job_queue", filters={"status": "blocked"})
    assert len(rows) == 1
    assert rows[0]["status"] == "blocked"


def test_migration_002_is_idempotent(tmp_path):
    """Running migration_002 twice must not raise or duplicate data."""
    db = str(tmp_path / "idem.db")
    apply_migration_001(db)
    apply_migration_002(db)
    apply_migration_002(db)   # second run — must be a no-op
    repo = RepositoryFactory.create({"type": "sqlite", "db_path": db})
    # Schema still usable
    repo.upsert("job_queue", [{
        "cycle_id": "2023-W01", "action": "run_pipeline",
        "status": "blocked", "created_at": "2023-01-01T00:00:00+00:00",
    }])
    assert len(repo.query("job_queue")) == 1


# ── Full poll-loop path tests ─────────────────────────────────────────────────

def test_poll_loop_sets_job_blocked_when_g1_unapproved(tmp_path):
    """
    _process_job() on an unapproved G1 cycle must write job status='blocked'
    (not leave it stuck at 'running' due to IntegrityError).

    This is the exact bug scenario: pre-fix, this assertion would fail because
    the job would remain 'running' after the IntegrityError was swallowed.
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    job = _insert_queued_job(repo, "2023-W01")
    _process_job(repo, job, data_dir)

    refreshed = _get_job(repo, job["job_id"])
    assert refreshed["status"] == "blocked", (
        f"Expected job status='blocked' after G1 block, got '{refreshed['status']}'. "
        "This indicates migration_002 was not applied — 'blocked' still triggers "
        "IntegrityError which leaves the job stuck as 'running'."
    )


def test_poll_loop_blocked_job_is_repickable(tmp_path):
    """
    A job written as 'blocked' must be queryable via status='blocked' filter —
    confirming the poll loop can re-pick it on the next tick.
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    job = _insert_queued_job(repo, "2023-W01")
    _process_job(repo, job, data_dir)

    blocked_jobs = repo.query("job_queue", filters={"status": "blocked"})
    assert len(blocked_jobs) == 1, (
        "Job must appear in status='blocked' so the poll loop can re-pick it. "
        f"Got: {[r['status'] for r in repo.query('job_queue')]}"
    )


def test_poll_loop_full_resume_path_reaches_cycle_complete(tmp_path):
    """
    Full end-to-end poll-loop path:
      1. Insert queued job
      2. _process_job() → G1 unapproved → job='blocked', state=G1_PROMOTIONS_BLOCKED
      3. Approve G1
      4. Re-pick blocked job → _process_job() → job='done', state=CYCLE_COMPLETE

    This exercises BOTH the status-write path AND the resume logic in a single test.
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    # ── First pass: queued → blocked ──────────────────────────────────────
    job = _insert_queued_job(repo, "2023-W01")
    _process_job(repo, job, data_dir)

    job_after_block = _get_job(repo, job["job_id"])
    assert job_after_block["status"] == "blocked"
    ps = repo.get_pipeline_state("2023-W01")
    assert ps["current_state"] == "G1_PROMOTIONS_BLOCKED"

    # ── Approve G1 ────────────────────────────────────────────────────────
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")

    # ── Second pass: re-pick blocked job → done ───────────────────────────
    blocked_job = _get_job(repo, job["job_id"])
    assert blocked_job["status"] == "blocked"   # confirming re-pick precondition
    _process_job(repo, blocked_job, data_dir)

    job_after_resume = _get_job(repo, job["job_id"])
    assert job_after_resume["status"] == "done", (
        f"Expected job status='done' after G1 approval + resume, "
        f"got '{job_after_resume['status']}'"
    )

    ps_final = repo.get_pipeline_state("2023-W01")
    assert ps_final["current_state"] == "CYCLE_COMPLETE", (
        f"Expected CYCLE_COMPLETE, got '{ps_final['current_state']}'"
    )


def test_poll_loop_result_json_written_on_block(tmp_path):
    """result_json in job_queue must contain gate and status info when blocked."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    job = _insert_queued_job(repo, "2023-W01")
    _process_job(repo, job, data_dir)

    refreshed = _get_job(repo, job["job_id"])
    result = json.loads(refreshed["result_json"])
    assert result.get("gate") == "G1"


def test_poll_loop_result_json_written_on_done(tmp_path):
    """result_json must contain module summaries when job completes successfully."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    # Block then approve then resume
    job = _insert_queued_job(repo, "2023-W01")
    _process_job(repo, job, data_dir)
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")

    blocked_job = _get_job(repo, job["job_id"])
    _process_job(repo, blocked_job, data_dir)

    finished = _get_job(repo, job["job_id"])
    result = json.loads(finished["result_json"])
    for key in ("ingestion", "baseline", "signals", "sensing", "scoring"):
        assert key in result, f"Expected '{key}' in done result_json; got {list(result)}"


def test_poll_loop_no_sensing_without_approval(tmp_path):
    """
    Safety invariant held through the full poll-loop path:
    demand_sensing_output must be empty after the first _process_job() call
    (G1 unapproved).
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    job = _insert_queued_job(repo, "2023-W01")
    _process_job(repo, job, data_dir)

    assert len(repo.query("demand_sensing_output")) == 0, (
        "Safety invariant violated via poll-loop path: sensing output written "
        "before G1 approval"
    )
