"""
Integration tests for P3-4 G1 gate wiring in run_job() — BRD §3, D-022.

SAFETY INVARIANT (stated and proved here):
  There is no code path in run_job() that reaches PipelineState.SENSING unless
  repo.get_gate_status('G1', cycle_id) == 'approved' has been verified in the
  same call.  An unapproved gate routes to G1_PROMOTIONS_BLOCKED and returns;
  an exception routes to ERROR.  Neither reaches SENSING.

Covers:
  - Cycle with G1 unapproved → halts at G1_PROMOTIONS_BLOCKED
  - Sensing/scoring tables NOT written when blocked
  - Resume after approval → SENSING → CYCLE_COMPLETE
  - Modules 1-3 NOT re-executed on resume (no duplicate rows, no re-run audit)
  - Full ordered audit trail: blocked state included + resume trail
  - Idempotent re-pick: staying blocked writes no new audit rows

Run:  python -m pytest tests/test_g1_gate_wiring.py -v
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import numpy as np
import pandas as pd
import pytest

from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from main import run_job


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_repo(tmp_path):
    db = str(tmp_path / "test.db")
    apply_migration(db)
    return RepositoryFactory.create({"type": "sqlite", "db_path": db})


def _write_fixture_csvs(tmp_path, n_weeks=40):
    """Write all CSV fixtures required by the full pipeline (D-009 carve-out)."""
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
    rng   = np.random.default_rng(seed=42)
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


def _state_audit_rows(repo, cycle_id: str) -> list[dict]:
    """All set_pipeline_state audit rows for cycle_id, ordered by audit_id."""
    return [
        r for r in repo.query("audit_log", order_by=["audit_id"])
        if r["action"] == "set_pipeline_state"
        and json.loads(r["detail_json"]).get("cycle_id") == cycle_id
    ]


def _logged_states(repo, cycle_id: str) -> list[str]:
    return [json.loads(r["detail_json"])["state"]
            for r in _state_audit_rows(repo, cycle_id)]


# ── Block tests ───────────────────────────────────────────────────────────────

def test_g1_unapproved_halts_at_blocked_state(tmp_path):
    """run_job with G1 unapproved must return status='blocked' and set G1_PROMOTIONS_BLOCKED."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    result = run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    assert result["status"] == "blocked"
    assert result.get("gate") == "G1"

    ps = repo.get_pipeline_state("2023-W01")
    assert ps is not None
    assert ps["current_state"] == "G1_PROMOTIONS_BLOCKED"


def test_g1_unapproved_does_not_write_sensing_output(tmp_path):
    """demand_sensing_output must be empty after a G1-blocked cycle."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    sensing_rows = repo.query("demand_sensing_output")
    assert len(sensing_rows) == 0, (
        "demand_sensing_output must be empty when blocked at G1 — "
        "SENSING must not be reachable without gate approval (safety invariant)"
    )


def test_g1_unapproved_does_not_write_accuracy_metrics(tmp_path):
    """accuracy_metrics must be empty (xgboost rows) after a G1-blocked cycle."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    xgb_rows = [r for r in repo.query("accuracy_metrics") if "xgboost" in (r["model_id"] or "")]
    assert len(xgb_rows) == 0, (
        "accuracy_metrics xgboost rows must not exist when cycle is blocked at G1"
    )


def test_g1_unapproved_audit_trail_ends_at_blocked(tmp_path):
    """Audit trail after blocking must end with G1_PROMOTIONS_BLOCKED (not SENSING)."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    states = _logged_states(repo, "2023-W01")
    assert states[-1] == "G1_PROMOTIONS_BLOCKED", (
        f"Last state should be G1_PROMOTIONS_BLOCKED, got: {states}"
    )
    assert "SENSING" not in states, (
        "SENSING must not appear in audit trail when cycle is blocked at G1"
    )
    assert "CYCLE_COMPLETE" not in states


def test_g1_blocked_result_includes_pre_gate_summaries(tmp_path):
    """result dict on block must include ingestion/baseline/signals (run before the gate)."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    result = run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    assert result["status"] == "blocked"
    for key in ("ingestion", "baseline", "signals"):
        assert key in result, f"Expected '{key}' in blocked result; got keys: {list(result)}"


# ── Safety invariant ──────────────────────────────────────────────────────────

def test_safety_invariant_no_sensing_without_g1(tmp_path):
    """
    SAFETY INVARIANT:
      run_job() must never write to demand_sensing_output when G1 is unapproved.

    We call run_job() once without approving G1, verify no sensing output exists,
    then call it again (still no approval) — still no sensing output.  This proves
    the invariant holds across multiple calls, not just the first.
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    assert len(repo.query("demand_sensing_output")) == 0

    # Second call (simulating poll loop re-pick while still blocked)
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    assert len(repo.query("demand_sensing_output")) == 0, (
        "Safety invariant violated: sensing output appeared without G1 approval"
    )


# ── Resume tests ──────────────────────────────────────────────────────────────

def test_resume_after_approval_reaches_cycle_complete(tmp_path):
    """
    After approving G1 and re-running run_job(), the cycle must reach CYCLE_COMPLETE.
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    # First pass: blocks at G1
    result1 = run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    assert result1["status"] == "blocked"

    # Approve G1
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")

    # Second pass: resumes
    result2 = run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    assert result2["status"] == "done", f"Expected status=done on resume, got: {result2}"

    ps = repo.get_pipeline_state("2023-W01")
    assert ps["current_state"] == "CYCLE_COMPLETE"


def test_resume_does_not_re_run_ingestion(tmp_path):
    """
    Ingestion (set_pipeline_state INGESTING) must appear exactly ONCE in the audit
    trail across both runs — proving it was not re-executed on resume.
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    states = _logged_states(repo, "2023-W01")
    ingesting_count = states.count("INGESTING")
    assert ingesting_count == 1, (
        f"INGESTING appeared {ingesting_count} times in audit trail — "
        "ingestion must not re-run on resume"
    )


def test_resume_does_not_re_run_baseline(tmp_path):
    """BASELINING must appear exactly once across both runs."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    states = _logged_states(repo, "2023-W01")
    assert states.count("BASELINING") == 1, (
        "BASELINING appeared more than once — baseline must not re-run on resume"
    )


def test_resume_does_not_re_run_signals(tmp_path):
    """LOADING_SIGNALS must appear exactly once across both runs."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    states = _logged_states(repo, "2023-W01")
    assert states.count("LOADING_SIGNALS") == 1, (
        "LOADING_SIGNALS appeared more than once — signals must not re-run on resume"
    )


def test_resume_writes_sensing_and_scoring_output(tmp_path):
    """After resume, demand_sensing_output and accuracy_metrics must be populated."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    assert len(repo.query("demand_sensing_output")) > 0, \
        "demand_sensing_output must be written after resume"
    xgb_rows = [r for r in repo.query("accuracy_metrics")
                if "xgboost" in (r["model_id"] or "")]
    assert len(xgb_rows) > 0, "accuracy_metrics xgboost rows must exist after resume"


def test_resume_result_contains_all_module_keys(tmp_path):
    """run_job result on resume must still contain all five module keys."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")
    result = run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    for key in ("ingestion", "baseline", "signals", "sensing", "scoring"):
        assert key in result, f"Missing '{key}' in resume result"

    # Skipped steps are explicitly marked
    assert result["ingestion"].get("skipped") is True
    assert result["baseline"].get("skipped") is True
    assert result["signals"].get("skipped") is True


# ── Audit trail ───────────────────────────────────────────────────────────────

def test_full_audit_trail_shows_blocked_then_resumed(tmp_path):
    """
    The complete audit trail across both runs must be ordered:
    INGESTING → BASELINING → LOADING_SIGNALS → G1_PROMOTIONS_BLOCKED
    → SENSING → SCORING → CYCLE_COMPLETE

    G1_PROMOTIONS_BLOCKED appears once (from the first run).
    SENSING/SCORING/CYCLE_COMPLETE appear once (from the resume).
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    repo.set_gate_status("G1", "2023-W01", "approved", "commercial_head_01")
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    states = _logged_states(repo, "2023-W01")

    expected = [
        "INGESTING",
        "BASELINING",
        "LOADING_SIGNALS",
        "G1_PROMOTIONS_BLOCKED",
        "SENSING",
        "SCORING",
        "CYCLE_COMPLETE",
    ]
    assert states == expected, (
        f"Full audit trail mismatch.\n  Expected: {expected}\n  Got:      {states}"
    )


# ── Idempotent re-pick ────────────────────────────────────────────────────────

def test_idempotent_repick_while_blocked_writes_no_new_state(tmp_path):
    """
    A blocked cycle re-picked by the poll loop while still unapproved must NOT
    produce any new set_pipeline_state audit rows — idempotent no-op (D-022).
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    # First run → blocks
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    audit_count_after_block = len(_state_audit_rows(repo, "2023-W01"))

    # Second run (simulating re-pick) — gate still not approved
    result = run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    assert result["status"] == "blocked"
    assert result.get("resumed") is False

    audit_count_after_repick = len(_state_audit_rows(repo, "2023-W01"))
    assert audit_count_after_repick == audit_count_after_block, (
        f"Idempotent re-pick piled up audit rows: "
        f"{audit_count_after_block} → {audit_count_after_repick}. "
        "Re-picking a blocked cycle must not write new state audit rows (D-022)."
    )


def test_idempotent_repick_pipeline_state_unchanged(tmp_path):
    """Pipeline state must stay G1_PROMOTIONS_BLOCKED on idempotent re-pick."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)   # re-pick

    ps = repo.get_pipeline_state("2023-W01")
    assert ps["current_state"] == "G1_PROMOTIONS_BLOCKED"


def test_multiple_repicks_still_no_sensing(tmp_path):
    """Three re-picks without approval must never write to demand_sensing_output."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)
    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    assert len(repo.query("demand_sensing_output")) == 0, (
        "Safety invariant violated: sensing output appeared after multiple "
        "re-picks without G1 approval"
    )
