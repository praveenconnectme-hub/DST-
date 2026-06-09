"""
Integration tests for Phase 2 pipeline orchestration — BRD §3, D-017.

Covers:
  - run_job advances pipeline_state through the full Phase 2 step sequence
  - One audit_log row per state transition, each in same transaction as state write
  - accuracy_metrics populated after scoring step
  - run_job result dict contains all five module summary keys

Run:  python -m pytest tests/test_pipeline_cycle.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import json
import numpy as np
import pandas as pd

from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from main import run_job


EXPECTED_STATES = [
    "INGESTING",
    "BASELINING",
    "LOADING_SIGNALS",
    "SENSING",
    "SCORING",
    "CYCLE_COMPLETE",
]


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

def _make_repo(tmp_path):
    db = str(tmp_path / "test.db")
    apply_migration(db)
    return RepositoryFactory.create({"type": "sqlite", "db_path": db})


def _write_fixture_csvs(tmp_path, n_weeks=40):
    """
    Write all CSV fixture files required by the full pipeline (D-009 carve-out).

    40 weeks × 2 SKUs × 2 states gives:
      baseline : 40 ≥ MIN_TRAIN_WEEKS(16) + BACKTEST_WEEKS(12) = 28 ✓
      sensing  : 24 valid training rows per tier (week_ord 4-27) > 20 ✓
                 12 holdout rows per series (week_ord 28-39) > 0 ✓
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()

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
    rng = np.random.default_rng(seed=99)

    holdout_start_idx = n_weeks - 12   # 0-based index of first holdout week

    sales_rows, actuals_rows = [], []
    for sku in SKUS:
        for state in STATES:
            for w_idx, wk in enumerate(weeks):
                qty = max(1, int(200 + 10 * w_idx + rng.normal(0, 10)))
                sales_rows.append({
                    "sku_id":          sku["sku_id"],
                    "state_code":      state["state_code"],
                    "week_index":      wk,
                    "quantity_actual": qty,
                })
                if w_idx >= holdout_start_idx:
                    actuals_rows.append({
                        "sku_id":          sku["sku_id"],
                        "state_code":      state["state_code"],
                        "week_index":      wk,
                        "quantity_actual": qty,
                        "loaded_at":       "2023-10-01T00:00:00+00:00",
                    })

    pd.DataFrame(sales_rows).to_csv(data_dir / "sales_history.csv", index=False)
    pd.DataFrame(actuals_rows).to_csv(data_dir / "actuals_holdout.csv", index=False)

    # Signal CSVs — distinctive values per state×week (lag tests depend on this)
    sig_rows = []
    for state in STATES:
        for w_idx, wk in enumerate(weeks):
            sig_rows.append({
                "state_code":             state["state_code"],
                "week_index":             wk,
                "temp_deviation":         float(w_idx + 1),
                "competitor_price_index": 1.0 + (w_idx + 1) * 0.01,
                "search_trend_index":     50.0 + (w_idx + 1),
            })
    sig_df = pd.DataFrame(sig_rows)
    sig_df[["state_code", "week_index", "temp_deviation"]].to_csv(
        data_dir / "weather_data.csv", index=False)
    sig_df[["state_code", "week_index", "competitor_price_index"]].to_csv(
        data_dir / "competitor_scrapes.csv", index=False)
    sig_df[["state_code", "week_index", "search_trend_index"]].to_csv(
        data_dir / "google_trends_export.csv", index=False)

    return str(data_dir)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_full_cycle_completes(tmp_path):
    """run_job must return status='done' and final pipeline_state must be CYCLE_COMPLETE."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)
    repo.set_gate_status("G1", "2023-W01", "approved", "test_setup")

    result = run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    assert result.get("status") == "done", f"Expected status=done, got: {result}"
    state = repo.get_pipeline_state("2023-W01")
    assert state is not None, "pipeline_state row not found for cycle"
    assert state["current_state"] == "CYCLE_COMPLETE"


def test_state_sequence_logged_in_order(tmp_path):
    """
    audit_log must record set_pipeline_state for every step in the expected order:
    INGESTING → BASELINING → LOADING_SIGNALS → SENSING → SCORING → CYCLE_COMPLETE
    (G1 gate check is silent when approved — no extra state transition.)
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)
    repo.set_gate_status("G1", "2023-W01", "approved", "test_setup")

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    state_audit = [
        r for r in repo.query("audit_log", order_by=["audit_id"])
        if r["action"] == "set_pipeline_state"
        and json.loads(r["detail_json"]).get("cycle_id") == "2023-W01"
    ]

    logged_states = [json.loads(r["detail_json"])["state"] for r in state_audit]

    assert logged_states == EXPECTED_STATES, (
        f"State sequence mismatch.\n  Expected: {EXPECTED_STATES}\n  Got:      {logged_states}"
    )


def test_one_audit_per_state_transition(tmp_path):
    """Exactly one set_pipeline_state audit row per state (6 total for one cycle)."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)
    repo.set_gate_status("G1", "2023-W01", "approved", "test_setup")

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    state_audit = [
        r for r in repo.query("audit_log")
        if r["action"] == "set_pipeline_state"
        and json.loads(r["detail_json"]).get("cycle_id") == "2023-W01"
    ]

    assert len(state_audit) == len(EXPECTED_STATES), (
        f"Expected {len(EXPECTED_STATES)} state-transition audit rows, "
        f"got {len(state_audit)}"
    )


def test_state_audit_atomicity_invariant(tmp_path):
    """
    For a completed cycle, every expected state must have a matching audit_log row —
    verifying that pipeline_state and audit_log are always written together (same
    transaction). No state should be in pipeline_state without a corresponding audit row.
    """
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)
    repo.set_gate_status("G1", "2023-W01", "approved", "test_setup")

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    logged_states = {
        json.loads(r["detail_json"])["state"]
        for r in repo.query("audit_log")
        if r["action"] == "set_pipeline_state"
        and json.loads(r["detail_json"]).get("cycle_id") == "2023-W01"
    }

    for expected in EXPECTED_STATES:
        assert expected in logged_states, (
            f"State '{expected}' has no matching audit_log row — "
            "pipeline_state and audit must be written atomically"
        )


def test_accuracy_metrics_populated_after_cycle(tmp_path):
    """After a full cycle, accuracy_metrics must contain rows for the XGBoost sensing output."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)
    repo.set_gate_status("G1", "2023-W01", "approved", "test_setup")

    run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    xgb_acc = [
        r for r in repo.query("accuracy_metrics")
        if "xgboost" in r["model_id"]
    ]

    assert len(xgb_acc) > 0, "No xgboost accuracy_metrics rows found after scoring"
    assert all(r["mape"] is not None for r in xgb_acc), "mape must not be None"
    assert all(r["mape"] >= 0.0 for r in xgb_acc), "Negative MAPE in accuracy_metrics"


def test_result_contains_all_module_keys(tmp_path):
    """run_job result must contain summaries from all five pipeline modules."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)
    repo.set_gate_status("G1", "2023-W01", "approved", "test_setup")

    result = run_job(repo, {"job_id": 1, "cycle_id": "2023-W01"}, data_dir)

    for key in ("ingestion", "baseline", "signals", "sensing", "scoring"):
        assert key in result, f"Missing '{key}' key in run_job result"
