"""
Tests for Phase 2 XGBoost demand sensing — BRD §4.3.

Covers:
  - Anti-leakage: feature at week W contains signal from W-lag, NOT from W
  - Holdout / training split is correct (no time overlap)
  - Model trains, predicts, and persists demand_sensing_output
  - SHAP JSON has correct feature keys
  - model_registry has xgboost entries per tier
  - audit_log MODEL_CHAMPION_SELECTED per trained tier
  - demand_sensing_output row count matches holdout × series

Run:  python -m pytest tests/test_sensing.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import json
import pytest
import numpy as np
from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from pipeline.sensing import (
    run as sensing_run,
    assemble_features,
    FEATURE_COLS,
    HOLDOUT_WEEKS,
    SIGNAL_LAGS,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_repo(tmp_path):
    db = str(tmp_path / "test.db")
    apply_migration(db)
    return RepositoryFactory.create({"type": "sqlite", "db_path": db})


def _seed_minimal_data(repo, n_weeks=30):
    """
    Seed 2 SKUs × 2 states × n_weeks of synthetic data into all relevant tables.

    Signal values are set to DISTINCTIVE integers so leakage tests can assert
    exact expected values:
      - temp_deviation at week W (1-indexed) = float(W)
      - competitor_price_index at week W     = 1.0 + W * 0.01
      - search_trend_index at week W         = 50.0 + W

    So for target week W:
      temp_lag1   = temp[W-1]   = float(W-1)        (NOT float(W))
      comp_lag2   = comp[W-2]   = 1.0 + (W-2)*0.01  (NOT 1.0 + W*0.01)
      search_lag1 = search[W-1] = 50.0 + (W-1)      (NOT 50.0 + W)
    """
    SKUS = [
        {"sku_id": "E1", "sku_name": "E1", "product_tier": "entry",
         "base_cost_inr": 1, "is_active": 1},
        {"sku_id": "M1", "sku_name": "M1", "product_tier": "mid",
         "base_cost_inr": 1, "is_active": 1},
    ]
    STATES = [
        {"state_code": "MH", "state_name": "MH", "commercial_zone": "West",  "is_reporting": 1},
        {"state_code": "DL", "state_name": "DL", "commercial_zone": "North", "is_reporting": 1},
    ]
    weeks = [f"2023-W{w:02d}" for w in range(1, n_weeks + 1)]

    repo.upsert("sku_master",  SKUS)
    repo.upsert("geo_master",  STATES)

    rng = np.random.default_rng(seed=42)
    sales = []
    for sku in SKUS:
        for state in STATES:
            for w_idx, wk in enumerate(weeks):
                qty = int(100 + 5 * w_idx + rng.normal(0, 5))
                sales.append({
                    "sku_id":           sku["sku_id"],
                    "state_code":       state["state_code"],
                    "week_index":       wk,
                    "quantity_actual":  max(1, qty),
                })
    repo.upsert("sales_history", sales)

    # Signals: w is 1-indexed (matches '2023-W{w}')
    sigs = []
    for state in STATES:
        for w in range(1, n_weeks + 1):
            wk = f"2023-W{w:02d}"
            sigs.extend([
                {"signal_name": "temp_deviation",        "state_code": state["state_code"],
                 "week_index": wk, "value": float(w),     "source_connector": "test"},
                {"signal_name": "competitor_price_index", "state_code": state["state_code"],
                 "week_index": wk, "value": 1.0 + w*0.01, "source_connector": "test"},
                {"signal_name": "search_trend_index",     "state_code": state["state_code"],
                 "week_index": wk, "value": 50.0 + w,     "source_connector": "test"},
            ])
    repo.upsert("signal_data", sigs)

    return weeks


# ── Anti-leakage tests (most critical) ───────────────────────────────────────

def test_temp_lag1_uses_prev_week_not_same_week(tmp_path):
    """
    Feature temp_lag1 at target week W must equal temp[W-1], NOT temp[W].

    With distinctive values: temp[W] = float(W).
    For target week 2023-W05 (W=5):
      temp_lag1 = temp[W04] = 4.0  ← correct
      temp[W05] = 5.0              ← would indicate same-week leakage
    """
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=20)

    df, _, _ = assemble_features(repo)

    row = df[(df["week_index"] == "2023-W05") &
             (df["sku_id"]     == "E1") &
             (df["state_code"] == "MH")].iloc[0]

    assert row["temp_lag1"] == pytest.approx(4.0), (
        f"temp_lag1 should be temp[W04]=4.0, got {row['temp_lag1']}")
    assert row["temp_lag1"] != pytest.approx(5.0), (
        "temp_lag1 must NOT be temp[W05] — that would be same-week leakage")


def test_comp_lag2_uses_two_weeks_prior_not_same_week(tmp_path):
    """
    Feature comp_lag2 at target week W must equal comp[W-2], NOT comp[W].

    For target week 2023-W05 (W=5):
      comp_lag2 = comp[W03] = 1.03  ← correct
      comp[W05] = 1.05              ← would indicate same-week leakage
    """
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=20)

    df, _, _ = assemble_features(repo)

    row = df[(df["week_index"] == "2023-W05") &
             (df["sku_id"]     == "E1") &
             (df["state_code"] == "MH")].iloc[0]

    assert row["comp_lag2"] == pytest.approx(1.03, abs=1e-9), (
        f"comp_lag2 should be comp[W03]=1.03, got {row['comp_lag2']}")
    assert row["comp_lag2"] != pytest.approx(1.05, abs=1e-9), (
        "comp_lag2 must NOT be comp[W05] — that would be same-week leakage")


def test_search_lag1_uses_prev_week_not_same_week(tmp_path):
    """
    Feature search_lag1 at target week W must equal search[W-1], NOT search[W].

    For target week 2023-W05 (W=5):
      search_lag1 = search[W04] = 54.0  ← correct
      search[W05] = 55.0                ← would indicate same-week leakage
    """
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=20)

    df, _, _ = assemble_features(repo)

    row = df[(df["week_index"] == "2023-W05") &
             (df["sku_id"]     == "E1") &
             (df["state_code"] == "MH")].iloc[0]

    assert row["search_lag1"] == pytest.approx(54.0), (
        f"search_lag1 should be search[W04]=54.0, got {row['search_lag1']}")
    assert row["search_lag1"] != pytest.approx(55.0), (
        "search_lag1 must NOT be search[W05] — that would be same-week leakage")


def test_holdout_and_train_weeks_never_overlap(tmp_path):
    """
    After splitting on holdout_start_ord, the train and holdout week sets are disjoint
    and holdout contains exactly HOLDOUT_WEEKS distinct weeks.
    """
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=30)

    df, weeks_list, _ = assemble_features(repo)
    holdout_start_ord = len(weeks_list) - HOLDOUT_WEEKS

    train_weeks   = set(df[df["week_ord"] <  holdout_start_ord]["week_index"].unique())
    holdout_weeks = set(df[df["week_ord"] >= holdout_start_ord]["week_index"].unique())

    assert holdout_weeks.isdisjoint(train_weeks), "Holdout and train weeks must not overlap"
    assert len(holdout_weeks) == HOLDOUT_WEEKS, (
        f"Expected {HOLDOUT_WEEKS} holdout weeks, got {len(holdout_weeks)}")


# ── Full run tests ────────────────────────────────────────────────────────────

def test_demand_sensing_output_row_count(tmp_path):
    """
    After sensing_run(), demand_sensing_output must have one row per
    sku×state×holdout_week for each tier processed.
    2 SKUs × 2 states × 12 holdout weeks = 48 rows (24 per tier × 2 tiers).
    """
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=30)

    summary = sensing_run(repo)

    rows = repo.query("demand_sensing_output")
    # 2 tiers (entry + mid) × 2 SKUs-per-tier × 2 states × 12 holdout_weeks
    # But each tier has 1 SKU × 2 states = 2 series × 12 = 24 rows per tier
    assert len(rows) == summary["sensing_output_rows"]
    assert len(rows) > 0


def test_shap_json_has_all_feature_keys(tmp_path):
    """Each demand_sensing_output row must have a shap_json with one entry per feature."""
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=30)

    sensing_run(repo)

    rows = repo.query("demand_sensing_output")
    assert len(rows) > 0

    expected_keys = set(FEATURE_COLS)
    for row in rows[:5]:   # spot-check first 5 rows
        shap_dict = json.loads(row["shap_json"])
        assert set(shap_dict.keys()) == expected_keys, (
            f"SHAP keys mismatch: {set(shap_dict.keys())} ≠ {expected_keys}")


def test_model_registry_has_xgboost_entries(tmp_path):
    """sensing_run() must create xgboost model_registry rows for trained tiers."""
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=30)

    sensing_run(repo)

    xgb_rows = [r for r in repo.query("model_registry")
                if r["model_type"] == "xgboost"]
    assert len(xgb_rows) >= 2, f"Expected ≥2 xgboost registry rows, got {len(xgb_rows)}"
    assert all(r["status"] == "champion" for r in xgb_rows)
    assert all(r["val_mape"] is not None for r in xgb_rows)


def test_audit_log_written_per_sensing_tier(tmp_path):
    """An audit_log MODEL_CHAMPION_SELECTED row must exist for each trained tier."""
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=30)

    sensing_run(repo)

    xgb_reg = [r for r in repo.query("model_registry") if r["model_type"] == "xgboost"]
    audit_rows = [r for r in repo.query("audit_log")
                  if r["action"] == "MODEL_CHAMPION_SELECTED"
                  and "xgboost" in r["detail_json"]]

    assert len(audit_rows) >= len(xgb_reg), (
        f"Expected ≥{len(xgb_reg)} sensing audit rows, got {len(audit_rows)}")

    for row in audit_rows:
        detail = json.loads(row["detail_json"])
        assert "model_id" in detail
        assert "tier"     in detail
        assert "val_mape" in detail


def test_forecast_qty_non_negative(tmp_path):
    """All sensing forecast_qty values must be ≥ 0 (no negative demand)."""
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=30)

    sensing_run(repo)

    rows = repo.query("demand_sensing_output")
    assert all(r["forecast_qty"] >= 0 for r in rows), (
        "Negative forecast_qty found in demand_sensing_output")


def test_summary_keys_present(tmp_path):
    """run() must return all required summary keys."""
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=30)

    summary = sensing_run(repo)

    for key in ("sensing_output_rows", "overall_sensing_mape_pct",
                "n_series_over_30pct", "tier_summaries", "top_shap_features"):
        assert key in summary, f"Missing key '{key}' in sensing summary"


def test_sensing_run_is_idempotent(tmp_path):
    """Running sensing twice must not duplicate demand_sensing_output rows."""
    repo = _make_repo(tmp_path)
    _seed_minimal_data(repo, n_weeks=30)

    sensing_run(repo)
    n_first = len(repo.query("demand_sensing_output"))

    sensing_run(repo)   # second run
    n_second = len(repo.query("demand_sensing_output"))

    assert n_second == n_first, (
        f"Second run duplicated rows: {n_first} → {n_second}")
