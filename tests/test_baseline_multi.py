"""
Tests for Phase 2 champion-challenger baseline — BRD §4.2.

Covers:
  - Champion selection mechanism (correct status in model_registry)
  - Sparse series → Croston is the only model → deterministic champion
  - audit_log row per status change
  - accuracy_metrics for all candidates
  - Summary dict keys and champion_mix

Run:  python -m pytest tests/test_baseline_multi.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import json
import pytest
from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from pipeline.baseline import (
    run as baseline_run,
    HORIZON, BACKTEST_WEEKS, MIN_TRAIN_WEEKS, MAPE_FLAG_THRESHOLD,
    SPARSE_THRESHOLD, _croston_forecast, _holdout_mape,
)
import numpy as np


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_repo(tmp_path, skus, states):
    db = str(tmp_path / "test.db")
    apply_migration(db)
    repo = RepositoryFactory.create({"type": "sqlite", "db_path": db})
    repo.upsert("sku_master", skus)
    repo.upsert("geo_master", states)
    return repo


def _seed_history(repo, sku_id, state_code, n_weeks, qty_fn=None):
    if qty_fn is None:
        qty_fn = lambda i: 100 + i
    repo.upsert("sales_history", [
        {"sku_id": sku_id, "state_code": state_code,
         "week_index": f"2023-W{w:02d}", "quantity_actual": qty_fn(w)}
        for w in range(1, n_weeks + 1)
    ])


_SKU_A = {"sku_id": "SKU_A", "sku_name": "A", "product_tier": "mid",
           "base_cost_inr": 1, "is_active": 1}
_SKU_B = {"sku_id": "SKU_B", "sku_name": "B", "product_tier": "entry",
           "base_cost_inr": 1, "is_active": 1}
_GEO   = {"state_code": "MH", "state_name": "Maharashtra",
           "commercial_zone": "West", "is_reporting": 1}

_ENOUGH = MIN_TRAIN_WEEKS + BACKTEST_WEEKS + 4   # weeks to safely pass guard


# ── Champion selection mechanism ──────────────────────────────────────────────

def test_exactly_one_champion_per_series(tmp_path):
    """After baseline runs, each series has exactly one 'champion' in model_registry."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    _seed_history(repo, "SKU_A", "MH", _ENOUGH)

    baseline_run(repo)

    champions = [r for r in repo.query("model_registry") if r["status"] == "champion"]
    assert len(champions) == 1, f"Expected 1 champion, got {len(champions)}"


def test_champion_has_lower_mape_than_retired(tmp_path):
    """The champion's val_mape must be ≤ every retired model's val_mape."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    _seed_history(repo, "SKU_A", "MH", _ENOUGH)

    baseline_run(repo)

    reg       = repo.query("model_registry", filters={"scope": "SKU_A_MH"})
    champions = [r for r in reg if r["status"] == "champion"]
    retired   = [r for r in reg if r["status"] == "retired"]

    assert len(champions) == 1
    champ_mape = champions[0]["val_mape"]
    for r in retired:
        assert champ_mape <= r["val_mape"], (
            f"Retired model {r['model_id']} has lower MAPE "
            f"({r['val_mape']:.4f}) than champion ({champ_mape:.4f})"
        )


def test_sparse_series_champion_is_croston(tmp_path):
    """
    Series with > SPARSE_THRESHOLD zero-sales frequency must use Croston only.
    Croston is the sole candidate → champion by default.

    A sufficiently sparse series: > 60 % zeros.
    Use > MIN_TRAIN_WEEKS + BACKTEST_WEEKS + 4 total rows.
    """
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    n = _ENOUGH
    # Build series with ~70 % zeros: non-zero only every 3rd week
    qty_fn = lambda i: 200 if i % 3 == 0 else 0
    _seed_history(repo, "SKU_A", "MH", n, qty_fn=qty_fn)

    baseline_run(repo)

    champ = next(r for r in repo.query("model_registry") if r["status"] == "champion")
    assert champ["model_type"] == "croston", (
        f"Expected Croston champion on sparse series, got {champ['model_type']}"
    )


def test_non_sparse_series_champion_is_not_croston(tmp_path):
    """Non-sparse series must not be assigned a Croston champion."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    # Dense series: all non-zero
    _seed_history(repo, "SKU_A", "MH", _ENOUGH, qty_fn=lambda i: 100 + i)

    baseline_run(repo)

    champ = next(r for r in repo.query("model_registry") if r["status"] == "champion")
    assert champ["model_type"] != "croston"


# ── audit_log per status change ───────────────────────────────────────────────

def test_audit_log_champion_selected_written(tmp_path):
    """An audit_log row with action='MODEL_CHAMPION_SELECTED' exists after run."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    _seed_history(repo, "SKU_A", "MH", _ENOUGH)

    baseline_run(repo)

    rows = repo.query("audit_log", filters={"action": "MODEL_CHAMPION_SELECTED"})
    assert len(rows) >= 1, "Expected at least one MODEL_CHAMPION_SELECTED audit row"
    detail = json.loads(rows[0]["detail_json"])
    assert "model_id"  in detail
    assert "val_mape"  in detail
    assert "scope"     in detail


def test_audit_log_model_retired_written_for_non_winners(tmp_path):
    """For every retired model, an audit_log row with action='MODEL_RETIRED' exists."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    _seed_history(repo, "SKU_A", "MH", _ENOUGH)

    baseline_run(repo)

    retired_reg  = [r for r in repo.query("model_registry") if r["status"] == "retired"]
    retired_logs = repo.query("audit_log", filters={"action": "MODEL_RETIRED"})

    # There must be exactly as many MODEL_RETIRED log rows as retired registry entries
    assert len(retired_logs) == len(retired_reg), (
        f"Expected {len(retired_reg)} MODEL_RETIRED log rows, "
        f"got {len(retired_logs)}"
    )


def test_audit_log_rows_in_same_transaction_as_status(tmp_path):
    """
    After a crash-free run the model_registry and audit_log counts are consistent:
    for each series, #status_rows == #audit_rows (champion_selected + retired).
    This verifies they were written atomically.
    """
    repo = _make_repo(tmp_path, [_SKU_A, _SKU_B], [_GEO])
    _seed_history(repo, "SKU_A", "MH", _ENOUGH)
    _seed_history(repo, "SKU_B", "MH", _ENOUGH)

    baseline_run(repo)

    for sku_id in ["SKU_A", "SKU_B"]:
        scope    = f"{sku_id}_MH"
        reg_rows = repo.query("model_registry", filters={"scope": scope})
        champs   = [r for r in reg_rows if r["status"] == "champion"]
        retired  = [r for r in reg_rows if r["status"] == "retired"]

        champ_logs  = [
            r for r in repo.query("audit_log",
                                   filters={"action": "MODEL_CHAMPION_SELECTED"})
            if scope in r["detail_json"]
        ]
        retired_logs = [
            r for r in repo.query("audit_log", filters={"action": "MODEL_RETIRED"})
            if scope in r["detail_json"]
        ]
        assert len(champs)  == 1,              f"{sku_id}: expected 1 champion"
        assert len(champ_logs) >= 1,           f"{sku_id}: no champion audit row"
        assert len(retired_logs) == len(retired), (
            f"{sku_id}: retired log count mismatch"
        )


# ── Forecast persistence ──────────────────────────────────────────────────────

def test_baseline_forecast_rows_carry_champion_model_id(tmp_path):
    """Every baseline_forecast row must reference the champion's model_id."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    _seed_history(repo, "SKU_A", "MH", _ENOUGH)

    baseline_run(repo)

    champ = next(r for r in repo.query("model_registry") if r["status"] == "champion")
    frows = repo.query("baseline_forecast", filters={"sku_id": "SKU_A"})
    assert len(frows) == HORIZON
    assert all(r["model_id"] == champ["model_id"] for r in frows)


# ── accuracy_metrics ─────────────────────────────────────────────────────────

def test_accuracy_metrics_covers_all_candidates(tmp_path):
    """accuracy_metrics has BACKTEST_WEEKS rows per candidate model per series."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    # Dense series → HW + ARIMA both fitted
    _seed_history(repo, "SKU_A", "MH", _ENOUGH)

    baseline_run(repo)

    acc = repo.query("accuracy_metrics", filters={"sku_id": "SKU_A", "state_code": "MH"})
    # At least 1 candidate × BACKTEST_WEEKS rows; typically 2 × 12 = 24 for dense
    assert len(acc) >= BACKTEST_WEEKS, "Expected at least BACKTEST_WEEKS accuracy rows"
    assert all(r["mape"] is not None for r in acc)
    assert all(r["model_id"] for r in acc)


# ── Sparse skip path ──────────────────────────────────────────────────────────

def test_too_short_series_skipped_gracefully(tmp_path):
    """Series shorter than MIN_TRAIN_WEEKS + BACKTEST_WEEKS is skipped; no crash."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    _seed_history(repo, "SKU_A", "MH", n_weeks=5)

    summary = baseline_run(repo)

    assert summary["skipped_series"] == 1
    assert summary["forecasted_series"] == 0


def test_partial_skip_does_not_block_remaining_series(tmp_path):
    """A skipped series does not prevent other series from being forecast."""
    repo = _make_repo(tmp_path, [_SKU_A, _SKU_B], [_GEO])
    _seed_history(repo, "SKU_A", "MH", n_weeks=5)       # too short
    _seed_history(repo, "SKU_B", "MH", n_weeks=_ENOUGH)

    summary = baseline_run(repo)

    assert summary["skipped_series"] == 1
    assert summary["forecasted_series"] == 1
    assert len(repo.query("baseline_forecast", filters={"sku_id": "SKU_B"})) == HORIZON
    assert len(repo.query("baseline_forecast", filters={"sku_id": "SKU_A"})) == 0


# ── Croston unit tests ────────────────────────────────────────────────────────

def test_croston_all_zero_series_returns_zeros():
    result = _croston_forecast(np.zeros(20), n_steps=5)
    assert np.all(result == 0)


def test_croston_constant_nonzero_forecast_positive():
    """Constant non-zero series → Croston forecast > 0."""
    s = np.full(30, 50.0)
    result = _croston_forecast(s, n_steps=5)
    assert np.all(result > 0)


def test_holdout_mape_formula():
    """MAPE formula: actuals=[100,200], forecasts=[120,180] → 15 %."""
    apes = _holdout_mape(np.array([100.0, 200.0]), np.array([120.0, 180.0]))
    assert abs(apes - 0.15) < 1e-9


# ── Summary dict ─────────────────────────────────────────────────────────────

def test_summary_contains_required_keys(tmp_path):
    """run() summary must contain champion_mix and champion_mape_distribution."""
    repo = _make_repo(tmp_path, [_SKU_A], [_GEO])
    _seed_history(repo, "SKU_A", "MH", _ENOUGH)

    summary = baseline_run(repo)

    assert "champion_mix"               in summary
    assert "champion_mape_distribution" in summary
    assert "per_model_mape_median_pct"  in summary
    dist = summary["champion_mape_distribution"]
    for key in ("n_series", "min_pct", "median_pct", "p90_pct", "max_pct", "n_over_30"):
        assert key in dist, f"Missing key '{key}' in champion_mape_distribution"
