"""
Tests for Phase 2 signal ingestion — BRD §4.4.

Covers:
  - All three signal names loaded into signal_data
  - Correct row count per signal
  - Summary dict keys and values
  - Values stored as floats
  - source_connector correct per signal
  - Idempotency (second run does not duplicate rows)
  - state_code and week_index preserved

Run:  python -m pytest tests/test_signals.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import pytest
import pandas as pd
from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from pipeline.signals import run as signals_run


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_repo(tmp_path):
    db = str(tmp_path / "test.db")
    apply_migration(db)
    return RepositoryFactory.create({"type": "sqlite", "db_path": db})


def _write_signal_files(data_dir, states=None, weeks=None):
    """Write minimal fixture signal CSV files; returns rows-per-signal count."""
    if states is None:
        states = ["MH", "DL"]
    if weeks is None:
        weeks = ["2023-W01", "2023-W02", "2023-W03"]

    weather_rows = []
    comp_rows    = []
    trend_rows   = []

    for sc in states:
        for i, wk in enumerate(weeks):
            weather_rows.append({"state_code": sc, "week_index": wk,
                                  "temp_deviation": round(i * 0.5, 3)})
            comp_rows.append({"state_code": sc, "week_index": wk,
                               "competitor_price_index": round(1.0 + i * 0.01, 4)})
            trend_rows.append({"state_code": sc, "week_index": wk,
                                "search_trend_index": round(50.0 + i, 2)})

    pd.DataFrame(weather_rows).to_csv(
        os.path.join(data_dir, "weather_data.csv"), index=False)
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(data_dir, "competitor_scrapes.csv"), index=False)
    pd.DataFrame(trend_rows).to_csv(
        os.path.join(data_dir, "google_trends_export.csv"), index=False)

    return len(states) * len(weeks)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_all_three_signal_names_present(tmp_path):
    """signal_data must contain all three signal names after run()."""
    repo = _make_repo(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    _write_signal_files(data_dir)

    signals_run(repo, data_dir)

    names = {r["signal_name"] for r in repo.query("signal_data")}
    assert "temp_deviation"         in names
    assert "competitor_price_index" in names
    assert "search_trend_index"     in names


def test_row_count_per_signal(tmp_path):
    """Each signal must have states × weeks rows in signal_data."""
    repo = _make_repo(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    rows_per_signal = _write_signal_files(
        data_dir, states=["MH", "DL", "KA"],
        weeks=["2023-W01", "2023-W02", "2023-W03", "2023-W04", "2023-W05"])

    signals_run(repo, data_dir)

    for sname in ("temp_deviation", "competitor_price_index", "search_trend_index"):
        rows = repo.query("signal_data", filters={"signal_name": sname})
        assert len(rows) == rows_per_signal, (
            f"{sname}: expected {rows_per_signal} rows, got {len(rows)}")


def test_run_returns_correct_summary(tmp_path):
    """run() must return signals_loaded=3 and total_rows=3×n."""
    repo = _make_repo(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    rows_per_signal = _write_signal_files(data_dir)

    summary = signals_run(repo, data_dir)

    assert summary["signals_loaded"] == 3
    assert summary["total_rows"] == 3 * rows_per_signal


def test_signal_values_are_numeric(tmp_path):
    """All stored values must be floats (not strings from read_csv_raw)."""
    repo = _make_repo(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    _write_signal_files(data_dir)

    signals_run(repo, data_dir)

    rows = repo.query("signal_data")
    assert all(isinstance(r["value"], float) for r in rows)


def test_source_connector_set_per_signal(tmp_path):
    """Each signal must carry the correct source_connector tag."""
    repo = _make_repo(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    _write_signal_files(data_dir)

    signals_run(repo, data_dir)

    expected = {
        "temp_deviation":         "synthetic_weather",
        "competitor_price_index": "synthetic_competitor",
        "search_trend_index":     "synthetic_trends",
    }
    for signal_name, expected_connector in expected.items():
        rows = repo.query("signal_data", filters={"signal_name": signal_name})
        assert all(r["source_connector"] == expected_connector for r in rows), (
            f"{signal_name}: wrong source_connector")


def test_run_is_idempotent(tmp_path):
    """Running signals.run() twice must not duplicate rows (upsert semantics)."""
    repo = _make_repo(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    rows_per_signal = _write_signal_files(data_dir)

    signals_run(repo, data_dir)
    signals_run(repo, data_dir)   # second run

    for sname in ("temp_deviation", "competitor_price_index", "search_trend_index"):
        rows = repo.query("signal_data", filters={"signal_name": sname})
        assert len(rows) == rows_per_signal, (
            f"{sname}: second run duplicated rows "
            f"(expected {rows_per_signal}, got {len(rows)})")


def test_state_code_preserved(tmp_path):
    """state_code values must match the input fixture exactly."""
    repo = _make_repo(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    states = ["MH", "DL"]
    _write_signal_files(data_dir, states=states)

    signals_run(repo, data_dir)

    stored = {r["state_code"]
              for r in repo.query("signal_data",
                                  filters={"signal_name": "temp_deviation"})}
    assert stored == set(states)


def test_week_index_preserved(tmp_path):
    """week_index strings must survive the round-trip through read_csv_raw → upsert."""
    repo = _make_repo(tmp_path)
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    weeks = ["2023-W01", "2023-W26", "2025-W52"]
    _write_signal_files(data_dir, weeks=weeks)

    signals_run(repo, data_dir)

    stored = {r["week_index"]
              for r in repo.query("signal_data",
                                  filters={"signal_name": "search_trend_index"})}
    assert stored == set(weeks)
