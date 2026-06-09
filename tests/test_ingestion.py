"""
Tests for Module 1 Ingestion — BRD §4.1 validation rules.

Run:  python -m pytest tests/test_ingestion.py -v
"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import pandas as pd
import pytest
from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from pipeline.ingestion import _validate, _impute, _complete_week_range, run as ingest_run


@pytest.fixture
def repo(tmp_path):
    db = str(tmp_path / "test.db")
    apply_migration(db)
    repo = RepositoryFactory.create({"type": "sqlite", "db_path": db})
    # Seed master tables
    repo.upsert("sku_master", [
        {"sku_id": "SKU_A", "sku_name": "A", "product_tier": "mid", "base_cost_inr": 1, "is_active": 1},
    ])
    repo.upsert("geo_master", [
        {"state_code": "MH", "state_name": "Maharashtra", "commercial_zone": "West", "is_reporting": 1},
    ])
    return repo


# ── Validation ───────────────────────────────────────────────────────────────

def _make_raw(rows):
    return pd.DataFrame(rows).astype(str)


def test_valid_row_passes(repo):
    raw = _make_raw([
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": "100"},
    ])
    clean, quarantine = _validate(raw, {"SKU_A"}, {"MH"})
    assert len(clean) == 1
    assert len(quarantine) == 0


def test_unknown_sku_quarantined(repo):
    raw = _make_raw([
        {"sku_id": "UNKNOWN_SKU", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": "100"},
    ])
    clean, quarantine = _validate(raw, {"SKU_A"}, {"MH"})
    assert len(clean) == 0
    assert len(quarantine) == 1
    assert "unknown sku_id" in quarantine[0]["reason"]


def test_unknown_state_quarantined(repo):
    raw = _make_raw([
        {"sku_id": "SKU_A", "state_code": "ZZ", "week_index": "2023-W01", "quantity_actual": "100"},
    ])
    clean, quarantine = _validate(raw, {"SKU_A"}, {"MH"})
    assert len(quarantine) == 1
    assert "unknown state_code" in quarantine[0]["reason"]


def test_non_integer_quantity_quarantined(repo):
    raw = _make_raw([
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": "12.5"},
    ])
    clean, quarantine = _validate(raw, {"SKU_A"}, {"MH"})
    assert len(quarantine) == 1
    assert "non-integer" in quarantine[0]["reason"]


def test_negative_quantity_quarantined(repo):
    raw = _make_raw([
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": "-10"},
    ])
    clean, quarantine = _validate(raw, {"SKU_A"}, {"MH"})
    assert len(quarantine) == 1


def test_unparseable_quantity_quarantined(repo):
    raw = _make_raw([
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": "abc"},
    ])
    clean, quarantine = _validate(raw, {"SKU_A"}, {"MH"})
    assert len(quarantine) == 1


def test_mixed_valid_invalid(repo):
    raw = _make_raw([
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": "100"},
        {"sku_id": "BAD",   "state_code": "MH", "week_index": "2023-W02", "quantity_actual": "50"},
        {"sku_id": "SKU_A", "state_code": "ZZ", "week_index": "2023-W03", "quantity_actual": "75"},
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W04", "quantity_actual": "200"},
    ])
    clean, quarantine = _validate(raw, {"SKU_A"}, {"MH"})
    assert len(clean) == 2
    assert len(quarantine) == 2


# ── Imputation ────────────────────────────────────────────────────────────────

def test_no_imputation_needed(repo):
    raw = pd.DataFrame([
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": f"2023-W{w:02d}", "quantity_actual": 100}
        for w in range(1, 6)
    ])
    result, n = _impute(raw)
    assert n == 0
    assert len(result) == 5


def test_full_ingestion_run(repo, tmp_path):
    """End-to-end: CSV on disk → ingested into sales_history via repo."""
    sku_csv   = str(tmp_path / "sku_master.csv")
    geo_csv   = str(tmp_path / "geo_master.csv")
    sales_csv = str(tmp_path / "sales_history.csv")

    pd.DataFrame([
        {"sku_id": "SKU_A", "sku_name": "A", "product_tier": "mid",
         "base_cost_inr": 1, "is_active": 1}
    ]).to_csv(sku_csv, index=False)

    pd.DataFrame([
        {"state_code": "MH", "state_name": "Maharashtra",
         "commercial_zone": "West", "is_reporting": 1}
    ]).to_csv(geo_csv, index=False)

    pd.DataFrame([
        {"sku_id": "SKU_A", "state_code": "MH",
         "week_index": f"2023-W{w:02d}", "quantity_actual": 100 + w}
        for w in range(1, 11)
    ] + [
        # Bad row — should be quarantined
        {"sku_id": "BAD_SKU", "state_code": "MH",
         "week_index": "2023-W11", "quantity_actual": 50}
    ]).to_csv(sales_csv, index=False)

    summary = ingest_run(repo, str(tmp_path))

    assert summary["valid_rows"] == 10
    assert summary["quarantined_rows"] == 1
    rows = repo.query("sales_history")
    assert len(rows) == 10


# ── _complete_week_range ──────────────────────────────────────────────────────

def test_complete_week_range_fills_gap():
    weeks = _complete_week_range(["2023-W01", "2023-W04"])
    assert weeks == ["2023-W01", "2023-W02", "2023-W03", "2023-W04"]


def test_complete_week_range_single_week():
    assert _complete_week_range(["2023-W05"]) == ["2023-W05"]


def test_complete_week_range_empty():
    assert _complete_week_range([]) == []


# ── Imputation — interior gap (Rule 1) ───────────────────────────────────────

def test_impute_fills_single_bounded_interior_gap():
    """One missing week between two data points is linearly interpolated."""
    df = pd.DataFrame([
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": 100},
        # 2023-W02 absent
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W03", "quantity_actual": 200},
    ])
    result, n_interp = _impute(df)

    assert n_interp == 1
    assert len(result) == 3
    w02 = result[result["week_index"] == "2023-W02"].iloc[0]["quantity_actual"]
    assert w02 == 150   # midpoint of 100 and 200


def test_impute_fills_multiple_bounded_interior_gaps():
    """Two consecutive missing weeks are both interpolated monotonically."""
    df = pd.DataFrame([
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": 100},
        # W02, W03 absent
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W04", "quantity_actual": 190},
    ])
    result, n_interp = _impute(df)

    assert n_interp == 2
    assert len(result) == 4
    vals = result.sort_values("week_index")["quantity_actual"].tolist()
    assert vals[0] == 100
    assert vals[3] == 190
    # Interpolated values rise monotonically between 100 and 190
    assert vals[0] < vals[1] < vals[2] < vals[3]


# ── Imputation — trailing zeros (Rule 2) ─────────────────────────────────────

def test_impute_trailing_zeros_beyond_12_all_preserved():
    """14 trailing zeros are kept as-is; n_interp stays 0 (no gaps)."""
    rows = (
        [{"sku_id": "SKU_A", "state_code": "MH",
          "week_index": f"2023-W{w:02d}", "quantity_actual": 100}
         for w in range(1, 6)]
        + [{"sku_id": "SKU_A", "state_code": "MH",
            "week_index": f"2023-W{w:02d}", "quantity_actual": 0}
           for w in range(6, 20)]   # 14 trailing zeros
    )
    result, n_interp = _impute(pd.DataFrame(rows))

    assert n_interp == 0
    trailing = result[result["week_index"] >= "2023-W06"]
    assert len(trailing) == 14
    assert (trailing["quantity_actual"] == 0).all()


def test_impute_exactly_12_trailing_zeros_preserved():
    """Exactly 12 trailing zeros (boundary) are also preserved unchanged."""
    rows = (
        [{"sku_id": "SKU_A", "state_code": "MH",
          "week_index": f"2023-W{w:02d}", "quantity_actual": 50}
         for w in range(1, 6)]
        + [{"sku_id": "SKU_A", "state_code": "MH",
            "week_index": f"2023-W{w:02d}", "quantity_actual": 0}
           for w in range(6, 18)]   # exactly 12 trailing zeros
    )
    result, n_interp = _impute(pd.DataFrame(rows))

    assert n_interp == 0
    trailing = result[result["week_index"] >= "2023-W06"]
    assert len(trailing) == 12
    assert (trailing["quantity_actual"] == 0).all()


# ── Quarantine persistence ────────────────────────────────────────────────────

def _make_ingest_env(tmp_path, sales_rows: list[dict]):
    """Create a fresh repo + minimal CSVs and return (repo, data_dir)."""
    import json as _json
    db = str(tmp_path / "test.db")
    apply_migration(db)
    repo = RepositoryFactory.create({"type": "sqlite", "db_path": db})

    pd.DataFrame([{
        "sku_id": "SKU_A", "sku_name": "A",
        "product_tier": "mid", "base_cost_inr": 1, "is_active": 1,
    }]).to_csv(str(tmp_path / "sku_master.csv"), index=False)

    pd.DataFrame([{
        "state_code": "MH", "state_name": "Maharashtra",
        "commercial_zone": "West", "is_reporting": 1,
    }]).to_csv(str(tmp_path / "geo_master.csv"), index=False)

    pd.DataFrame(sales_rows).to_csv(str(tmp_path / "sales_history.csv"), index=False)

    return repo, str(tmp_path)


def test_non_integer_qty_quarantine_persisted_to_audit_log(tmp_path):
    """Non-integer quantity rows appear in audit_log with action='quarantine'."""
    import json as _json
    repo, data_dir = _make_ingest_env(tmp_path, [
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": 100},
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W02", "quantity_actual": "12.5"},
    ])
    summary = ingest_run(repo, data_dir)

    assert summary["quarantined_rows"] == 1
    entries = repo.query("audit_log", filters={"action": "quarantine"})
    assert len(entries) == 1
    detail = _json.loads(entries[0]["detail_json"])
    assert "non-integer" in detail["reason"]
    assert detail["state_code"] == "MH"


def test_unknown_geo_quarantine_persisted_to_audit_log(tmp_path):
    """Unknown state_code rows appear in audit_log with action='quarantine'."""
    import json as _json
    repo, data_dir = _make_ingest_env(tmp_path, [
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": 100},
        {"sku_id": "SKU_A", "state_code": "ZZ", "week_index": "2023-W02", "quantity_actual": 50},
    ])
    summary = ingest_run(repo, data_dir)

    assert summary["quarantined_rows"] == 1
    entries = repo.query("audit_log", filters={"action": "quarantine"})
    assert len(entries) == 1
    detail = _json.loads(entries[0]["detail_json"])
    assert "unknown state_code" in detail["reason"]
    assert detail["state_code"] == "ZZ"


def test_high_severity_notification_raised_on_quarantine(tmp_path):
    """HIGH_SEVERITY_NOTIFICATION is written to audit_log when any rows are quarantined."""
    import json as _json
    repo, data_dir = _make_ingest_env(tmp_path, [
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": "abc"},
    ])
    summary = ingest_run(repo, data_dir)

    assert summary["notifications_raised"] == 1
    notifs = repo.query("audit_log", filters={"action": "HIGH_SEVERITY_NOTIFICATION"})
    assert len(notifs) >= 1
    detail = _json.loads(notifs[0]["detail_json"])
    assert detail["quarantine_count"] >= 1


def test_run_does_not_abort_on_quarantine(tmp_path):
    """A quarantined row must not abort the run; valid rows (plus any gap-fill) persist.

    W01=100 (valid), W02=12.5 (quarantined non-integer), W03=200 (valid).
    After quarantine, W02 is a bounded interior gap → imputed to 150.
    Final sales_history must have 3 rows and the run must not raise.
    """
    repo, data_dir = _make_ingest_env(tmp_path, [
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W01", "quantity_actual": 100},
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W02", "quantity_actual": "12.5"},
        {"sku_id": "SKU_A", "state_code": "MH", "week_index": "2023-W03", "quantity_actual": 200},
    ])
    summary = ingest_run(repo, data_dir)

    assert summary["quarantined_rows"] == 1
    # W02 gap filled by interpolation → 3 rows persisted
    assert summary["valid_rows"] == 3
    assert summary["interpolated_weeks"] == 1
    persisted = repo.query("sales_history")
    assert len(persisted) == 3
    w02 = next(r for r in persisted if r["week_index"] == "2023-W02")
    assert w02["quantity_actual"] == 150   # interpolated, not the raw 12.5
