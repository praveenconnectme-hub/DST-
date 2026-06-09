"""
Tests for AbstractRepository / SQLiteRepository — BRD §5.0.

Run:  cd "Demand Sensing Tower" && python -m pytest tests/ -v
"""
import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory


@pytest.fixture
def repo(tmp_path):
    db = str(tmp_path / "test.db")
    apply_migration(db)
    return RepositoryFactory.create({"type": "sqlite", "db_path": db})


# ── Factory ──────────────────────────────────────────────────────────────────

def test_factory_returns_sqlite_repo(repo):
    from repository.sqlite_repo import SQLiteRepository
    assert isinstance(repo, SQLiteRepository)


def test_factory_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown repository type"):
        RepositoryFactory.create({"type": "postgres"})


# ── CRUD ─────────────────────────────────────────────────────────────────────

def test_upsert_and_get(repo):
    repo.upsert("sku_master", [{
        "sku_id": "TST_01", "sku_name": "Test SKU",
        "product_tier": "entry", "base_cost_inr": 10000, "is_active": 1,
    }])
    row = repo.get("sku_master", {"sku_id": "TST_01"})
    assert row is not None
    assert row["sku_name"] == "Test SKU"


def test_get_missing_returns_none(repo):
    assert repo.get("sku_master", {"sku_id": "DOESNOTEXIST"}) is None


def test_query_with_filter(repo):
    repo.upsert("sku_master", [
        {"sku_id": "A", "sku_name": "A", "product_tier": "entry", "base_cost_inr": 1, "is_active": 1},
        {"sku_id": "B", "sku_name": "B", "product_tier": "mid",   "base_cost_inr": 2, "is_active": 1},
    ])
    results = repo.query("sku_master", filters={"product_tier": "entry"})
    assert len(results) == 1
    assert results[0]["sku_id"] == "A"


def test_upsert_replaces_existing(repo):
    repo.upsert("sku_master", [{"sku_id": "X", "sku_name": "Old", "product_tier": "mid",
                                 "base_cost_inr": 5, "is_active": 1}])
    repo.upsert("sku_master", [{"sku_id": "X", "sku_name": "New", "product_tier": "mid",
                                 "base_cost_inr": 5, "is_active": 1}])
    row = repo.get("sku_master", {"sku_id": "X"})
    assert row["sku_name"] == "New"


def test_delete(repo):
    repo.upsert("sku_master", [{"sku_id": "DEL", "sku_name": "Del",
                                 "product_tier": "entry", "base_cost_inr": 1, "is_active": 1}])
    repo.delete("sku_master", {"sku_id": "DEL"})
    assert repo.get("sku_master", {"sku_id": "DEL"}) is None


# ── read_frame / write_frame ─────────────────────────────────────────────────

def test_read_write_frame(repo):
    import pandas as pd
    repo.upsert("geo_master", [
        {"state_code": "MH", "state_name": "Maharashtra", "commercial_zone": "West", "is_reporting": 1},
        {"state_code": "DL", "state_name": "Delhi",       "commercial_zone": "North","is_reporting": 1},
    ])
    df = repo.read_frame("geo_master")
    assert len(df) == 2
    assert "state_code" in df.columns


def test_write_frame_upsert(repo):
    import pandas as pd
    repo.upsert("geo_master", [
        {"state_code": "MH", "state_name": "Maharashtra", "commercial_zone": "West", "is_reporting": 1},
    ])
    new_df = pd.DataFrame([
        {"state_code": "MH", "state_name": "Maharashtra Updated", "commercial_zone": "West", "is_reporting": 1},
    ])
    repo.write_frame("geo_master", new_df, mode="upsert")
    row = repo.get("geo_master", {"state_code": "MH"})
    assert row["state_name"] == "Maharashtra Updated"


# ── Transaction ──────────────────────────────────────────────────────────────

def test_transaction_commits(repo):
    with repo.transaction():
        repo.upsert("sku_master", [{"sku_id": "TXN", "sku_name": "Txn",
                                     "product_tier": "mid", "base_cost_inr": 1, "is_active": 1}])
    assert repo.get("sku_master", {"sku_id": "TXN"}) is not None


def test_transaction_rolls_back_on_error(repo):
    try:
        with repo.transaction():
            repo.upsert("sku_master", [{"sku_id": "ROLLBACK", "sku_name": "R",
                                         "product_tier": "mid", "base_cost_inr": 1, "is_active": 1}])
            raise RuntimeError("forced rollback")
    except RuntimeError:
        pass
    assert repo.get("sku_master", {"sku_id": "ROLLBACK"}) is None


# ── Gate operations ──────────────────────────────────────────────────────────

def test_get_gate_status_default_pending(repo):
    status = repo.get_gate_status("g1_promotions", "2026-W01")
    assert status == "pending"


def test_set_and_get_gate_status(repo):
    repo.set_gate_status("g1_promotions", "2026-W01", "approved", "test_user")
    assert repo.get_gate_status("g1_promotions", "2026-W01") == "approved"


def test_set_gate_status_writes_audit_log(repo):
    repo.set_gate_status("g2_consensus", "2026-W02", "blocked", "system")
    audit = repo.query("audit_log", filters={"action": "set_gate_status"})
    assert len(audit) >= 1
    detail = json.loads(audit[-1]["detail_json"])
    assert detail["gate_id"] == "g2_consensus"


# ── Pipeline state ────────────────────────────────────────────────────────────

def test_get_pipeline_state_missing_returns_none(repo):
    assert repo.get_pipeline_state("2999-W99") is None


def test_set_and_get_pipeline_state(repo):
    repo.set_pipeline_state("2026-W10", "PRE_SENSING", {"x": 1}, "system")
    state = repo.get_pipeline_state("2026-W10")
    assert state["current_state"] == "PRE_SENSING"


def test_set_pipeline_state_writes_audit_log(repo):
    repo.set_pipeline_state("2026-W11", "IDLE", {}, "system")
    audit = repo.query("audit_log", filters={"action": "set_pipeline_state"})
    assert len(audit) >= 1


# ── import_csv ───────────────────────────────────────────────────────────────

def test_import_csv(repo, tmp_path):
    # Writing a fixture CSV to test import_csv is the sole permitted use of
    # file I/O in tests.  The purpose is to provide input TO the repo layer,
    # not to bypass it.  See DECISIONS.md D-008.
    import pandas as pd
    csv_path = str(tmp_path / "geo.csv")
    pd.DataFrame([
        {"state_code": "KA", "state_name": "Karnataka",  "commercial_zone": "South", "is_reporting": 1},
        {"state_code": "TN", "state_name": "Tamil Nadu", "commercial_zone": "South", "is_reporting": 1},
    ]).to_csv(csv_path, index=False)

    count = repo.import_csv("geo_master", csv_path)
    assert count == 2
    assert repo.get("geo_master", {"state_code": "KA"}) is not None


# ── Schema completeness ───────────────────────────────────────────────────────

EXPECTED_TABLES = [
    "sku_master", "geo_master", "users", "sales_history",
    "model_registry", "baseline_forecast", "signal_data",
    "event_calendar", "demand_sensing_output", "promotions_ledger",
    "field_estimates", "sop_consensus_grid", "actuals",
    "accuracy_metrics", "drift_metrics", "pipeline_state",
    "gate_status", "job_queue", "audit_log",
]

def test_all_brd_tables_exist(repo):
    """Verify every §5.1 table was created by the migration.
    repo.query() raises if the table doesn't exist; empty list means it does."""
    missing = []
    for table in EXPECTED_TABLES:
        try:
            repo.query(table)
        except Exception as exc:
            if "no such table" in str(exc).lower():
                missing.append(table)
    assert missing == [], f"Missing tables: {missing}"
