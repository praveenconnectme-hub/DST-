"""
Integration tests for P3-6 sensing endpoint — BRD §4.3.

Covers:
  - GET /api/sensing?sku_id=...&state_code=...
      - 401 when unauthenticated
      - 404 when no sensing output exists
      - Correct response shape (sku_id, state_code, model_id, weeks)
      - Each week row has week_index and sensing_qty
      - Actuals populated from sales_history where available
  - GET /api/sensing/summary
      - 401 when unauthenticated
      - Returns overall_sensing_mape_pct and series_count (None/0 when no data)
      - Correct MAPE when accuracy_metrics rows exist

All reads go through the repository — no sqlite3 / direct SQL in route.

Run:  python -m pytest tests/test_sensing_endpoint.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from seed import ensure_users_seeded
from routes.auth    import router as auth_router
from routes.sensing import router as sensing_router


# ── Mini-app factory ──────────────────────────────────────────────────────────

def _make_app(db_path: str) -> FastAPI:
    apply_migration(db_path)
    repo = RepositoryFactory.create({"type": "sqlite", "db_path": db_path})
    ensure_users_seeded(repo)

    mini = FastAPI()
    mini.add_middleware(SessionMiddleware, secret_key="test-secret")
    mini.state.repo = repo
    mini.include_router(auth_router,    prefix="/api")
    mini.include_router(sensing_router, prefix="/api")
    return mini


def _seed_sensing_data(repo, sku_id="SKU_A", state_code="MH", n_weeks=4):
    """Insert minimal master + sensing + actuals data for a single series."""
    repo.upsert("sku_master", [{"sku_id": sku_id, "sku_name": sku_id,
                                 "product_tier": "mid", "base_cost_inr": 1000, "is_active": 1}])
    repo.upsert("geo_master", [{"state_code": state_code, "state_name": "Maharashtra",
                                 "commercial_zone": "West", "is_reporting": 1}])
    repo.upsert("model_registry", [{"model_id": "xgboost_mid_v1", "model_type": "xgboost",
                                     "scope": "mid", "status": "champion"}])

    weeks = [f"2024-W{w:02d}" for w in range(1, n_weeks + 1)]
    sensing_rows = [
        {"sku_id": sku_id, "state_code": state_code, "week_index": wk,
         "forecast_qty": 100.0 + i * 10, "model_id": "xgboost_mid_v1"}
        for i, wk in enumerate(weeks)
    ]
    repo.upsert("demand_sensing_output", sensing_rows)

    sales_rows = [
        {"sku_id": sku_id, "state_code": state_code, "week_index": wk,
         "quantity_actual": 105 + i * 10}
        for i, wk in enumerate(weeks)
    ]
    repo.upsert("sales_history", sales_rows)


def _seed_accuracy_metrics(repo, sku_id="SKU_A", state_code="MH"):
    """Insert xgboost accuracy_metrics rows for the summary endpoint."""
    repo.upsert("accuracy_metrics", [
        {"sku_id": sku_id, "state_code": state_code, "model_id": "xgboost_mid_v1",
         "week_index": f"2024-W{w:02d}", "mape": 0.15, "bias": 0.01}
        for w in range(1, 5)
    ])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def auth_client(tmp_path):
    """Client pre-logged-in as planner_01 (any authenticated role suffices)."""
    app = _make_app(str(tmp_path / "sensing_ep.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login", json={"username": "planner_01", "password": "pl-demo-2024"})
        yield c, app.state.repo


@pytest.fixture()
def anon_client(tmp_path):
    app = _make_app(str(tmp_path / "sensing_anon.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── GET /api/sensing — auth guard ────────────────────────────────────────────

def test_sensing_requires_auth(anon_client):
    resp = anon_client.get("/api/sensing?sku_id=SKU_A&state_code=MH")
    assert resp.status_code == 401


def test_sensing_summary_requires_auth(anon_client):
    resp = anon_client.get("/api/sensing/summary")
    assert resp.status_code == 401


# ── GET /api/sensing — no data ────────────────────────────────────────────────

def test_sensing_404_when_no_data(auth_client):
    c, _ = auth_client
    resp = c.get("/api/sensing?sku_id=UNKNOWN&state_code=MH")
    assert resp.status_code == 404


# ── GET /api/sensing — correct shape ─────────────────────────────────────────

def test_sensing_returns_correct_top_level_shape(auth_client):
    c, repo = auth_client
    _seed_sensing_data(repo)
    resp = c.get("/api/sensing?sku_id=SKU_A&state_code=MH")
    assert resp.status_code == 200
    data = resp.json()
    assert data["sku_id"]     == "SKU_A"
    assert data["state_code"] == "MH"
    assert "model_id" in data
    assert "weeks" in data


def test_sensing_weeks_have_required_fields(auth_client):
    c, repo = auth_client
    _seed_sensing_data(repo, n_weeks=4)
    data = c.get("/api/sensing?sku_id=SKU_A&state_code=MH").json()
    assert len(data["weeks"]) == 4
    for w in data["weeks"]:
        assert "week_index"  in w, "week_index missing from week row"
        assert "sensing_qty" in w, "sensing_qty missing from week row"


def test_sensing_weeks_ordered_by_week_index(auth_client):
    c, repo = auth_client
    _seed_sensing_data(repo, n_weeks=4)
    data = c.get("/api/sensing?sku_id=SKU_A&state_code=MH").json()
    week_indices = [w["week_index"] for w in data["weeks"]]
    assert week_indices == sorted(week_indices)


def test_sensing_actuals_populated_from_sales_history(auth_client):
    c, repo = auth_client
    _seed_sensing_data(repo, n_weeks=4)
    data = c.get("/api/sensing?sku_id=SKU_A&state_code=MH").json()
    # All weeks have actuals because sales_history is seeded for the same weeks
    for w in data["weeks"]:
        assert w["actual"] is not None, f"Expected actual for {w['week_index']}"


def test_sensing_qty_values_are_positive(auth_client):
    c, repo = auth_client
    _seed_sensing_data(repo, n_weeks=4)
    data = c.get("/api/sensing?sku_id=SKU_A&state_code=MH").json()
    for w in data["weeks"]:
        assert w["sensing_qty"] > 0


def test_sensing_model_id_references_xgboost(auth_client):
    c, repo = auth_client
    _seed_sensing_data(repo)
    data = c.get("/api/sensing?sku_id=SKU_A&state_code=MH").json()
    assert data["model_id"] is not None
    assert "xgboost" in data["model_id"]


# ── GET /api/sensing/summary ──────────────────────────────────────────────────

def test_sensing_summary_no_data_returns_none(auth_client):
    """Before any sensing run, summary returns None MAPE and 0 series."""
    c, _ = auth_client
    resp = c.get("/api/sensing/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_sensing_mape_pct"] is None
    assert data["series_count"] == 0


def test_sensing_summary_returns_mape_when_data_exists(auth_client):
    c, repo = auth_client
    _seed_sensing_data(repo)
    _seed_accuracy_metrics(repo)
    resp = c.get("/api/sensing/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_sensing_mape_pct"] is not None
    assert data["overall_sensing_mape_pct"] > 0
    assert data["series_count"] >= 1


def test_sensing_summary_mape_is_pct_not_fraction(auth_client):
    """Stored MAPE is a fraction (0.15); endpoint must return percentage (15.0)."""
    c, repo = auth_client
    _seed_sensing_data(repo)
    _seed_accuracy_metrics(repo)  # mape=0.15 per row
    data = c.get("/api/sensing/summary").json()
    # 0.15 stored → 15.0 returned
    assert data["overall_sensing_mape_pct"] == pytest.approx(15.0, abs=0.1)


def test_sensing_summary_excludes_non_xgboost_metrics(auth_client):
    """Baseline (holt_winters) accuracy rows must NOT be counted in the sensing MAPE."""
    c, repo = auth_client
    _seed_sensing_data(repo)
    # Register the baseline model so the FK constraint is satisfied
    repo.upsert("model_registry", [{"model_id": "hw_sku_a_mh_v1",
                                     "model_type": "holt_winters",
                                     "scope": "SKU_A×MH", "status": "champion"}])
    # Insert a baseline (non-xgboost) metric with a very different MAPE
    repo.upsert("accuracy_metrics", [
        {"sku_id": "SKU_A", "state_code": "MH", "model_id": "hw_sku_a_mh_v1",
         "week_index": "2024-W01", "mape": 0.99, "bias": 0.0}
    ])
    _seed_accuracy_metrics(repo)  # xgboost rows: mape=0.15
    data = c.get("/api/sensing/summary").json()
    # If baseline rows were included, MAPE would be skewed toward 0.99
    assert data["overall_sensing_mape_pct"] == pytest.approx(15.0, abs=0.5)
