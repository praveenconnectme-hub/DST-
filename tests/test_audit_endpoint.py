"""
Integration tests for GET /api/audit endpoint.

Regression guard for the frontend bug where audit.js called .slice() on the
API response and got "rows.slice is not a function" — caused by the endpoint
returning a bare JSON array while the JS had an operator-precedence error
that left an unresolved Promise in `rows`.  These tests confirm:

  - The endpoint returns a bare list (NOT {rows:[...]}, NOT {data:[...]})
  - Each row has the keys the frontend template reads
  - Actions are recorded (LOGIN, GATE_APPROVED, PROMO_CREATED)
  - The limit parameter works
  - No auth is required (audit is public — Phase 1 decision, D-004 pattern)

Run:  python -m pytest tests/test_audit_endpoint.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from seed import ensure_users_seeded
from routes.auth   import router as auth_router
from routes.audit  import router as audit_router
from routes.gates  import router as gates_router
from routes.promotions import router as promotions_router


# ── Mini-app factory ──────────────────────────────────────────────────────────

def _make_app(db_path: str) -> FastAPI:
    apply_migration(db_path)
    repo = RepositoryFactory.create({"type": "sqlite", "db_path": db_path})
    ensure_users_seeded(repo)

    mini = FastAPI()
    mini.add_middleware(SessionMiddleware, secret_key="test-secret")
    mini.state.repo = repo
    mini.include_router(auth_router,       prefix="/api")
    mini.include_router(audit_router,      prefix="/api")
    mini.include_router(gates_router,      prefix="/api")
    mini.include_router(promotions_router, prefix="/api")
    return mini


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path):
    app = _make_app(str(tmp_path / "audit_ep.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def ch_client(tmp_path):
    """Client pre-logged-in as commercial_head_01."""
    app = _make_app(str(tmp_path / "audit_ch.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        yield c


# ── Shape tests (the regression guard) ───────────────────────────────────────

def test_audit_returns_bare_list_not_wrapped_object(client):
    """
    Frontend calls rows.slice() directly.  If the endpoint ever wraps the list
    in a dict (e.g. {rows:[...]}) this test catches it before the browser does.
    """
    resp = client.get("/api/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list), (
        f"Expected bare list, got {type(data).__name__}: {str(data)[:120]}"
    )


def test_audit_row_has_required_frontend_keys(client):
    """Every row the frontend template reads must be present."""
    client.post("/api/auth/login",
                json={"username": "planner_01", "password": "pl-demo-2024"})
    resp = client.get("/api/audit")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) > 0, "Expected at least one audit row after login"
    for row in rows:
        for key in ("audit_id", "timestamp", "actor", "action", "entity", "detail_json"):
            assert key in row, f"Key '{key}' missing from audit row: {row}"


# ── Content tests ─────────────────────────────────────────────────────────────

def test_login_creates_audit_row(client):
    client.post("/api/auth/login",
                json={"username": "planner_01", "password": "pl-demo-2024"})
    rows = client.get("/api/audit").json()
    login_rows = [r for r in rows if r["action"] == "LOGIN" and r["actor"] == "planner_01"]
    assert len(login_rows) >= 1


def test_gate_approval_creates_audit_row(ch_client):
    ch_client.post("/api/gates/G1/2024-W43/approve")
    rows = ch_client.get("/api/audit").json()
    gate_rows = [r for r in rows if r["action"] == "GATE_APPROVED"]
    assert len(gate_rows) >= 1
    assert gate_rows[0]["actor"] == "commercial_head_01"


def test_promo_create_creates_audit_row(tmp_path):
    app = _make_app(str(tmp_path / "promo_audit.db"))
    repo = app.state.repo
    # promotions_ledger.sku_id FK → sku_master
    repo.upsert("sku_master", [{"sku_id": "SKU_MID_01", "sku_name": "Mid TV",
                                 "product_tier": "mid", "base_cost_inr": 1000, "is_active": 1}])
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        c.post("/api/promotions", json={
            "event_name": "Test Sale", "sku_id": "SKU_MID_01",
            "start_week": "2024-W43", "end_week": "2024-W44",
            "offer_type": "price_discount",
        })
        rows = c.get("/api/audit").json()
    promo_rows = [r for r in rows if r["action"] == "PROMO_CREATED"]
    assert len(promo_rows) >= 1


def test_audit_empty_before_any_action(tmp_path):
    """Fresh DB with no logins → empty list (not an error)."""
    app = _make_app(str(tmp_path / "empty.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        resp = c.get("/api/audit")
    assert resp.status_code == 200
    assert resp.json() == []


def test_audit_limit_parameter(client):
    """limit= caps the number of rows returned."""
    # Generate several audit rows via repeated logins
    for _ in range(5):
        client.post("/api/auth/login",
                    json={"username": "planner_01", "password": "pl-demo-2024"})
    all_rows  = client.get("/api/audit").json()
    capped    = client.get("/api/audit?limit=2").json()
    assert isinstance(capped, list)
    assert len(capped) <= 2
    assert len(all_rows) > len(capped)
