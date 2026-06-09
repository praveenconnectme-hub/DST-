"""
Integration tests for P3-3 G1 Gate API — BRD §3, D-021.

Covers:
  - GET  /api/gates/G1/{cycle_id}         — any authenticated role may read
  - POST /api/gates/G1/{cycle_id}/approve — commercial_head ONLY

SECURITY:
  test_approve_wrong_role_returns_403
  test_approve_unauthenticated_returns_401

AUDIT ATOMICITY (load-bearing for G1 approval trail):
  test_approve_writes_gate_and_audit_atomically
  test_approve_creates_exactly_one_gate_approved_audit_row

IDEMPOTENCY (D-021):
  test_double_approve_is_safe_noop
  test_double_approve_produces_no_duplicate_audit_rows

CYCLE ISOLATION:
  test_approve_cycle_a_leaves_cycle_b_pending

Run:  python -m pytest tests/test_gates.py -v
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from seed import ensure_users_seeded
from routes.auth import router as auth_router
from routes.gates import router as gates_router


# ── Mini-app factory ──────────────────────────────────────────────────────────

def _make_app(db_path: str) -> FastAPI:
    apply_migration(db_path)
    repo = RepositoryFactory.create({"type": "sqlite", "db_path": db_path})
    ensure_users_seeded(repo)

    mini = FastAPI()
    mini.add_middleware(SessionMiddleware, secret_key="test-secret")
    mini.state.repo = repo
    mini.include_router(auth_router,  prefix="/api")
    mini.include_router(gates_router, prefix="/api")
    return mini


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def ch_client(tmp_path):
    """Client pre-logged-in as commercial_head_01."""
    app = _make_app(str(tmp_path / "gates_ch.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        yield c


@pytest.fixture()
def planner_client(tmp_path):
    """Client pre-logged-in as planner_01 (wrong role for approve)."""
    app = _make_app(str(tmp_path / "gates_pl.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "planner_01", "password": "pl-demo-2024"})
        yield c


@pytest.fixture()
def anon_client(tmp_path):
    """Client with no session (unauthenticated)."""
    app = _make_app(str(tmp_path / "gates_anon.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── GET /api/gates/G1/{cycle_id} ─────────────────────────────────────────────

def test_get_g1_status_returns_pending_for_new_cycle(ch_client):
    resp = ch_client.get("/api/gates/G1/2024-W43")
    assert resp.status_code == 200
    data = resp.json()
    assert data["gate_id"] == "G1"
    assert data["cycle_id"] == "2024-W43"
    assert data["status"] == "pending"
    assert data["approved_by"] is None


def test_get_g1_status_requires_auth(anon_client):
    resp = anon_client.get("/api/gates/G1/2024-W43")
    assert resp.status_code == 401


def test_get_g1_status_accessible_by_planner(planner_client):
    """Any authenticated role may read gate status (not just commercial_head)."""
    resp = planner_client.get("/api/gates/G1/2024-W43")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"


def test_get_g1_status_reflects_approved_after_approval(tmp_path):
    """GET returns 'approved' after a successful approve call."""
    app = _make_app(str(tmp_path / "get_reflect.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        c.post("/api/gates/G1/2024-W43/approve")
        resp = c.get("/api/gates/G1/2024-W43")
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"


# ── POST /api/gates/G1/{cycle_id}/approve — success ──────────────────────────

def test_approve_g1_success(ch_client):
    resp = ch_client.post("/api/gates/G1/2024-W43/approve")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "approved"
    assert data["gate_id"] == "G1"
    assert data["cycle_id"] == "2024-W43"
    assert data["approved_by"] == "commercial_head_01"
    assert data["approved_at"] is not None


def test_approve_g1_response_includes_approved_by(ch_client):
    resp = ch_client.post("/api/gates/G1/2024-W44/approve")
    assert resp.json()["approved_by"] == "commercial_head_01"


# ── SECURITY ──────────────────────────────────────────────────────────────────

def test_approve_wrong_role_returns_403(planner_client):
    resp = planner_client.post("/api/gates/G1/2024-W43/approve")
    assert resp.status_code == 403


def test_approve_sop_chair_returns_403(tmp_path):
    """sop_chair may not approve G1 (only commercial_head)."""
    app = _make_app(str(tmp_path / "sop_test.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "sop_chair_01", "password": "sop-demo-2024"})
        resp = c.post("/api/gates/G1/2024-W43/approve")
    assert resp.status_code == 403


def test_approve_sales_manager_returns_403(tmp_path):
    app = _make_app(str(tmp_path / "sm_test.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "sales_mgr_01", "password": "sm-demo-2024"})
        resp = c.post("/api/gates/G1/2024-W43/approve")
    assert resp.status_code == 403


def test_approve_unauthenticated_returns_401(anon_client):
    resp = anon_client.post("/api/gates/G1/2024-W43/approve")
    assert resp.status_code == 401


# ── AUDIT ATOMICITY (load-bearing for G1 approval trail) ─────────────────────

def test_approve_writes_gate_and_audit_atomically(tmp_path):
    """
    After approve, BOTH the gate_status row (status='approved') AND a
    GATE_APPROVED audit_log row must exist — committed in the same transaction.

    Structural guarantee: the route wraps set_gate_status() and the
    audit_log upsert in a single `with repo.transaction()` at txn_depth 1,
    which issues exactly one conn.commit(). This test proves both rows landed.
    """
    app = _make_app(str(tmp_path / "atomic_gate.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        resp = c.post("/api/gates/G1/2024-W43/approve")

    assert resp.status_code == 200
    repo = app.state.repo

    # Gate status row must exist and be 'approved'
    gate_rows = repo.query("gate_status",
                           filters={"gate_id": "G1", "cycle_id": "2024-W43"})
    assert len(gate_rows) == 1, "gate_status row missing after approve"
    assert gate_rows[0]["status"] == "approved"

    # GATE_APPROVED audit row must exist
    audit_rows = [
        r for r in repo.query("audit_log")
        if r["action"] == "GATE_APPROVED"
        and json.loads(r["detail_json"]).get("cycle_id") == "2024-W43"
    ]
    assert len(audit_rows) == 1, (
        "GATE_APPROVED audit_log row missing — "
        "gate_status and audit_log must be written in the same transaction"
    )


def test_approve_creates_exactly_one_gate_approved_audit_row(tmp_path):
    """
    Exactly one GATE_APPROVED audit row must be written per approval.
    (Double-approve idempotency test below confirms no second row on repeat call.)
    """
    app = _make_app(str(tmp_path / "one_audit.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        c.post("/api/gates/G1/2024-W43/approve")

    repo = app.state.repo
    gate_approved_rows = [
        r for r in repo.query("audit_log")
        if r["action"] == "GATE_APPROVED"
        and json.loads(r["detail_json"]).get("gate_id") == "G1"
    ]
    assert len(gate_approved_rows) == 1, (
        f"Expected exactly 1 GATE_APPROVED audit row, got {len(gate_approved_rows)}"
    )


def test_approve_audit_row_records_correct_actor(tmp_path):
    app = _make_app(str(tmp_path / "audit_actor.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        c.post("/api/gates/G1/2024-W43/approve")

    repo = app.state.repo
    row = next(
        r for r in repo.query("audit_log")
        if r["action"] == "GATE_APPROVED"
    )
    assert row["actor"] == "commercial_head_01"
    assert json.loads(row["detail_json"])["cycle_id"] == "2024-W43"


# ── IDEMPOTENCY (D-021) ───────────────────────────────────────────────────────

def test_double_approve_is_safe_noop(tmp_path):
    """
    Second approve call must return 200 with the same approved state.
    It must NOT raise an error or change the approval timestamp.
    """
    app = _make_app(str(tmp_path / "idem_test.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        first  = c.post("/api/gates/G1/2024-W43/approve")
        second = c.post("/api/gates/G1/2024-W43/approve")

    assert first.status_code  == 200
    assert second.status_code == 200
    assert first.json()["status"]  == "approved"
    assert second.json()["status"] == "approved"
    # Timestamps are preserved (no re-write on second call)
    assert first.json()["approved_at"] == second.json()["approved_at"]


def test_double_approve_produces_no_duplicate_audit_rows(tmp_path):
    """
    Second approve must produce NO additional GATE_APPROVED audit rows (D-021).
    One approval → exactly one GATE_APPROVED row, always.
    """
    app = _make_app(str(tmp_path / "idem_audit.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        c.post("/api/gates/G1/2024-W43/approve")
        c.post("/api/gates/G1/2024-W43/approve")   # second call — no-op

    repo = app.state.repo
    gate_approved = [
        r for r in repo.query("audit_log")
        if r["action"] == "GATE_APPROVED"
        and json.loads(r["detail_json"]).get("cycle_id") == "2024-W43"
    ]
    assert len(gate_approved) == 1, (
        f"Expected exactly 1 GATE_APPROVED row after double-approve, "
        f"got {len(gate_approved)} — idempotency guard (D-021) may be broken"
    )


# ── CYCLE ISOLATION ───────────────────────────────────────────────────────────

def test_approve_cycle_a_leaves_cycle_b_pending(tmp_path):
    """
    Approving cycle A must not affect cycle B — gate approval is scoped to
    exactly one cycle_id.
    """
    app = _make_app(str(tmp_path / "isolation.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        c.post("/api/gates/G1/2024-W43/approve")

        # Cycle B was never touched
        resp_b = c.get("/api/gates/G1/2024-W44")

    assert resp_b.status_code == 200
    assert resp_b.json()["status"] == "pending", (
        "Approving 2024-W43 should not affect 2024-W44 gate status"
    )


def test_each_cycle_has_independent_gate_state(tmp_path):
    """Both cycles can be approved independently without interfering."""
    app = _make_app(str(tmp_path / "independent.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "commercial_head_01", "password": "ch-demo-2024"})
        c.post("/api/gates/G1/2024-W43/approve")
        c.post("/api/gates/G1/2024-W44/approve")

        resp_a = c.get("/api/gates/G1/2024-W43")
        resp_b = c.get("/api/gates/G1/2024-W44")

    assert resp_a.json()["status"] == "approved"
    assert resp_b.json()["status"] == "approved"

    # Two separate gate_status rows, two separate GATE_APPROVED audit rows
    repo = app.state.repo
    gate_rows = repo.query("gate_status", filters={"gate_id": "G1"})
    assert len(gate_rows) == 2

    approved_audits = [
        r for r in repo.query("audit_log") if r["action"] == "GATE_APPROVED"
    ]
    assert len(approved_audits) == 2
