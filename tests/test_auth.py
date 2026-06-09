"""
Integration tests for P3-1 auth (BRD §12, D-018).

Covers:
  - POST /api/auth/login  success / wrong password / unknown user
  - GET  /api/auth/me     with valid session / without session (401)
  - POST /api/auth/logout clears session
  - RBAC: correct role passes / wrong role → 403 / unauthenticated → 401

Uses an isolated mini-FastAPI app to avoid touching the real DB.

Run:  python -m pytest tests/test_auth.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from seed import ensure_users_seeded
from routes.auth import router as auth_router
from dependencies import get_current_user, require_role


# ── Mini-app factory ──────────────────────────────────────────────────────────

def _make_test_app(db_path: str) -> FastAPI:
    """
    Create an isolated FastAPI app with real auth wiring.
    Bypasses lifespan: sets app.state.repo directly.
    """
    apply_migration(db_path)
    repo = RepositoryFactory.create({"type": "sqlite", "db_path": db_path})
    ensure_users_seeded(repo)

    mini = FastAPI()
    mini.add_middleware(SessionMiddleware, secret_key="test-secret")
    mini.state.repo = repo

    mini.include_router(auth_router, prefix="/api")

    # ── Two test-only RBAC endpoints ─────────────────────────────────────────
    @mini.get("/api/_test/ch-only")
    def _ch_only(user=Depends(require_role("commercial_head"))):
        return {"ok": True, "user_id": user["user_id"]}

    @mini.get("/api/_test/any-authenticated")
    def _any_auth(user=Depends(get_current_user)):
        return {"ok": True, "user_id": user["user_id"]}

    return mini


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path):
    app = _make_test_app(str(tmp_path / "auth_test.db"))
    # use_cookies=True keeps the session cookie across requests in the same client
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── Login tests ───────────────────────────────────────────────────────────────

def test_login_success(client):
    resp = client.post("/api/auth/login",
                       json={"username": "commercial_head_01", "password": "ch-demo-2024"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "commercial_head_01"
    assert data["role"] == "commercial_head"
    assert "password_hash" not in data


def test_login_wrong_password(client):
    resp = client.post("/api/auth/login",
                       json={"username": "commercial_head_01", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user(client):
    resp = client.post("/api/auth/login",
                       json={"username": "ghost_user", "password": "anything"})
    assert resp.status_code == 401


# ── /me tests ─────────────────────────────────────────────────────────────────

def test_me_with_valid_session(client):
    client.post("/api/auth/login",
                json={"username": "planner_01", "password": "pl-demo-2024"})
    resp = client.get("/api/auth/me")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "planner_01"
    assert "password_hash" not in resp.json()


def test_me_without_session_returns_401(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


# ── Logout tests ──────────────────────────────────────────────────────────────

def test_logout_clears_session(client):
    client.post("/api/auth/login",
                json={"username": "planner_01", "password": "pl-demo-2024"})
    assert client.get("/api/auth/me").status_code == 200

    logout_resp = client.post("/api/auth/logout")
    assert logout_resp.status_code == 200

    # Session must be gone after logout
    assert client.get("/api/auth/me").status_code == 401


# ── RBAC tests ────────────────────────────────────────────────────────────────

def test_rbac_correct_role_passes(client):
    client.post("/api/auth/login",
                json={"username": "commercial_head_01", "password": "ch-demo-2024"})
    resp = client.get("/api/_test/ch-only")
    assert resp.status_code == 200
    assert resp.json()["user_id"] == "commercial_head_01"


def test_rbac_wrong_role_denied(client):
    client.post("/api/auth/login",
                json={"username": "planner_01", "password": "pl-demo-2024"})
    resp = client.get("/api/_test/ch-only")
    assert resp.status_code == 403


def test_rbac_unauthenticated_returns_401(client):
    # No login — no session cookie
    resp = client.get("/api/_test/ch-only")
    assert resp.status_code == 401


def test_rbac_any_authenticated_passes_all_roles(client):
    """All 5 seed users should reach the any-authenticated endpoint."""
    credentials = [
        ("commercial_head_01", "ch-demo-2024"),
        ("planner_01",         "pl-demo-2024"),
        ("planner_02",         "pl2-demo-2024"),
        ("sales_mgr_01",       "sm-demo-2024"),
        ("sop_chair_01",       "sop-demo-2024"),
    ]
    for username, password in credentials:
        # Fresh client per user to avoid session bleed-over
        app = _make_test_app.__wrapped__ if hasattr(_make_test_app, "__wrapped__") else None
        # Re-use the same app fixture; log in and log out
        client.post("/api/auth/login", json={"username": username, "password": password})
        resp = client.get("/api/_test/any-authenticated")
        assert resp.status_code == 200, f"{username} failed: {resp.json()}"
        client.post("/api/auth/logout")


# ── Audit log test ─────────────────────────────────────────────────────────────

def test_login_logout_writes_audit_rows(tmp_path):
    """LOGIN and LOGOUT must each produce an audit_log row."""
    app = _make_test_app(str(tmp_path / "audit_test.db"))
    with TestClient(app, raise_server_exceptions=True) as c:
        c.post("/api/auth/login",
               json={"username": "planner_01", "password": "pl-demo-2024"})
        c.post("/api/auth/logout")

    repo = app.state.repo
    logs = repo.query("audit_log")
    actions = [r["action"] for r in logs]
    assert "LOGIN" in actions
    assert "LOGOUT" in actions
