"""
Auth routes — BRD §12, D-018.

POST /auth/login   — bcrypt verify, set session, write LOGIN audit row
GET  /auth/me      — return current user (no password_hash)
POST /auth/logout  — clear session, write LOGOUT audit row

All DB reads/writes go through the repository (Rule 1).
No sqlite3, no file I/O, no inline SQL in this file.
"""
import json
from datetime import datetime, timezone

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request

from dependencies import get_current_user
from models.schemas import LoginRequest, UserResponse

router = APIRouter(tags=["auth"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/auth/login")
def login(body: LoginRequest, request: Request) -> UserResponse:
    repo = request.app.state.repo

    rows = repo.query("users", filters={"user_id": body.username})
    if not rows:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = rows[0]
    if not bcrypt.checkpw(body.password.encode(), user["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    request.session["user_id"] = user["user_id"]

    with repo.transaction():
        repo.upsert("audit_log", [{
            "timestamp":   _now_iso(),
            "actor":       user["user_id"],
            "action":      "LOGIN",
            "entity":      "users",
            "detail_json": json.dumps({"role": user["role"]}),
        }])

    return UserResponse(
        user_id=user["user_id"],
        display_name=user["display_name"],
        role=user["role"],
        assigned_states_json=user.get("assigned_states_json", "[]"),
    )


@router.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        user_id=current_user["user_id"],
        display_name=current_user["display_name"],
        role=current_user["role"],
        assigned_states_json=current_user.get("assigned_states_json", "[]"),
    )


@router.post("/auth/logout")
def logout(request: Request, current_user: dict = Depends(get_current_user)) -> dict:
    repo = request.app.state.repo

    with repo.transaction():
        repo.upsert("audit_log", [{
            "timestamp":   _now_iso(),
            "actor":       current_user["user_id"],
            "action":      "LOGOUT",
            "entity":      "users",
            "detail_json": json.dumps({}),
        }])

    request.session.clear()
    return {"message": "Logged out"}
