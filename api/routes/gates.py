"""
Gate API routes — BRD §3, Phase 3 P3-3.

GET  /api/gates/G1/{cycle_id}         — current G1 status (any authenticated role)
POST /api/gates/G1/{cycle_id}/approve — approve G1 (commercial_head ONLY)

SECURITY:
  - Commercial head approves; any other authenticated role → 403.
  - Unauthenticated → 401.
  - Each approval is scoped to exactly one cycle_id; approving cycle A never
    affects cycle B.

ATOMICITY:
  set_gate_status() is called inside with repo.transaction(), which nests the
  repo method's own inner transaction (txn_depth 1→2→1 without inner commit).
  A GATE_APPROVED audit row is then added to the same outer transaction before
  the single commit at depth 1. All three writes (gate_status INSERT OR REPLACE,
  set_gate_status audit row, GATE_APPROVED audit row) land in one commit.

IDEMPOTENCY (D-021):
  If the gate is already 'approved', the endpoint returns 200 with the current
  state and makes no further writes — no duplicate audit rows.

All reads/writes go through the repository (Rule 1).
No sqlite3, no file I/O, no inline SQL in this file.
"""
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request

from dependencies import get_current_user, require_role
from models.schemas import GateStatusResponse

router = APIRouter(tags=["gates"])

GATE_ID = "G1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_gate_row(repo, cycle_id: str) -> GateStatusResponse:
    """Return current gate row, or a synthetic 'pending' if no row exists yet."""
    rows = repo.query("gate_status", filters={"gate_id": GATE_ID, "cycle_id": cycle_id})
    if not rows:
        return GateStatusResponse(
            gate_id=GATE_ID,
            cycle_id=cycle_id,
            status="pending",
            approved_by=None,
            approved_at=None,
        )
    return GateStatusResponse(**rows[0])


@router.get("/gates/G1/{cycle_id}", response_model=GateStatusResponse)
def get_g1_status(
    cycle_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> GateStatusResponse:
    """
    Return the current G1 gate status for the given cycle.
    Returns status='pending' if the cycle has never been processed yet.
    Open to any authenticated role (read-only).
    """
    return _fetch_gate_row(request.app.state.repo, cycle_id)


@router.post("/gates/G1/{cycle_id}/approve", response_model=GateStatusResponse)
def approve_g1(
    cycle_id: str,
    request: Request,
    current_user: dict = Depends(require_role("commercial_head")),
) -> GateStatusResponse:
    """
    Approve the G1 Promotions Calendar gate for the given cycle.

    RBAC: commercial_head ONLY (D-020 established pattern; G1 is a commercial_head
    responsibility per BRD §2).
    IDEMPOTENCY (D-021): if already approved, returns 200 with current state and
    makes no further writes — preventing duplicate audit rows.
    ATOMICITY: gate_status + GATE_APPROVED audit row committed in one transaction.
    """
    repo = request.app.state.repo

    # ── Idempotency guard (D-021) ─────────────────────────────────────────────
    current_status = repo.get_gate_status(GATE_ID, cycle_id)
    if current_status == "approved":
        return _fetch_gate_row(repo, cycle_id)

    # ── Approve: gate_status + GATE_APPROVED audit in one transaction ─────────
    with repo.transaction():
        # set_gate_status opens its own nested transaction (txn_depth 1→2→1);
        # it writes gate_status and a set_gate_status audit row — no inner commit.
        repo.set_gate_status(GATE_ID, cycle_id, "approved", current_user["user_id"])

        # GATE_APPROVED audit row — explicit named action for G1 audit reviewers.
        # Written in the same outer transaction (still at txn_depth 1).
        repo.upsert("audit_log", [{
            "timestamp":   _now_iso(),
            "actor":       current_user["user_id"],
            "action":      "GATE_APPROVED",
            "entity":      "gate_status",
            "detail_json": json.dumps({
                "gate_id":  GATE_ID,
                "cycle_id": cycle_id,
            }),
        }])
    # Outer with block exits at txn_depth 1 → single commit for everything above.

    return _fetch_gate_row(repo, cycle_id)
