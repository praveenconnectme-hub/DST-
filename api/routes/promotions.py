"""
Promotions Ledger routes — BRD §12, Phase 3 P3-2.

GET   /api/promotions              — list promotions, optional cycle_id filter
POST  /api/promotions              — create promotion (commercial_head only — D-020)
PATCH /api/promotions/{promo_id}   — edit promotion (commercial_head only — D-020)
POST  /api/promotions/ai-draft     — static draft suggestions (NO LLM — D-019)

AUDIT REQUIREMENT (load-bearing for G1):
Every create and edit writes an audit_log row in the SAME transaction as the
promotions_ledger write. The G1 approval trail depends on this atomicity.

All reads/writes go through the repository (Rule 1).
No sqlite3, no file I/O, no inline SQL in this file.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from dependencies import get_current_user, require_role
from models.schemas import (
    AiDraftItem,
    AiDraftResponse,
    PromoCreate,
    PromoResponse,
    PromoUpdate,
)

router = APIRouter(tags=["promotions"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Static festival reference (D-019: no LLM, static_fallback) ───────────────
# Approximate ISO week numbers for recurring Indian TV promotional events.
_STATIC_EVENTS = [
    {
        "event_name":          "Navratri/Dussehra Sale",
        "event_type":          "festival",
        "typical_start_week":  "W40",
        "typical_end_week":    "W42",
        "offer_type":          "price_discount",
        "expected_uplift_pct": 18.0,
    },
    {
        "event_name":          "Diwali Bonanza",
        "event_type":          "festival",
        "typical_start_week":  "W43",
        "typical_end_week":    "W45",
        "offer_type":          "bundle_offer",
        "expected_uplift_pct": 25.0,
    },
    {
        "event_name":          "Christmas & New Year Sale",
        "event_type":          "seasonal",
        "typical_start_week":  "W51",
        "typical_end_week":    "W52",
        "offer_type":          "price_discount",
        "expected_uplift_pct": 12.0,
    },
    {
        "event_name":          "Republic Day Sale",
        "event_type":          "seasonal",
        "typical_start_week":  "W04",
        "typical_end_week":    "W05",
        "offer_type":          "cashback",
        "expected_uplift_pct": 10.0,
    },
    {
        "event_name":          "Holi Festival Promotion",
        "event_type":          "festival",
        "typical_start_week":  "W10",
        "typical_end_week":    "W11",
        "offer_type":          "bundle_offer",
        "expected_uplift_pct": 8.0,
    },
    {
        "event_name":          "Independence Day Sale",
        "event_type":          "seasonal",
        "typical_start_week":  "W32",
        "typical_end_week":    "W33",
        "offer_type":          "price_discount",
        "expected_uplift_pct": 10.0,
    },
    {
        "event_name":          "Onam Festival (South India)",
        "event_type":          "festival",
        "typical_start_week":  "W35",
        "typical_end_week":    "W37",
        "offer_type":          "price_discount",
        "expected_uplift_pct": 15.0,
    },
    {
        "event_name":          "IPL Season Promotion",
        "event_type":          "sporting",
        "typical_start_week":  "W15",
        "typical_end_week":    "W24",
        "offer_type":          "bundle_offer",
        "expected_uplift_pct": 8.0,
    },
]

_IMPACT_TO_UPLIFT = {"high": 20.0, "medium": 10.0, "low": 5.0}


@router.get("/promotions", response_model=list[PromoResponse])
def list_promotions(
    request: Request,
    cycle_id: Optional[str] = Query(
        None,
        description="ISO cycle week (e.g. 2024-W43). Returns promos active during "
                    "that week: start_week <= cycle_id <= end_week.",
    ),
    current_user: dict = Depends(get_current_user),
) -> list[PromoResponse]:
    repo = request.app.state.repo
    all_promos = repo.query("promotions_ledger")

    if cycle_id:
        # ISO week strings compare correctly as plain strings when zero-padded ("2024-W03")
        all_promos = [
            p for p in all_promos
            if (p.get("start_week") or "") <= cycle_id
            and cycle_id <= (p.get("end_week") or cycle_id)
        ]

    return [PromoResponse(**p) for p in all_promos]


@router.post("/promotions", response_model=PromoResponse, status_code=201)
def create_promotion(
    body: PromoCreate,
    request: Request,
    current_user: dict = Depends(require_role("commercial_head")),
) -> PromoResponse:
    """
    Create a promotion entry in the ledger.

    RBAC: commercial_head only (D-020).
    AUDIT: promotions_ledger write and PROMO_CREATED audit row are committed
    in the same transaction — atomically (load-bearing for the G1 approval trail).
    """
    repo = request.app.state.repo
    promo_id = str(uuid.uuid4())

    promo_row = {
        "promo_id":            promo_id,
        "event_name":          body.event_name,
        "sku_id":              body.sku_id,
        "start_week":          body.start_week,
        "end_week":            body.end_week,
        "offer_type":          body.offer_type,
        "financial_value":     body.financial_value,
        "is_approved":         0,
        "is_ai_generated":     1 if body.is_ai_generated else 0,
        "expected_uplift_pct": body.expected_uplift_pct,
    }

    with repo.transaction():
        repo.upsert("promotions_ledger", [promo_row])
        repo.upsert("audit_log", [{
            "timestamp":   _now_iso(),
            "actor":       current_user["user_id"],
            "action":      "PROMO_CREATED",
            "entity":      "promotions_ledger",
            "detail_json": json.dumps({
                "promo_id":   promo_id,
                "event_name": body.event_name,
                "start_week": body.start_week,
                "end_week":   body.end_week,
            }),
        }])

    return PromoResponse(**promo_row)


@router.patch("/promotions/{promo_id}", response_model=PromoResponse)
def update_promotion(
    promo_id: str,
    body: PromoUpdate,
    request: Request,
    current_user: dict = Depends(require_role("commercial_head")),
) -> PromoResponse:
    """
    Edit an existing promotion.

    RBAC: commercial_head only (D-020).
    AUDIT: promotions_ledger write and PROMO_UPDATED audit row are committed
    in the same transaction — atomically (load-bearing for the G1 approval trail).
    """
    repo = request.app.state.repo

    rows = repo.query("promotions_ledger", filters={"promo_id": promo_id})
    if not rows:
        raise HTTPException(status_code=404, detail=f"Promotion '{promo_id}' not found")

    existing = rows[0]
    changes = body.model_dump(exclude_unset=True)
    merged = {**existing, **changes}

    with repo.transaction():
        repo.upsert("promotions_ledger", [merged])
        repo.upsert("audit_log", [{
            "timestamp":   _now_iso(),
            "actor":       current_user["user_id"],
            "action":      "PROMO_UPDATED",
            "entity":      "promotions_ledger",
            "detail_json": json.dumps({"promo_id": promo_id, "changes": changes}),
        }])

    return PromoResponse(**merged)


@router.post("/promotions/ai-draft", response_model=AiDraftResponse)
def ai_draft_promotions(
    request: Request,
    cycle_id: Optional[str] = Query(
        None,
        description="Target cycle year for week anchoring (e.g. 2024-W43 → year 2024)",
    ),
    current_user: dict = Depends(get_current_user),
) -> AiDraftResponse:
    """
    Return STATIC draft promotion suggestions. NO LLM calls are made (D-019).

    Sources (in order):
    1. event_calendar rows with source='static_fallback' already in the DB.
    2. Built-in Indian TV market festival/event reference list.

    All drafts have is_system_suggested=True (system-offered, not human-typed)
    and llm_used=False (no LLM involved, per D-019). They must be accepted/edited
    by the user before they appear in the promotions_ledger.
    """
    repo = request.app.state.repo

    # Derive year for week anchoring (default to current year)
    year = cycle_id[:4] if (cycle_id and len(cycle_id) >= 4) else "2024"

    drafts: list[AiDraftItem] = []

    # ── Source 1: event_calendar (static_fallback entries) ───────────────────
    cal_rows = repo.query("event_calendar", filters={"source": "static_fallback"})
    for row in cal_rows:
        drafts.append(AiDraftItem(
            event_name=row["event_name"],
            event_type=row.get("event_type") or "other",
            suggested_start_week=f"{year}-{row['week_index']}" if not row["week_index"].startswith(year)
                                  else row["week_index"],
            suggested_end_week=f"{year}-{row['week_index']}" if not row["week_index"].startswith(year)
                                else row["week_index"],
            offer_type="price_discount",
            expected_uplift_pct=_IMPACT_TO_UPLIFT.get(row.get("expected_impact", "medium"), 10.0),
            is_system_suggested=True,
            source="event_calendar",
            note="Sourced from static event calendar in DB. Accept or edit before saving.",
        ))

    # ── Source 2: built-in static festival reference ─────────────────────────
    for ev in _STATIC_EVENTS:
        drafts.append(AiDraftItem(
            event_name=ev["event_name"],
            event_type=ev["event_type"],
            suggested_start_week=f"{year}-{ev['typical_start_week']}",
            suggested_end_week=f"{year}-{ev['typical_end_week']}",
            offer_type=ev["offer_type"],
            expected_uplift_pct=ev["expected_uplift_pct"],
            is_system_suggested=True,
            source="static_fallback",
            note="Static Indian TV market reference. Accept or edit before saving.",
        ))

    return AiDraftResponse(
        source="static_fallback",
        llm_used=False,
        drafts=drafts,
    )
