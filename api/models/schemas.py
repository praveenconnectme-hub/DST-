"""Pydantic schemas used by FastAPI routes."""
from pydantic import BaseModel
from typing import Any


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: str
    display_name: str
    role: str
    assigned_states_json: str = "[]"
    # NOTE: password_hash is intentionally excluded


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0-phase1"
    pipeline_state: str | None = None
    current_cycle: str | None = None
    counts: dict[str, int] = {}


class IngestResponse(BaseModel):
    message: str
    job_id: int | None = None


class ForecastRow(BaseModel):
    week_index: str
    forecast_qty: float
    actual: float | None = None


class BaselineForecastResponse(BaseModel):
    sku_id: str
    state_code: str
    model_id: str | None
    forecasts: list[ForecastRow]


class BaselineForecastListItem(BaseModel):
    sku_id: str
    state_code: str
    model_id: str | None
    weeks: int


# ── Promotions Ledger (P3-2) ─────────────────────────────────────────────────

class PromoCreate(BaseModel):
    event_name: str
    sku_id: str | None = None
    start_week: str
    end_week: str
    offer_type: str | None = None
    financial_value: float | None = None
    expected_uplift_pct: float | None = None
    is_ai_generated: bool = False


class PromoUpdate(BaseModel):
    """All fields optional — PATCH applies only what is explicitly sent."""
    event_name: str | None = None
    sku_id: str | None = None
    start_week: str | None = None
    end_week: str | None = None
    offer_type: str | None = None
    financial_value: float | None = None
    expected_uplift_pct: float | None = None
    # NOTE: is_approved intentionally excluded — approval goes through the G1 gate (P3-3)


class PromoResponse(BaseModel):
    promo_id: str
    event_name: str | None = None
    sku_id: str | None = None
    start_week: str | None = None
    end_week: str | None = None
    offer_type: str | None = None
    financial_value: float | None = None
    is_approved: int = 0
    is_ai_generated: int = 0
    expected_uplift_pct: float | None = None


class AiDraftItem(BaseModel):
    event_name: str
    event_type: str
    suggested_start_week: str
    suggested_end_week: str
    offer_type: str
    expected_uplift_pct: float
    # is_system_suggested=True means "offered by the system, not typed by a user."
    # It is INDEPENDENT of LLM use. Per D-019 no LLM is ever called; llm_used=False
    # in AiDraftResponse is the authoritative flag for that. A reviewer seeing
    # is_system_suggested=True alongside llm_used=False should read: "system-drafted
    # (static), not LLM-drafted."
    is_system_suggested: bool
    source: str
    note: str


class AiDraftResponse(BaseModel):
    source: str
    llm_used: bool
    drafts: list[AiDraftItem]


# ── Gates (P3-3) ──────────────────────────────────────────────────────────────

class GateStatusResponse(BaseModel):
    gate_id: str
    cycle_id: str
    status: str                  # pending | blocked | approved
    approved_by: str | None = None
    approved_at: str | None = None


# ── Sensing (P3-6) ────────────────────────────────────────────────────────────

class SensingWeekRow(BaseModel):
    week_index: str
    sensing_qty: float
    actual: float | None = None


class SensingResponse(BaseModel):
    sku_id: str
    state_code: str
    model_id: str | None
    weeks: list[SensingWeekRow]


class SensingSummaryResponse(BaseModel):
    overall_sensing_mape_pct: float | None
    series_count: int
