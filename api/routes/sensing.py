"""
Sensing API routes — BRD §4.3, Phase 3 P3-6.

GET /api/sensing         — per-series XGBoost sensing output + actuals
GET /api/sensing/summary — overall MAPE across all sensing series

All reads go through the repository (Rule 1).
No sqlite3, no file I/O, no inline SQL in this file.
Auth required: all endpoints require get_current_user() (401 if unauthenticated).
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from dependencies import get_current_user
from models.schemas import SensingResponse, SensingSummaryResponse, SensingWeekRow

router = APIRouter(tags=["sensing"])


@router.get("/sensing/summary", response_model=SensingSummaryResponse)
def get_sensing_summary(
    request: Request,
    current_user: dict = Depends(get_current_user),
) -> SensingSummaryResponse:
    """
    Return the overall XGBoost sensing MAPE (mean across all series).
    Reads accuracy_metrics rows whose model_id starts with 'xgboost_'.
    Returns overall_sensing_mape_pct=None if no sensing metrics exist yet.
    """
    repo = request.app.state.repo
    all_metrics = repo.query("accuracy_metrics")

    sensing_metrics = [
        r for r in all_metrics
        if str(r.get("model_id", "")).startswith("xgboost_")
        and r.get("mape") is not None
    ]

    if not sensing_metrics:
        return SensingSummaryResponse(overall_sensing_mape_pct=None, series_count=0)

    overall_mape = sum(r["mape"] for r in sensing_metrics) / len(sensing_metrics)

    # Distinct sku×state series
    series_count = len({(r.get("sku_id"), r.get("state_code")) for r in sensing_metrics})

    return SensingSummaryResponse(
        overall_sensing_mape_pct=round(overall_mape * 100, 2),
        series_count=series_count,
    )


@router.get("/sensing", response_model=SensingResponse)
def get_sensing_forecast(
    request: Request,
    sku_id: str = Query(..., description="SKU identifier"),
    state_code: str = Query(..., description="Two-letter state code"),
    current_user: dict = Depends(get_current_user),
) -> SensingResponse:
    """
    Return XGBoost sensing predictions + actuals for a single sku×state series.

    Data sources (both via repository):
      - demand_sensing_output : sensing predictions for the holdout period
      - sales_history         : actuals for the same weeks (where available)

    Returns 404 if no sensing output exists for the series (worker hasn't run yet).
    """
    repo = request.app.state.repo

    sensing_rows = repo.query(
        "demand_sensing_output",
        filters={"sku_id": sku_id, "state_code": state_code},
        order_by=["week_index"],
    )
    if not sensing_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No sensing output found for {sku_id} × {state_code}. "
                   "Run the full pipeline (ingestion → baseline → signals → sensing) first.",
        )

    # Build actuals map from sales_history
    actuals_map: dict[str, float] = {}
    for row in repo.query("sales_history", filters={"sku_id": sku_id, "state_code": state_code}):
        actuals_map[row["week_index"]] = float(row["quantity_actual"])

    model_id = sensing_rows[0].get("model_id")

    weeks = [
        SensingWeekRow(
            week_index=r["week_index"],
            sensing_qty=float(r["forecast_qty"]),
            actual=actuals_map.get(r["week_index"]),
        )
        for r in sensing_rows
    ]

    return SensingResponse(
        sku_id=sku_id,
        state_code=state_code,
        model_id=model_id,
        weeks=weeks,
    )
