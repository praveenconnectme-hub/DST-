from fastapi import APIRouter, Query, HTTPException, Request
from models.schemas import BaselineForecastResponse, ForecastRow, BaselineForecastListItem

router = APIRouter()


@router.get("/forecast/baseline", response_model=BaselineForecastResponse)
def get_baseline_forecast(
    request: Request,
    sku_id: str = Query(...),
    state_code: str = Query(...),
):
    repo = request.app.state.repo

    forecast_rows = repo.query(
        "baseline_forecast",
        filters={"sku_id": sku_id, "state_code": state_code},
        order_by=["week_index"],
    )
    if not forecast_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No baseline forecast found for {sku_id} × {state_code}. "
                   "Run ingestion + baseline worker first.",
        )

    # Pull actuals for the same series to compute deviation
    actuals_map = {}
    actuals_rows = repo.query(
        "sales_history",
        filters={"sku_id": sku_id, "state_code": state_code},
    )
    for row in actuals_rows:
        actuals_map[row["week_index"]] = row["quantity_actual"]

    model_id = forecast_rows[0].get("model_id") if forecast_rows else None

    forecasts = [
        ForecastRow(
            week_index=r["week_index"],
            forecast_qty=r["forecast_qty"],
            actual=actuals_map.get(r["week_index"]),
        )
        for r in forecast_rows
    ]

    return BaselineForecastResponse(
        sku_id=sku_id,
        state_code=state_code,
        model_id=model_id,
        forecasts=forecasts,
    )


@router.get("/forecast/baseline/list", response_model=list[BaselineForecastListItem])
def list_baseline_forecasts(request: Request):
    """Return distinct (sku_id, state_code) series that have baseline forecasts."""
    repo = request.app.state.repo
    rows = repo.query("baseline_forecast", order_by=["sku_id", "state_code", "week_index"])

    # Group by sku × state
    seen: dict[tuple, BaselineForecastListItem] = {}
    for r in rows:
        key = (r["sku_id"], r["state_code"])
        if key not in seen:
            seen[key] = BaselineForecastListItem(
                sku_id=r["sku_id"],
                state_code=r["state_code"],
                model_id=r.get("model_id"),
                weeks=1,
            )
        else:
            seen[key].weeks += 1

    return list(seen.values())
