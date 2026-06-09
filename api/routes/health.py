from fastapi import APIRouter, Request
from models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    repo = request.app.state.repo
    counts = {}
    try:
        counts["skus"]              = len(repo.query("sku_master"))
        counts["states"]            = len(repo.query("geo_master"))
        counts["sales_history"]     = len(repo.query("sales_history"))
        counts["baseline_forecasts"]= len(repo.query("baseline_forecast"))
        counts["quarantine"]        = 0  # populated when quarantine table is added Phase2+
    except Exception:
        pass

    pipeline_state = None
    current_cycle  = None
    try:
        rows = repo.query("pipeline_state", order_by=["updated_at"])
        if rows:
            latest = rows[-1]
            pipeline_state = latest.get("current_state")
            current_cycle  = latest.get("cycle_id")
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        pipeline_state=pipeline_state,
        current_cycle=current_cycle,
        counts=counts,
    )
