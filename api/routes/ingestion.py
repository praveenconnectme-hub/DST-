import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from models.schemas import IngestResponse

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/ingest", response_model=IngestResponse)
def trigger_ingest(request: Request):
    """Queue an ingestion job for the worker to pick up."""
    repo = request.app.state.repo

    # Determine or create a cycle_id for today
    from datetime import date
    today = date.today().isocalendar()
    cycle_id = f"{today.year}-W{today.week:02d}"

    # Check if a job is already running
    running = repo.query("job_queue", filters={"cycle_id": cycle_id, "status": "running"})
    if running:
        return IngestResponse(
            message=f"Ingestion already running for cycle {cycle_id}",
            job_id=running[0]["job_id"],
        )

    # Enqueue the job
    rows = [{
        "cycle_id":   cycle_id,
        "action":     "ingest_and_baseline",
        "status":     "queued",
        "created_at": _now_iso(),
    }]
    repo.upsert("job_queue", rows)

    # Retrieve the job_id
    queued = repo.query("job_queue", filters={"cycle_id": cycle_id, "status": "queued"},
                        order_by=["job_id"])
    job_id = queued[-1]["job_id"] if queued else None

    return IngestResponse(
        message=f"Ingestion queued for cycle {cycle_id}. Worker will pick it up shortly.",
        job_id=job_id,
    )


@router.get("/ingest/status")
def ingest_status(request: Request):
    """Return recent job_queue entries."""
    repo = request.app.state.repo
    jobs = repo.query("job_queue", order_by=["job_id"])
    return {"jobs": jobs[-20:]}  # last 20
