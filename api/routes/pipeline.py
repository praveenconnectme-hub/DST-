from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/pipeline/state")
def get_pipeline_state(request: Request, cycle_id: str = Query(...)):
    repo = request.app.state.repo
    state = repo.get_pipeline_state(cycle_id)
    if not state:
        return {"cycle_id": cycle_id, "current_state": "IDLE", "state_meta_json": None}
    return state
