from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/audit")
def get_audit_log(request: Request, limit: int = 200):
    repo = request.app.state.repo
    rows = repo.query("audit_log", order_by=["audit_id"])
    return rows[-limit:]
