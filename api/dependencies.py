"""
FastAPI dependencies — BRD §12, D-018.

get_current_user() is the SINGLE point of auth resolution for all routes.
Every gate route MUST depend on this function — never read request.session
directly in a route handler.

Future SSO migration: change only get_current_user() to validate an OIDC/SAML
token instead of the session cookie — every protected route benefits automatically.
"""
from fastapi import Depends, HTTPException, Request


def get_current_user(request: Request) -> dict:
    """
    Resolve the authenticated user from the signed session cookie.

    Future SSO/SAML migration point: replace `request.session.get("user_id")`
    with OIDC token introspection here. No other file needs to change.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    repo = request.app.state.repo
    rows = repo.query("users", filters={"user_id": user_id})
    if not rows:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Session user not found")

    return rows[0]


def require_role(*allowed_roles: str):
    """
    Return a FastAPI dependency that enforces RBAC.

    Usage:
        @router.post("/gate/approve")
        def approve(user = Depends(require_role("commercial_head"))):
            ...
    """
    def _check(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Role '{current_user.get('role')}' not permitted; "
                    f"required one of {allowed_roles}"
                ),
            )
        return current_user
    return _check
