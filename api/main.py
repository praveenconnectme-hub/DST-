"""
FastAPI application entry point.

On startup:
  1. Run migration (creates all §5.1 tables if not present).
  2. Initialise the SQLiteRepository and attach to app.state.
  3. Seed demo users if users table is empty (D-018).
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from repository.factory import RepositoryFactory
from migrations import migration_001
from seed import ensure_users_seeded
from routes import health, ingestion, forecast, audit, pipeline, auth as auth_routes, promotions as promotions_routes, gates as gates_routes, sensing as sensing_routes


DB_PATH        = os.environ.get("DB_PATH",         "/data/dst.db")
DATA_DIR       = os.environ.get("DATA_DIR",        "/data")
SESSION_SECRET = os.environ.get("SESSION_SECRET",  "dev-secret-change-in-prod")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ─────────────────────────────────────────────────────────────
    print(f"[api] Running migration against {DB_PATH}")
    migration_001.run(DB_PATH)

    repo = RepositoryFactory.create({"type": "sqlite", "db_path": DB_PATH})
    app.state.repo     = repo
    app.state.data_dir = DATA_DIR

    ensure_users_seeded(repo)
    print("[api] Repository ready.")
    yield
    # ── shutdown ─────────────────────────────────────────────────────────────
    print("[api] Shutting down.")


app = FastAPI(title="Demand Sensing Control Tower API", version="1.0.0-phase3",
              lifespan=lifespan)

# SessionMiddleware must be added before CORSMiddleware (innermost first).
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(health.router,       prefix="/api")
app.include_router(ingestion.router,    prefix="/api")
app.include_router(forecast.router,     prefix="/api")
app.include_router(audit.router,        prefix="/api")
app.include_router(pipeline.router,     prefix="/api")
app.include_router(auth_routes.router,       prefix="/api")
app.include_router(promotions_routes.router, prefix="/api")
app.include_router(gates_routes.router,      prefix="/api")
app.include_router(sensing_routes.router,    prefix="/api")
