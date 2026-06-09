"""
Worker entry point — custom Python state-machine loop (BRD §9.1).

Responsibilities:
1. Run migration on startup.
2. Generate synthetic data if not already present.
3. Poll job_queue every POLL_INTERVAL seconds.
4. For each queued job, execute: ingest → baseline → mark done.

No Airflow/Prefect/Dagster — custom loop only.
api ↔ worker communicate via the job_queue table in SQLite.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add api directory to path so we can share the repository layer.
# In the container: worker runs from /app/, api is at /app/api/
# Locally (tests): worker is in worker/, api is in ../api/
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(_this_dir, "api"),        # container: /app/api
    os.path.join(_this_dir, "..", "api"),  # local: ../api
]:
    if os.path.isdir(_candidate):
        sys.path.insert(0, _candidate)
        break

from repository.factory import RepositoryFactory
from migrations import migration_001, migration_002
from pipeline.states import PipelineState
from pipeline import ingestion as ingestion_module
from pipeline import baseline as baseline_module
from pipeline import signals  as signals_module
from pipeline import sensing  as sensing_module
from pipeline import scoring  as scoring_module
from data_gen.synthetic import generate as generate_synthetic


DB_PATH        = os.environ.get("DB_PATH",   "/data/dst.db")
DATA_DIR       = os.environ.get("DATA_DIR",  "/data")
POLL_INTERVAL  = int(os.environ.get("POLL_INTERVAL", "5"))   # seconds


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_cycle() -> str:
    from datetime import date
    iso = date.today().isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def ensure_synthetic_data(repo, data_dir: str) -> None:
    """Generate synthetic CSVs if sales_history is empty."""
    existing = repo.query("sales_history")
    if existing:
        print("[worker] Sales history already loaded, skipping synthetic generation.")
        return

    sales_csv = os.path.join(data_dir, "sales_history.csv")
    if not Path(sales_csv).exists():
        print("[worker] Generating synthetic dataset…")
        generate_synthetic(data_dir)
        print("[worker] Synthetic data generated.")
    else:
        print("[worker] Synthetic CSVs exist on disk.")


def run_job(repo, job: dict, data_dir: str) -> dict:
    """
    Execute one sensing cycle.  Handles both fresh runs and G1-blocked resumes.

    State machine (Phase 3, D-022):
      Fresh run  : INGESTING → BASELINING → LOADING_SIGNALS
                   ── G1 gate check ──
                     unapproved → G1_PROMOTIONS_BLOCKED  (return, release worker)
                     approved   → SENSING → SCORING → CYCLE_COMPLETE
      Resume     : pipeline_state already G1_PROMOTIONS_BLOCKED
                     gate still not approved → no-op return (no new audit rows)
                     gate now approved       → SENSING → SCORING → CYCLE_COMPLETE
                                              (modules 1-3 are NOT re-run)

    SAFETY INVARIANT
    ────────────────
    There is NO code path from this function that reaches PipelineState.SENSING
    unless repo.get_gate_status('G1', cycle_id) == 'approved' has been explicitly
    verified in the same call.  This is enforced by structure:

      • Fresh path:  the gate check (line marked ── GATE CHECK ──) is the ONLY
        branch that leads to the sensing block; unapproved returns immediately.
      • Resume path: the same gate check is the ONLY entry point to sensing;
        unapproved returns immediately.
      • Exception path: any exception is caught and routes to PipelineState.ERROR,
        which is a terminal state — SENSING is not reachable from the except block.

    A reviewer can confirm the invariant by searching this function for
    `PipelineState.SENSING` — every occurrence is preceded by a gate check.
    """
    cycle_id = job["cycle_id"]

    try:
        # ── Resume check ──────────────────────────────────────────────────────
        # If the cycle is sitting at G1_PROMOTIONS_BLOCKED from a prior run, skip
        # modules 1-3 entirely and re-enter at the gate check below.
        ps = repo.get_pipeline_state(cycle_id)
        is_resuming = (
            ps is not None and
            ps["current_state"] == PipelineState.G1_PROMOTIONS_BLOCKED
        )

        if is_resuming:
            # ── GATE CHECK (resume path) ──────────────────────────────────
            if repo.get_gate_status("G1", cycle_id) != "approved":
                # Still blocked. Return without any writes — no duplicate audit rows.
                print(f"[worker] cycle={cycle_id} re-picked: G1 still not approved — staying blocked")
                return {"status": "blocked", "gate": "G1", "resumed": False}

            # Gate now approved: resume from sensing.
            print(f"[worker] cycle={cycle_id} RESUMING after G1 approval — skipping ingestion/baseline/signals")
            ingest_summary   = {"skipped": True, "resumed": True}
            baseline_summary = {"skipped": True, "resumed": True}
            signals_summary  = {"skipped": True, "resumed": True}

        else:
            # ── Module 1: Ingestion ───────────────────────────────────────────
            repo.set_pipeline_state(cycle_id, PipelineState.INGESTING,
                                    {"step": "ingestion"}, "system")
            print(f"[worker] cycle={cycle_id} INGESTING …")
            ingest_summary = ingestion_module.run(repo, data_dir)
            print(f"[worker] Ingestion done: {ingest_summary}")

            # ── Module 2: Baseline ────────────────────────────────────────────
            repo.set_pipeline_state(cycle_id, PipelineState.BASELINING,
                                    {"step": "baseline"}, "system")
            print(f"[worker] cycle={cycle_id} BASELINING …")
            baseline_summary = baseline_module.run(repo)
            print(f"[worker] Baseline done: {baseline_summary.get('forecasted_series')} series")

            # ── Module 3: Signal ingestion ────────────────────────────────────
            repo.set_pipeline_state(cycle_id, PipelineState.LOADING_SIGNALS,
                                    {"step": "signals"}, "system")
            print(f"[worker] cycle={cycle_id} LOADING_SIGNALS …")
            signals_summary = signals_module.run(repo, data_dir)
            print(f"[worker] Signals done: {signals_summary}")

            # ── GATE CHECK (fresh path) ───────────────────────────────────────
            # Safety invariant: SENSING is unreachable without passing this check.
            if repo.get_gate_status("G1", cycle_id) != "approved":
                repo.set_pipeline_state(cycle_id, PipelineState.G1_PROMOTIONS_BLOCKED,
                                        {"step": "gate_check", "gate": "G1"}, "system")
                print(f"[worker] cycle={cycle_id} G1_PROMOTIONS_BLOCKED — awaiting commercial_head approval")
                return {
                    "status":    "blocked",
                    "gate":      "G1",
                    "ingestion": ingest_summary,
                    "baseline":  baseline_summary,
                    "signals":   signals_summary,
                }
            # Gate approved on first pass — fall through to sensing.

        # ── Module 4: XGBoost sensing ─────────────────────────────────────────
        # Reachable ONLY if gate check above returned "approved".
        repo.set_pipeline_state(cycle_id, PipelineState.SENSING,
                                {"step": "sensing"}, "system")
        print(f"[worker] cycle={cycle_id} SENSING …")
        sensing_summary = sensing_module.run(repo)
        print(f"[worker] Sensing done: MAPE={sensing_summary.get('overall_sensing_mape_pct')}%")

        # ── Module 5: Accuracy scoring ────────────────────────────────────────
        repo.set_pipeline_state(cycle_id, PipelineState.SCORING,
                                {"step": "scoring"}, "system")
        print(f"[worker] cycle={cycle_id} SCORING …")
        scoring_summary = scoring_module.run(repo)
        print(f"[worker] Scoring done: {scoring_summary}")

        # ── CYCLE_COMPLETE ────────────────────────────────────────────────────
        repo.set_pipeline_state(cycle_id, PipelineState.CYCLE_COMPLETE,
                                {"step": "done"}, "system")
        print(f"[worker] cycle={cycle_id} CYCLE_COMPLETE")

        return {
            "status":    "done",
            "ingestion": ingest_summary,
            "baseline":  baseline_summary,
            "signals":   signals_summary,
            "sensing":   sensing_summary,
            "scoring":   scoring_summary,
        }

    except Exception as exc:
        print(f"[worker] Job failed: {exc}")
        repo.set_pipeline_state(cycle_id, PipelineState.ERROR,
                                {"error": str(exc)}, "system")
        return {"status": "error", "error": str(exc)}


def _process_job(repo, job: dict, data_dir: str) -> None:
    """
    Execute one job from the job_queue table — the inner body of the poll loop.

    Extracted as a standalone function so tests can call it directly without
    running the infinite while-True loop.  This is the only path that writes
    job_queue.status back; exercising it is what validates the CHECK constraint
    fix from migration_002.

    Exception safety (D-023):
      Any exception — including a DB/OS error on the upsert-as-running call —
      is caught here.  We attempt a best-effort write of status='error' so the
      job goes terminal.  If that write also fails (DB completely unreachable)
      the job stays 'running', which the poll loop never re-picks — also
      terminal.  In no case does an exception propagate to the poll loop's outer
      except, which would leave the job as 'queued' and cause an infinite retry.
    """
    job_id = job["job_id"]
    print(f"[worker] Picked up job_id={job_id} cycle={job['cycle_id']}")

    try:
        # Mark as running
        repo.upsert("job_queue", [{
            **job,
            "status":     "running",
            "started_at": _now_iso(),
        }])

        result = run_job(repo, job, data_dir)

        status = result.pop("status", "done")
        repo.upsert("job_queue", [{
            **job,
            "status":      status,
            "finished_at": _now_iso(),
            "result_json": json.dumps(result),
        }])
        print(f"[worker] Job {job_id} → {status}")

    except Exception as exc:
        print(f"[worker] Job {job_id} raised unexpected exception: {exc}")
        # Best-effort: mark as error so the job does not stay 'queued' / retry.
        # If this write also fails, the job stays 'running' — still terminal.
        try:
            repo.upsert("job_queue", [{
                **job,
                "status":      "error",
                "finished_at": _now_iso(),
                "result_json": json.dumps({"error": str(exc)}),
            }])
        except Exception as inner:
            print(f"[worker] Could not mark job {job_id} as error: {inner}")
        print(f"[worker] Job {job_id} → error")


def main():
    print(f"[worker] Starting. DB={DB_PATH}, DATA={DATA_DIR}")

    # ── Run migrations ─────────────────────────────────────────────────────
    migration_001.run(DB_PATH)
    migration_002.run(DB_PATH)

    repo = RepositoryFactory.create({"type": "sqlite", "db_path": DB_PATH})

    # ── Generate synthetic data on first boot ──────────────────────────────
    ensure_synthetic_data(repo, DATA_DIR)

    # ── Poll loop ──────────────────────────────────────────────────────────
    print(f"[worker] Polling job_queue every {POLL_INTERVAL}s…")
    while True:
        try:
            # Pick up new jobs AND blocked jobs (D-022: blocked jobs re-enter
            # on every tick; run_job() checks the gate and resumes or no-ops).
            queued_jobs  = repo.query("job_queue", filters={"status": "queued"},
                                      order_by=["job_id"])
            blocked_jobs = repo.query("job_queue", filters={"status": "blocked"},
                                      order_by=["job_id"])

            for job in queued_jobs + blocked_jobs:
                _process_job(repo, job, DATA_DIR)

        except Exception as exc:
            print(f"[worker] Poll loop error: {exc}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
