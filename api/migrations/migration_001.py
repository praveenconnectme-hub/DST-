"""
Migration 001 — Create ALL §5.1 tables.

This migration is idempotent (CREATE TABLE IF NOT EXISTS).
It is called once on API + worker boot.

IMPORTANT: Only this file (and SQLiteRepository) may use sqlite3.
All other code must go through the repository interface.
"""
import sqlite3
from pathlib import Path


DDL = [
    # ── Master / reference ────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS sku_master (
        sku_id          TEXT PRIMARY KEY,
        sku_name        TEXT NOT NULL,
        product_tier    TEXT CHECK(product_tier IN ('entry','mid','upper','premium','luxury')),
        base_cost_inr   REAL,
        is_active       INTEGER DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS geo_master (
        state_code      TEXT PRIMARY KEY CHECK(length(state_code) = 2),
        state_name      TEXT,
        commercial_zone TEXT CHECK(commercial_zone IN ('North','South','East','West')),
        is_reporting    INTEGER DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id             TEXT PRIMARY KEY,
        display_name        TEXT,
        role                TEXT CHECK(role IN ('planner','commercial_head','sales_manager','sop_chair')),
        assigned_states_json TEXT,
        password_hash       TEXT
    )
    """,

    # ── Transactional / history ───────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS sales_history (
        sku_id          TEXT NOT NULL REFERENCES sku_master(sku_id),
        state_code      TEXT NOT NULL REFERENCES geo_master(state_code),
        week_index      TEXT NOT NULL,
        quantity_actual INTEGER NOT NULL,
        PRIMARY KEY (sku_id, state_code, week_index)
    )
    """,

    # ── Forecasting ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS model_registry (
        model_id          TEXT PRIMARY KEY,
        model_type        TEXT CHECK(model_type IN ('holt_winters','auto_arima','croston','xgboost')),
        scope             TEXT,
        status            TEXT CHECK(status IN ('candidate','champion','production','retired')),
        trained_at        TEXT,
        train_window      TEXT,
        hyperparams_json  TEXT,
        val_mape          REAL,
        val_bias          REAL,
        feature_set_json  TEXT,
        artifact_path     TEXT,
        parent_model_id   TEXT REFERENCES model_registry(model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS baseline_forecast (
        sku_id      TEXT NOT NULL REFERENCES sku_master(sku_id),
        state_code  TEXT NOT NULL REFERENCES geo_master(state_code),
        week_index  TEXT NOT NULL,
        forecast_qty REAL NOT NULL,
        model_id    TEXT REFERENCES model_registry(model_id),
        PRIMARY KEY (sku_id, state_code, week_index)
    )
    """,

    # ── Signal intelligence ───────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS signal_data (
        signal_name      TEXT NOT NULL,
        state_code       TEXT,
        week_index       TEXT NOT NULL,
        value            REAL,
        source_connector TEXT,
        PRIMARY KEY (signal_name, state_code, week_index)
    )
    """,

    # ── Event calendar ────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS event_calendar (
        event_id        TEXT PRIMARY KEY,
        event_name      TEXT,
        event_type      TEXT CHECK(event_type IN ('festival','sporting','seasonal','other')),
        state_code      TEXT,
        week_index      TEXT,
        expected_impact TEXT CHECK(expected_impact IN ('high','medium','low')),
        source          TEXT CHECK(source IN ('llm','static_fallback'))
    )
    """,

    # ── Demand sensing output ─────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS demand_sensing_output (
        sku_id      TEXT NOT NULL REFERENCES sku_master(sku_id),
        state_code  TEXT NOT NULL REFERENCES geo_master(state_code),
        week_index  TEXT NOT NULL,
        forecast_qty REAL,
        model_id    TEXT REFERENCES model_registry(model_id),
        shap_json   TEXT,
        PRIMARY KEY (sku_id, state_code, week_index)
    )
    """,

    # ── Promotions ────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS promotions_ledger (
        promo_id           TEXT PRIMARY KEY,
        event_name         TEXT,
        sku_id             TEXT REFERENCES sku_master(sku_id),
        start_week         TEXT,
        end_week           TEXT,
        offer_type         TEXT,
        financial_value    REAL,
        is_approved        INTEGER DEFAULT 0,
        is_ai_generated    INTEGER DEFAULT 0,
        expected_uplift_pct REAL
    )
    """,

    # ── Field estimates ───────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS field_estimates (
        sku_id       TEXT NOT NULL REFERENCES sku_master(sku_id),
        state_code   TEXT NOT NULL REFERENCES geo_master(state_code),
        week_index   TEXT NOT NULL,
        estimate_qty REAL,
        submitted_by TEXT,
        submitted_at TEXT,
        PRIMARY KEY (sku_id, state_code, week_index)
    )
    """,

    # ── S&OP consensus ────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS sop_consensus_grid (
        sku_id                TEXT NOT NULL REFERENCES sku_master(sku_id),
        state_code            TEXT NOT NULL REFERENCES geo_master(state_code),
        week_index            TEXT NOT NULL,
        volume_baseline       REAL,
        volume_sensed         REAL,
        volume_field          REAL,
        volume_final_approved REAL,
        chosen_source         TEXT CHECK(chosen_source IN ('baseline','sensed','field','override')),
        deviation_pct         REAL,
        override_reason_code  TEXT,
        planner_justification TEXT,
        PRIMARY KEY (sku_id, state_code, week_index)
    )
    """,

    # ── Actuals ───────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS actuals (
        sku_id          TEXT NOT NULL REFERENCES sku_master(sku_id),
        state_code      TEXT NOT NULL REFERENCES geo_master(state_code),
        week_index      TEXT NOT NULL,
        quantity_actual INTEGER,
        loaded_at       TEXT,
        PRIMARY KEY (sku_id, state_code, week_index)
    )
    """,

    # ── MLOps ─────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS accuracy_metrics (
        sku_id              TEXT NOT NULL,
        state_code          TEXT NOT NULL,
        week_index          TEXT NOT NULL,
        model_id            TEXT NOT NULL REFERENCES model_registry(model_id),
        mape                REAL,
        bias                REAL,
        flagged_for_retrain INTEGER DEFAULT 0,
        PRIMARY KEY (sku_id, state_code, week_index, model_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS drift_metrics (
        feature_name       TEXT NOT NULL,
        week_index         TEXT NOT NULL,
        psi                REAL,
        reference_model_id TEXT REFERENCES model_registry(model_id),
        breached           INTEGER DEFAULT 0,
        PRIMARY KEY (feature_name, week_index, reference_model_id)
    )
    """,

    # ── Orchestration ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS pipeline_state (
        cycle_id        TEXT PRIMARY KEY,
        current_state   TEXT NOT NULL,
        state_meta_json TEXT,
        updated_at      TEXT,
        updated_by      TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gate_status (
        gate_id     TEXT NOT NULL,
        cycle_id    TEXT NOT NULL,
        status      TEXT CHECK(status IN ('pending','blocked','approved')) DEFAULT 'pending',
        approved_by TEXT,
        approved_at TEXT,
        PRIMARY KEY (gate_id, cycle_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_queue (
        job_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        cycle_id    TEXT NOT NULL,
        action      TEXT NOT NULL,
        status      TEXT CHECK(status IN ('queued','running','done','error')) DEFAULT 'queued',
        created_at  TEXT,
        started_at  TEXT,
        finished_at TEXT,
        result_json TEXT
    )
    """,

    # ── Audit ─────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        actor       TEXT NOT NULL,
        action      TEXT NOT NULL,
        entity      TEXT,
        detail_json TEXT
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales_history(sku_id)",
    "CREATE INDEX IF NOT EXISTS idx_sales_state ON sales_history(state_code)",
    "CREATE INDEX IF NOT EXISTS idx_baseline_sku_state ON baseline_forecast(sku_id, state_code)",
    "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_job_status ON job_queue(status)",
]


def run(db_path: str) -> None:
    """Execute all DDL statements against the SQLite database at db_path."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        for stmt in DDL:
            conn.execute(stmt)
        for idx in INDEXES:
            conn.execute(idx)
        conn.commit()
        print(f"[migration-001] Schema applied to {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/data/dst.db"
    run(path)
