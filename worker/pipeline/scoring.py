"""
Module 5 — Sensing accuracy scoring (BRD §4.7, Phase 2).

D-017: Scores demand_sensing_output (XGBoost holdout, last HOLDOUT_WEEKS of
sales_history) against the actuals table. baseline_forecast is NOT scored
here — those rows cover HORIZON weeks beyond the sales_history window and
no actuals exist for them. Baseline holdout accuracy was already persisted
to accuracy_metrics by baseline.py's leave-last-12-out backtest.

Writes to:
  - accuracy_metrics : one row per sku×state×holdout_week (xgboost model_id)
  - audit_log        : SENSING_SCORED row with aggregate summary stats (same transaction)

All reads/writes go through the repository (Rule 1).
No sqlite3, no file I/O, no inline SQL in this file.
"""
import json
from datetime import datetime, timezone

import numpy as np

MAPE_FLAG_THRESHOLD = 0.30   # flag all rows in a series if mean series MAPE ≥ 30 %


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(repo) -> dict:
    """
    Join demand_sensing_output with actuals; compute per-row MAPE/bias;
    persist results to accuracy_metrics.

    Returns
    -------
    dict: matched_rows (int), overall_mape_pct (float|None), n_flagged (int)
    """
    sensing = repo.read_frame("demand_sensing_output")
    actuals = repo.read_frame("actuals")

    if sensing.empty or actuals.empty:
        summary = {"matched_rows": 0, "overall_mape_pct": None, "n_flagged": 0}
        _write_audit(repo, summary)
        return summary

    merged = sensing.merge(
        actuals[["sku_id", "state_code", "week_index", "quantity_actual"]],
        on=["sku_id", "state_code", "week_index"],
        how="inner",
    )

    if merged.empty:
        summary = {"matched_rows": 0, "overall_mape_pct": None, "n_flagged": 0}
        _write_audit(repo, summary)
        return summary

    # Identify series whose mean MAPE exceeds the retrain threshold
    flagged_series = set()
    for (sku, state), grp in merged.groupby(["sku_id", "state_code"]):
        denom = grp["quantity_actual"].clip(lower=1.0)
        series_mape = float(
            np.mean(np.abs(grp["forecast_qty"] - grp["quantity_actual"]) / denom)
        )
        if series_mape >= MAPE_FLAG_THRESHOLD:
            flagged_series.add((sku, state))

    acc_rows = []
    row_apes = []
    for _, r in merged.iterrows():
        actual = max(1.0, float(r["quantity_actual"]))
        pred   = float(r["forecast_qty"])
        ape    = abs(pred - actual) / actual
        bias_v = (pred - actual) / actual
        row_apes.append(ape)
        acc_rows.append({
            "sku_id":              r["sku_id"],
            "state_code":          r["state_code"],
            "week_index":          r["week_index"],
            "model_id":            r["model_id"],
            "mape":                round(ape, 6),
            "bias":                round(bias_v, 6),
            "flagged_for_retrain": 1 if (r["sku_id"], r["state_code"]) in flagged_series else 0,
        })

    repo.upsert("accuracy_metrics", acc_rows)

    overall_mape = round(float(np.mean(row_apes)) * 100, 1) if row_apes else None
    n_flagged    = len(flagged_series)

    summary = {
        "matched_rows":     len(acc_rows),
        "overall_mape_pct": overall_mape,
        "n_flagged":        n_flagged,
    }
    _write_audit(repo, summary)

    print(f"[scoring] Sensing MAPE={overall_mape}% on {len(acc_rows)} matched rows, "
          f"{n_flagged} series flagged for retrain")
    return summary


def _write_audit(repo, summary: dict) -> None:
    with repo.transaction():
        repo.upsert("audit_log", [{
            "timestamp":   _now_iso(),
            "actor":       "scoring_module",
            "action":      "SENSING_SCORED",
            "entity":      "accuracy_metrics",
            "detail_json": json.dumps(summary),
        }])
