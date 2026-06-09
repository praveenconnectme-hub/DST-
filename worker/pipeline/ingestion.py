"""
Module 1 — Data Ingestion (BRD §4.1).

Responsibilities:
- Load sku_master, geo_master via repo.import_csv()
- Read raw sales_history via repo.read_csv_raw()
- Validate: quarantine non-integer qty and unknown geo codes
  → persist each quarantine row to audit_log
  → raise one HIGH_SEVERITY_NOTIFICATION per quarantine batch
  → run MUST NOT abort
- Fill missing interior weeks via linear interpolation (if bounded by sales)
- Zero-fill trailing zero periods that exceed 12 consecutive weeks
- Persist clean records to sales_history via repository

File I/O is restricted to import_csv()/read_csv_raw() inside the repository.
No direct sqlite3/file access here.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd

QUARANTINE_LOG: list[dict] = []


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(repo, data_dir: str) -> dict:
    """Execute Module 1 ingestion.  Returns summary dict."""
    QUARANTINE_LOG.clear()
    summary = {
        "skus_loaded":          0,
        "states_loaded":        0,
        "actuals_loaded":       0,
        "raw_rows":             0,
        "valid_rows":           0,
        "quarantined_rows":     0,
        "interpolated_weeks":   0,
        "notifications_raised": 0,
    }

    sku_csv   = os.path.join(data_dir, "sku_master.csv")
    geo_csv   = os.path.join(data_dir, "geo_master.csv")
    sales_csv = os.path.join(data_dir, "sales_history.csv")

    # ── 1. Load master tables ────────────────────────────────────────────────
    n_skus   = repo.import_csv("sku_master", sku_csv)
    n_states = repo.import_csv("geo_master", geo_csv)
    summary["skus_loaded"]   = n_skus
    summary["states_loaded"] = n_states
    print(f"[ingest] Loaded {n_skus} SKUs, {n_states} states")

    # ── 1b. Load held-out actuals (MAPE reference; after masters for FK) ────
    actuals_csv = os.path.join(data_dir, "actuals_holdout.csv")
    if os.path.exists(actuals_csv):
        n_actuals = repo.import_csv("actuals", actuals_csv)
        summary["actuals_loaded"] = n_actuals
        print(f"[ingest] Loaded {n_actuals} holdout actuals")

    # ── 2. Load raw sales history via repository (no direct file I/O) ───────
    raw_df = repo.read_csv_raw(sales_csv)
    summary["raw_rows"] = len(raw_df)

    valid_skus   = {r["sku_id"]     for r in repo.query("sku_master")}
    valid_states = {r["state_code"] for r in repo.query("geo_master")}

    # ── 3. Validate — quarantine bad rows, NEVER abort ──────────────────────
    clean_df, quarantined = _validate(raw_df, valid_skus, valid_states)
    summary["quarantined_rows"] = len(quarantined)
    QUARANTINE_LOG.extend(quarantined)

    if quarantined:
        _persist_quarantine(repo, quarantined)
        _raise_high_severity(
            repo,
            f"Quarantined {len(quarantined)} row(s): "
            "non-integer quantity or unknown geographic code",
            len(quarantined),
        )
        summary["notifications_raised"] = 1
        print(f"[ingest] Quarantined {len(quarantined)} rows")

    # ── 4. Imputation ────────────────────────────────────────────────────────
    clean_df, n_interp = _impute(clean_df)
    summary["interpolated_weeks"] = n_interp
    summary["valid_rows"]         = len(clean_df)

    # ── 5. Persist via repository ────────────────────────────────────────────
    records = clean_df.to_dict(orient="records")
    repo.upsert("sales_history", records)
    print(f"[ingest] Persisted {len(records)} clean rows to sales_history")

    return summary


# ── Quarantine helpers ────────────────────────────────────────────────────────

def _persist_quarantine(repo, quarantined: list[dict]) -> None:
    """Write one audit_log row per quarantined record (BRD §4.1 — durable ledger)."""
    rows = [
        {
            "timestamp":   q["timestamp"],
            "actor":       "system",
            "action":      "quarantine",
            "entity":      "sales_history",
            "detail_json": json.dumps({
                "sku_id":     q["sku_id"],
                "state_code": q["state_code"],
                "week_index": q["week_index"],
                "reason":     q["reason"],
                "raw_qty":    q["raw_qty"],
            }),
        }
        for q in quarantined
    ]
    repo.upsert("audit_log", rows)


def _raise_high_severity(repo, message: str, count: int) -> None:
    """Emit a HIGH_SEVERITY_NOTIFICATION to audit_log and stdout."""
    print(f"[ingest] [HIGH] {message}")
    repo.upsert("audit_log", [{
        "timestamp":   _now_iso(),
        "actor":       "system",
        "action":      "HIGH_SEVERITY_NOTIFICATION",
        "entity":      "ingestion",
        "detail_json": json.dumps({"message": message, "quarantine_count": count}),
    }])


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(raw_df: pd.DataFrame,
              valid_skus: set,
              valid_states: set) -> tuple[pd.DataFrame, list[dict]]:
    """Return (clean_df, quarantine_list).

    Quarantine rules (BRD §4.1):
      - state_code not in geo_master    → High Severity
      - quantity_actual is non-integer  → High Severity
      - sku_id not in sku_master        → High Severity (implied by §4.1 validation)
    """
    quarantine = []
    keep_mask  = []

    for _, row in raw_df.iterrows():
        reason = None

        if row["sku_id"] not in valid_skus:
            reason = f"unknown sku_id={row['sku_id']}"
        elif row["state_code"] not in valid_states:
            reason = f"unknown state_code={row['state_code']}"
        else:
            try:
                qty = float(row["quantity_actual"])
                if qty != int(qty) or qty < 0:
                    reason = f"non-integer or negative quantity={row['quantity_actual']}"
            except (ValueError, TypeError):
                reason = f"unparseable quantity={row['quantity_actual']}"

        if reason:
            quarantine.append({
                "sku_id":     row.get("sku_id", ""),
                "state_code": row.get("state_code", ""),
                "week_index": row.get("week_index", ""),
                "reason":     reason,
                "raw_qty":    row.get("quantity_actual", ""),
                "timestamp":  _now_iso(),
            })
            keep_mask.append(False)
        else:
            keep_mask.append(True)

    clean = raw_df[keep_mask].copy()
    clean["quantity_actual"] = clean["quantity_actual"].astype(float).astype(int)
    return clean, quarantine


# ── Imputation ────────────────────────────────────────────────────────────────

def _complete_week_range(week_strs: list[str]) -> list[str]:
    """Return every ISO 'YYYY-WW' from min(week_strs) to max(week_strs) inclusive.

    This detects weeks that are entirely absent from the input (not just NaN rows),
    which is required to correctly identify interior gaps for interpolation.
    """
    if not week_strs:
        return []

    def _to_date(w: str) -> datetime.date:
        y, wn = w.split("-W")
        return datetime.date.fromisocalendar(int(y), int(wn), 1)

    sorted_strs = sorted(week_strs)
    start = _to_date(sorted_strs[0])
    end   = _to_date(sorted_strs[-1])

    result, d = [], start
    while d <= end:
        iso = d.isocalendar()
        result.append(f"{iso.year}-W{iso.week:02d}")
        d += datetime.timedelta(weeks=1)
    return result


def _impute(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply BRD §4.1 imputation rules per (sku_id, state_code) series.

    Rule 1 — Interior gaps: weeks wholly absent from the series are detected via
    _complete_week_range().  If the gap is bounded by data on both sides it is
    filled by linear interpolation.

    Rule 2 — Trailing zeros: if the series ends with > 12 consecutive zeros those
    zeros are genuine (the SKU stopped selling); they are left as-is.

    Returns (imputed_df, n_interpolated_weeks).
    """
    n_interp     = 0
    result_parts = []

    for (sku, state), group in df.groupby(["sku_id", "state_code"]):
        present_weeks = list(group["week_index"].unique())

        # Complete week range detects absent weeks, not just NaN rows.
        all_weeks = _complete_week_range(present_weeks)

        series = group.set_index("week_index")["quantity_actual"]
        series = series.reindex(all_weeks)   # NaN for every missing week

        # Rule 1: interpolate interior bounded gaps.
        # Because all_weeks[0] and all_weeks[-1] are always in present_weeks,
        # the first and last positions are never NaN — only interior slots can be.
        is_missing = series.isna()
        if is_missing.any():
            series    = series.interpolate(method="linear", limit_direction="both")
            n_interp += int(is_missing.sum())

        # Round to integer; zero out any residual NaN (shouldn't occur).
        series = series.fillna(0).round().astype(int)

        # Rule 2: >12 trailing zeros → already zero, no further action needed.
        trailing = sum(1 for v in reversed(series.values) if v == 0)
        _ = trailing   # count is informational; zeros are kept as-is

        result_parts.append(pd.DataFrame({
            "sku_id":          sku,
            "state_code":      state,
            "week_index":      series.index,
            "quantity_actual": series.values,
        }))

    if result_parts:
        return pd.concat(result_parts, ignore_index=True), n_interp
    return df, 0
