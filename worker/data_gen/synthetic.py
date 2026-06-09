"""
Synthetic dataset generator — BRD §11.

Produces:
  - sku_master.csv              (9 rows)
  - geo_master.csv              (12 rows)
  - sales_history.csv           (9 × 12 × 156 = 16,848 rows)
  - weather_data.json           (temp_deviation, correlated to demand, lag 0)
  - competitor_scrapes.csv      (competitor_price_index, leads Samsung demand by 2 weeks)
  - google_trends_export.csv    (search_trend_index, leads demand by 1 week)
  - actuals_holdout.csv         (last 26 weeks, held out for MAPE testing)

Fixed random seed = 42 for full reproducibility.
Dimensions: 9 SKUs × 12 states × 156 weeks (3 years).

File I/O lives only here (generator writes CSVs/JSON).
The caller then loads them via repo.import_csv().
"""

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED          = 42
rng           = np.random.default_rng(SEED)
random.seed(SEED)

N_WEEKS       = 156
HOLDOUT_WEEKS = 26   # last 26 weeks held out as actuals for MAPE testing

# ── SKU master ────────────────────────────────────────────────────────────────

SKUS = [
    {"sku_id": "SKU_ENT_01",  "sku_name": "Samsung Crystal 4K 32\"",  "product_tier": "entry",   "base_cost_inr": 18000},
    {"sku_id": "SKU_ENT_02",  "sku_name": "Samsung Crystal 4K 43\"",  "product_tier": "entry",   "base_cost_inr": 22000},
    {"sku_id": "SKU_MID_01",  "sku_name": "Samsung AU8000 55\"",       "product_tier": "mid",     "base_cost_inr": 38000},
    {"sku_id": "SKU_MID_02",  "sku_name": "Samsung AU8000 65\"",       "product_tier": "mid",     "base_cost_inr": 48000},
    {"sku_id": "SKU_UPR_01",  "sku_name": "Samsung QN85B 55\"",        "product_tier": "upper",   "base_cost_inr": 70000},
    {"sku_id": "SKU_UPR_02",  "sku_name": "Samsung QN85B 65\"",        "product_tier": "upper",   "base_cost_inr": 90000},
    {"sku_id": "SKU_PRM_01",  "sku_name": "Samsung Neo QLED 8K 65\"",  "product_tier": "premium", "base_cost_inr": 150000},
    {"sku_id": "SKU_PRM_02",  "sku_name": "Samsung Neo QLED 8K 75\"",  "product_tier": "premium", "base_cost_inr": 200000},
    {"sku_id": "SKU_LUX_01",  "sku_name": "Samsung The Frame 85\"",    "product_tier": "luxury",  "base_cost_inr": 350000},
]

# ── Geo master ────────────────────────────────────────────────────────────────

STATES = [
    {"state_code": "MH", "state_name": "Maharashtra",    "commercial_zone": "West",  "pop_weight": 1.8},
    {"state_code": "DL", "state_name": "Delhi",          "commercial_zone": "North", "pop_weight": 1.6},
    {"state_code": "KA", "state_name": "Karnataka",      "commercial_zone": "South", "pop_weight": 1.4},
    {"state_code": "TN", "state_name": "Tamil Nadu",     "commercial_zone": "South", "pop_weight": 1.3},
    {"state_code": "GJ", "state_name": "Gujarat",        "commercial_zone": "West",  "pop_weight": 1.2},
    {"state_code": "RJ", "state_name": "Rajasthan",      "commercial_zone": "North", "pop_weight": 1.0},
    {"state_code": "WB", "state_name": "West Bengal",    "commercial_zone": "East",  "pop_weight": 1.1},
    {"state_code": "UP", "state_name": "Uttar Pradesh",  "commercial_zone": "North", "pop_weight": 1.5},
    {"state_code": "AP", "state_name": "Andhra Pradesh", "commercial_zone": "South", "pop_weight": 0.9},
    {"state_code": "HR", "state_name": "Haryana",        "commercial_zone": "North", "pop_weight": 0.8},
    {"state_code": "KL", "state_name": "Kerala",         "commercial_zone": "South", "pop_weight": 0.7},
    {"state_code": "OR", "state_name": "Odisha",         "commercial_zone": "East",  "pop_weight": 0.6},
]

# Base weekly demand per tier (national)
TIER_BASE = {
    "entry":   850,
    "mid":     420,
    "upper":   180,
    "premium": 60,
    "luxury":  15,
}

# Long-term trend multiplier per tier over 156 weeks
# entry: declining -10%, mid: flat, upper: +15%, premium: +35%, luxury: +50%
TIER_TREND = {
    "entry":   -0.10,
    "mid":     0.00,
    "upper":   0.15,
    "premium": 0.35,
    "luxury":  0.50,
}

# ── Calendar helpers ──────────────────────────────────────────────────────────

def _build_weeks(n_weeks: int = 156, start_year: int = 2023, start_week: int = 1):
    """Return list of ISO 'YYYY-WW' strings."""
    import datetime
    weeks = []
    d = datetime.date.fromisocalendar(start_year, start_week, 1)
    for _ in range(n_weeks):
        iso = d.isocalendar()
        weeks.append(f"{iso.year}-W{iso.week:02d}")
        d += datetime.timedelta(weeks=1)
    return weeks


def _week_to_ordinal(week_str: str) -> int:
    """Convert 'YYYY-WW' to a 0-based integer position."""
    y, w = week_str.split("-W")
    return (int(y) - 2023) * 52 + (int(w) - 1)


def _seasonal_factor(week_ordinal: int) -> float:
    """Annual seasonality + festival spikes."""
    # Base sinusoidal seasonality (peak ~week 45 = Diwali region)
    annual_phase = 2 * math.pi * (week_ordinal % 52) / 52
    base_seasonal = 1.0 + 0.25 * math.sin(annual_phase - math.pi / 2)  # trough in summer

    # Week within year
    w_in_year = (week_ordinal % 52) + 1

    # Diwali spike (weeks 42–45, Oct/Nov)
    if 42 <= w_in_year <= 45:
        base_seasonal *= rng.uniform(2.4, 2.8)

    # Akshaya Tritiya (weeks 17–19, Apr/May)
    elif 17 <= w_in_year <= 19:
        base_seasonal *= rng.uniform(1.7, 1.9)

    # IPL window (weeks 12–22, Mar–May) — larger screen boost handled by sku modifier
    elif 12 <= w_in_year <= 22:
        base_seasonal *= rng.uniform(1.1, 1.2)

    # Year-end sales (weeks 50–52)
    elif 50 <= w_in_year <= 52:
        base_seasonal *= rng.uniform(1.3, 1.5)

    return max(0.1, base_seasonal)


def _promotion_uplift(week_ordinal: int, tier: str) -> float:
    """Approx 18% of weeks have promos (more near festivals)."""
    w_in_year = (week_ordinal % 52) + 1
    # Higher promo probability near festivals
    near_festival = (40 <= w_in_year <= 46) or (15 <= w_in_year <= 20)
    prob = 0.35 if near_festival else 0.12
    if rng.random() < prob:
        if tier in ("entry", "mid"):
            return rng.uniform(1.1, 1.35)
        else:
            return rng.uniform(1.05, 1.20)
    return 1.0


# ── Main generator ────────────────────────────────────────────────────────────

def generate(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    weeks = _build_weeks(N_WEEKS)

    # ── sku_master.csv ──────────────────────────────────────────────────────
    sku_df = pd.DataFrame([{**s, "is_active": 1} for s in SKUS])
    sku_df.to_csv(os.path.join(output_dir, "sku_master.csv"), index=False)
    print(f"[synth] sku_master: {len(sku_df)} rows")

    # ── geo_master.csv ──────────────────────────────────────────────────────
    geo_df = pd.DataFrame([{k: v for k, v in s.items() if k != "pop_weight"}
                            for s in STATES])
    geo_df["is_reporting"] = 1
    geo_df.to_csv(os.path.join(output_dir, "geo_master.csv"), index=False)
    print(f"[synth] geo_master: {len(geo_df)} rows")

    # ── sales_history.csv ───────────────────────────────────────────────────
    pop_map  = {s["state_code"]: s["pop_weight"] for s in STATES}
    records  = []
    n_weeks  = len(weeks)

    for sku in SKUS:
        tier       = sku["product_tier"]
        base_qty   = TIER_BASE[tier]
        trend_mult = TIER_TREND[tier]

        # IPL screen-size boost: larger screens (upper/premium/luxury) get bigger boost
        ipl_boost = 1.3 if tier in ("upper", "premium", "luxury") else 1.0

        for state in STATES:
            sc         = state["state_code"]
            pop_w      = state["pop_weight"]
            state_base = base_qty * pop_w

            for i, wk in enumerate(weeks):
                t          = i / n_weeks  # 0→1 over 3 years
                trend      = 1.0 + trend_mult * t
                seasonal   = _seasonal_factor(i)

                # IPL modifier
                w_in_year  = (i % 52) + 1
                ipl_factor = ipl_boost if 12 <= w_in_year <= 22 else 1.0

                promo      = _promotion_uplift(i, tier)

                # Lognormal noise — ensures MAPE ~8-18% band
                noise      = rng.lognormal(mean=0.0, sigma=0.12)

                qty_float  = state_base * trend * seasonal * ipl_factor * promo * noise
                qty        = max(0, round(qty_float))

                records.append({
                    "sku_id":          sku["sku_id"],
                    "state_code":      sc,
                    "week_index":      wk,
                    "quantity_actual": qty,
                })

    sales_df = pd.DataFrame(records)
    sales_df.to_csv(os.path.join(output_dir, "sales_history.csv"), index=False)
    print(f"[synth] sales_history: {len(sales_df)} rows")

    # ── Correlated signals ──────────────────────────────────────────────────
    # Generated AFTER sales so demand index can be computed from actual quantities.
    # RNG state at this point is deterministic (seed 42 + fixed loop order above).
    _write_signals(output_dir, weeks, sales_df)

    # ── Held-out actuals ────────────────────────────────────────────────────
    _write_actuals_holdout(output_dir, sales_df, weeks)

    print(f"[synth] All synthetic files written to {output_dir}")


# ── Signal generation ─────────────────────────────────────────────────────────

def _compute_state_demand_index(sales_df: pd.DataFrame) -> dict:
    """
    For each state, return a z-score normalized demand array of shape (n_weeks,),
    ordered by week_index ascending.  Used to derive correlated signal values.
    """
    agg = (sales_df
           .groupby(["state_code", "week_index"])["quantity_actual"]
           .sum()
           .reset_index()
           .sort_values(["state_code", "week_index"]))
    result = {}
    for sc in agg["state_code"].unique():
        vals = agg[agg["state_code"] == sc]["quantity_actual"].values.astype(float)
        mean, std = vals.mean(), vals.std()
        result[sc] = (vals - mean) / (std if std > 0 else 1.0)
    return result


def _india_temp_seasonal(week_idx: int) -> float:
    """Typical Indian temperature deviation from annual mean (°C).
    Peaks in May (~week 20), coldest in January (~week 2).
    """
    w = week_idx % 52
    return 6.0 * math.sin(2 * math.pi * (w - 2) / 52)


def _write_signals(output_dir: str, weeks: list, sales_df: pd.DataFrame) -> None:
    """
    Generate three correlated signal files after sales quantities are known.

    Lag semantics (all lags are relative to Samsung demand):
    - temp_deviation[t]          : correlated with demand[t]   (same week)
    - competitor_price_index[t]  : correlated with demand[t+2] (high comp price → Samsung gain 2w later)
    - search_trend_index[t]      : correlated with demand[t+1] (search precedes purchase by ~1 week)
    """
    demand_idx = _compute_state_demand_index(sales_df)
    n = len(weeks)

    weather_rows = []
    comp_rows    = []
    trend_rows   = []

    for s in STATES:
        sc = s["state_code"]
        d  = demand_idx[sc]   # z-score array, shape (n,)

        for i, wk in enumerate(weeks):
            # ── Temperature deviation (°C from seasonal norm) ─────────────
            # Physical: India heat waves (Apr–Jun) align with IPL/Akshaya Tritiya
            # demand peaks; festive season peaks (Oct–Nov) align with mild weather.
            # Signal leads demand by 0 weeks (same seasonal driver).
            temp = (_india_temp_seasonal(i)
                    + 1.5 * d[i]
                    + float(rng.normal(0.0, 0.8)))

            # ── Competitor price index (relative to Samsung) ──────────────
            # When competitors raise prices, consumers switch to Samsung ~2 weeks
            # later after comparison shopping.  Index > 1.0 means comp is pricier.
            future = d[min(i + 2, n - 1)]
            comp   = 1.0 + 0.06 * future + float(rng.normal(0.0, 0.025))
            comp   = float(np.clip(comp, 0.85, 1.20))

            # ── Google Search trend index (0–100) ─────────────────────────
            # Search interest precedes purchase by ~1 week.  Higher search this
            # week → higher Samsung demand next week.
            nxt    = d[min(i + 1, n - 1)]
            search = 50.0 + 20.0 * nxt + float(rng.normal(0.0, 5.0))
            search = float(np.clip(search, 10.0, 100.0))

            weather_rows.append({
                "state_code":    sc,
                "week_index":    wk,
                "temp_deviation": round(temp, 3),
            })
            comp_rows.append({
                "state_code":             sc,
                "week_index":             wk,
                "competitor_price_index": round(comp, 4),
            })
            trend_rows.append({
                "state_code":         sc,
                "week_index":         wk,
                "search_trend_index": round(search, 2),
            })

    with open(os.path.join(output_dir, "weather_data.json"), "w") as f:
        json.dump(weather_rows, f)
    pd.DataFrame(weather_rows).to_csv(
        os.path.join(output_dir, "weather_data.csv"), index=False)
    print(f"[synth] weather_data: {len(weather_rows)} rows")

    pd.DataFrame(comp_rows).to_csv(
        os.path.join(output_dir, "competitor_scrapes.csv"), index=False)
    print(f"[synth] competitor_scrapes: {len(comp_rows)} rows")

    pd.DataFrame(trend_rows).to_csv(
        os.path.join(output_dir, "google_trends_export.csv"), index=False)
    print(f"[synth] google_trends_export: {len(trend_rows)} rows")


def _write_actuals_holdout(output_dir: str, sales_df: pd.DataFrame,
                            weeks: list) -> None:
    """Write the last HOLDOUT_WEEKS of sales as actuals_holdout.csv.

    This file is loaded into the actuals table by ingestion and used to
    compute MAPE against the baseline/sensing forecasts.
    """
    holdout_week_set = set(weeks[-HOLDOUT_WEEKS:])
    holdout_df = sales_df[sales_df["week_index"].isin(holdout_week_set)].copy()
    holdout_df.to_csv(os.path.join(output_dir, "actuals_holdout.csv"), index=False)
    n_series = len(set(zip(holdout_df["sku_id"], holdout_df["state_code"])))
    print(f"[synth] actuals_holdout: {len(holdout_df)} rows "
          f"({HOLDOUT_WEEKS} weeks × {n_series} series)")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/data"
    generate(out)
