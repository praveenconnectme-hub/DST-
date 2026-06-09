"""
Module 4 — XGBoost demand sensing (BRD §4.3, Phase 2).

D-012: One XGBoost model per product tier (5 tiers: entry / mid / upper / premium / luxury).
       sku_id and state_code are label-encoded categorical features in each model.

Anti-leakage lag table — every feature at target week W uses ONLY data from week < W:

  Feature          | Source week | Lag | Rationale
  -----------------+-------------+-----+--------------------------------------------------
  temp_lag1        | W − 1       |  1  | temp[t] ~ demand[t] (same-week); lag-1 prevents
                   |             |     | look-ahead entirely
  comp_lag2        | W − 2       |  2  | comp[t] ~ demand[t+2]; comp[W-2] ~ demand[W]
                   |             |     | and is available before forecast time
  search_lag1      | W − 1       |  1  | search[t] ~ demand[t+1]; search[W-1] ~ demand[W]
                   |             |     | and is available before forecast time
  qty_lag1/2/4     | W − 1/2/4   | 1-4 | Past actuals, always pre-available
  qty_roll4_mean   | W-4 … W-1   |  1  | Trailing 4-week mean, no same-week exposure
  week_of_year     | derived     |  0  | Calendar feature — no data leakage
  sku_id_enc       | static      |  0  | LabelEncoded identity feature
  state_code_enc   | static      |  0  | LabelEncoded identity feature

Note: baseline_forecast_qty is NOT a training feature — the baseline_forecast table only
holds the last HOLDOUT_WEEKS rows per series; there are no in-sample baseline forecasts
for the 144 training weeks, so including it would cause structural NaN leakage.

Train window : weeks 5..144  (ordinals 0-143; first 4 omitted for qty_lag4 warmup)
Holdout window: weeks 145..156 (ordinals 144-155; last HOLDOUT_WEEKS weeks)
Target        : quantity_actual[W]
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ── Constants ──────────────────────────────────────────────────────────────────

TIERS          = ["entry", "mid", "upper", "premium", "luxury"]
HOLDOUT_WEEKS  = 12     # matches baseline BACKTEST_WEEKS for apples-to-apples MAPE comparison
MAPE_THRESHOLD = 0.30   # D-014 flag: series MAPE ≥ 30 % exercises G3 retraining path

FEATURE_COLS = [
    "qty_lag1",        # actual demand t-1
    "qty_lag2",        # actual demand t-2
    "qty_lag4",        # actual demand t-4
    "qty_roll4_mean",  # 4-week trailing mean ending at t-1
    "temp_lag1",       # temp_deviation from week t-1
    "comp_lag2",       # competitor_price_index from week t-2
    "search_lag1",     # search_trend_index from week t-1
    "week_of_year",    # 1-52 seasonality proxy (no data leakage)
    "sku_id_enc",      # LabelEncoded sku_id
    "state_code_enc",  # LabelEncoded state_code
]

# Lag per signal name: weeks before target week W to read the signal value
SIGNAL_LAGS = {
    "temp_deviation":         1,
    "competitor_price_index": 2,
    "search_trend_index":     1,
}

_XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, objective="reg:squarederror", verbosity=0,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_week_maps(sales_df: pd.DataFrame):
    """Return (sorted_weeks_list, {week_index: ordinal}) from a sales DataFrame."""
    weeks_list = sorted(sales_df["week_index"].unique())
    w2i = {w: i for i, w in enumerate(weeks_list)}
    return weeks_list, w2i


def _shift_signal_col(sig_df: pd.DataFrame, value_col: str,
                      lag: int, w2i: dict, weeks_list: list) -> pd.DataFrame:
    """
    Shift a signal column forward by `lag` weeks so that joining on (state_code, week_index)
    maps each target week W to the signal value from week W - lag.

    Mechanism:
      sig_df has rows (state, week=t, value).
      After shift: rows become (state, week=t+lag, value).
      Join to sales on week=W → fetches value from signal week W-lag. ✓

    Example (lag=1):
      Signal row (MH, W03, 3.0) → becomes (MH, W04, 3.0).
      Joined to sales at W04 → feature temp_lag1 = 3.0 = temp[W03] = temp[W-1]. ✓
    """
    df = sig_df[["state_code", "week_index", value_col]].copy()
    df["ord"] = df["week_index"].map(w2i)
    df["target_ord"] = df["ord"] + lag
    # Drop rows where target week is beyond the data window
    df = df[df["target_ord"] < len(weeks_list)].copy()
    df["week_index"] = df["target_ord"].map(lambda o: weeks_list[o])
    return df[["state_code", "week_index", value_col]]


def assemble_features(repo) -> tuple:
    """
    Read sales, signals, and SKU master from the repository; apply lags; return features.

    Anti-leakage guarantee: every feature value for target row (sku, state, week=W)
    derives from data strictly before week W.  See SIGNAL_LAGS and the docstring above.

    Returns
    -------
    df         : DataFrame with FEATURE_COLS + 'quantity_actual' + metadata
                 ('sku_id', 'state_code', 'week_index', 'week_ord', 'product_tier')
    weeks_list : sorted list of week_index strings present in sales_history
    w2i        : {week_index -> ordinal} mapping
    """
    sales = repo.read_frame("sales_history")   # sku_id, state_code, week_index, quantity_actual
    sigs  = repo.read_frame("signal_data")     # signal_name, state_code, week_index, value
    skus  = repo.read_frame("sku_master")      # sku_id, product_tier, ...

    weeks_list, w2i = _build_week_maps(sales)

    # ── Sales lags (per sku×state group) ──────────────────────────────────────
    sales = sales.sort_values(["sku_id", "state_code", "week_index"]).copy()
    sales["week_ord"] = sales["week_index"].map(w2i)

    g = sales.groupby(["sku_id", "state_code"], sort=False)
    sales["qty_lag1"]       = g["quantity_actual"].shift(1)
    sales["qty_lag2"]       = g["quantity_actual"].shift(2)
    sales["qty_lag4"]       = g["quantity_actual"].shift(4)

    # qty_roll4_mean: trailing 4-week mean of quantity ending at t-1 (no look-ahead).
    #
    # Computed as GroupBy.rolling on the already-shifted qty_lag1 column instead of
    # transform(lambda s: s.shift(1).rolling(...).mean()).  The lambda form uses
    # pandas' Python slow path which, in pandas 2.2.x, can produce object-dtype output
    # when the lambda crosses an int64→float64 dtype boundary (shift introduces NaN);
    # the per-group result objects are then stored as numpy array cells rather than
    # scalar floats, and numpy's astype(float) raises on them.  The GroupBy.rolling
    # approach uses the cython fast path throughout and always emits float64.
    sales["qty_roll4_mean"] = (
        sales["qty_lag1"]                                          # float64, already shifted
        .groupby([sales["sku_id"], sales["state_code"]])
        .rolling(4, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)                     # drop group-key index levels
    )

    sales["week_of_year"] = sales["week_index"].apply(lambda w: int(w.split("-W")[1]))

    # ── Signal lags (vectorised join, no row-by-row apply) ────────────────────
    if len(sigs) > 0:
        sigs_wide = sigs.pivot_table(
            index=["state_code", "week_index"],
            columns="signal_name",
            values="value",
            aggfunc="first",
        ).reset_index()
    else:
        sigs_wide = pd.DataFrame(columns=["state_code", "week_index",
                                          "temp_deviation", "competitor_price_index",
                                          "search_trend_index"])

    # Ensure all expected signal columns are present
    for col in ["temp_deviation", "competitor_price_index", "search_trend_index"]:
        if col not in sigs_wide.columns:
            sigs_wide[col] = np.nan

    temp_shifted   = _shift_signal_col(sigs_wide, "temp_deviation",         1, w2i, weeks_list)
    comp_shifted   = _shift_signal_col(sigs_wide, "competitor_price_index",  2, w2i, weeks_list)
    search_shifted = _shift_signal_col(sigs_wide, "search_trend_index",      1, w2i, weeks_list)

    sales = (sales
             .merge(temp_shifted.rename(columns={"temp_deviation":         "temp_lag1"}),
                    on=["state_code", "week_index"], how="left")
             .merge(comp_shifted.rename(columns={"competitor_price_index": "comp_lag2"}),
                    on=["state_code", "week_index"], how="left")
             .merge(search_shifted.rename(columns={"search_trend_index":   "search_lag1"}),
                    on=["state_code", "week_index"], how="left"))

    # ── Join product tier ──────────────────────────────────────────────────────
    sales = sales.merge(skus[["sku_id", "product_tier"]], on="sku_id", how="left")

    # ── Numeric guard ──────────────────────────────────────────────────────────
    # Validate every feature column that is present at this stage.
    # (sku_id_enc / state_code_enc are added later in run() and are not checked
    # here.)  If a column has object dtype, any value that pd.to_numeric cannot
    # coerce is a hard error — reported with the column name and sample values so
    # the root cause is immediately actionable rather than surfacing as a
    # cryptic XGBoost "could not convert string to float" deep in training.
    # Non-object numeric columns (Int64, Float64 nullable types) are normalised
    # to plain float64 so .values.astype(float) in the training loop always
    # receives a clean numpy array.
    _cols_present = [c for c in FEATURE_COLS if c in sales.columns]
    for col in _cols_present:
        if sales[col].dtype == object:
            coerced = pd.to_numeric(sales[col], errors="coerce")
            bad = sales.loc[coerced.isna() & sales[col].notna(), col]
            if len(bad) > 0:
                raise ValueError(
                    f"assemble_features: non-numeric values in feature column "
                    f"'{col}' — cannot coerce to float. "
                    f"Sample bad values: {bad.head(3).tolist()}"
                )
            sales[col] = coerced.astype(float)
        else:
            sales[col] = pd.to_numeric(sales[col], errors="raise").astype(float)

    return sales, weeks_list, w2i


def run(repo) -> dict:
    """
    Train one XGBoost model per product tier; score on holdout; persist output.

    Writes to:
      - model_registry   : one xgboost_<tier>_v1 row per tier, status='champion'
      - audit_log        : MODEL_CHAMPION_SELECTED per tier (same transaction)
      - demand_sensing_output : one row per sku×state×holdout_week with feature-contribution JSON

    Returns a summary dict including:
      - sensing vs baseline MAPE comparison
      - D-014 diagnostic (n series over 30 % MAPE on sensing holdout)
      - top SHAP feature contributions
    """
    print("[sensing] Assembling feature matrix ...")
    df, weeks_list, w2i = assemble_features(repo)

    n_total           = len(weeks_list)
    holdout_start_ord = n_total - HOLDOUT_WEEKS   # e.g. 144 for 156-week dataset

    # ── Label-encode categoricals (fit on full dataset, never unseen labels) ───
    sku_le   = LabelEncoder().fit(sorted(df["sku_id"].unique()))
    state_le = LabelEncoder().fit(sorted(df["state_code"].unique()))
    df = df.copy()
    df["sku_id_enc"]     = sku_le.transform(df["sku_id"])
    df["state_code_enc"] = state_le.transform(df["state_code"])

    # ── Drop NaN rows (lag warmup; also absent signals fill as NaN) ────────────
    df_clean = df.dropna(subset=FEATURE_COLS).copy()

    train_df   = df_clean[df_clean["week_ord"] < holdout_start_ord].copy()
    holdout_df = df_clean[df_clean["week_ord"] >= holdout_start_ord].copy()

    print(f"[sensing] Feature matrix: train={len(train_df)} rows, "
          f"holdout={len(holdout_df)} rows, features={len(FEATURE_COLS)}")

    all_row_apes    = []
    all_output_rows = []
    tier_summaries  = {}
    shap_mean_abs   = np.zeros(len(FEATURE_COLS))  # accumulated across tiers
    n_tiers_trained = 0
    n_over_30_total = 0

    for tier in TIERS:
        tr = train_df[train_df["product_tier"] == tier].reset_index(drop=True)
        ho = holdout_df[holdout_df["product_tier"] == tier].reset_index(drop=True)

        if len(tr) < 20 or len(ho) == 0:
            print(f"[sensing] SKIP tier={tier}: train={len(tr)}, holdout={len(ho)}")
            continue

        # ── Feature matrix: column-by-column conversion for precise failure ──
        # On the first column that can't be coerced to float, log the column
        # name, the offending value(s), and their Python type, then re-raise
        # so the stack trace still points here.
        def _to_float_arr(frame, frame_label):
            for _col in FEATURE_COLS:
                try:
                    pd.to_numeric(frame[_col], errors="raise")
                except (ValueError, TypeError) as _exc:
                    _bad = frame.loc[
                        pd.to_numeric(frame[_col], errors="coerce").isna()
                        & frame[_col].notna(),
                        _col,
                    ]
                    print(
                        f"[sensing] DIAG tier={tier} {frame_label} "
                        f"CONVERSION FAILED  col={_col!r}  "
                        f"error={_exc!r}  "
                        f"bad_vals={_bad.head(3).tolist()}  "
                        f"bad_types={[type(v).__name__ for v in _bad.head(3)]}"
                    )
                    raise
            return frame[FEATURE_COLS].values.astype(float)

        # ── Target vector: flatten to 1-D float64 (defensive against array cells) ──
        def _to_y_array(series, frame_label):
            arr = np.asarray(series)
            if arr.ndim > 1:
                arr = arr.ravel()
            if arr.dtype == object:
                flat = []
                for v in arr:
                    a = np.asarray(v)
                    flat.append(float(a.ravel()[0]) if a.ndim > 0 else float(v))
                arr = np.array(flat, dtype=float)
            else:
                arr = arr.astype(float)
            if arr.ndim != 1:
                raise ValueError(
                    f"[sensing] target y is not 1-D after coercion "
                    f"tier={tier} {frame_label}: shape={arr.shape}"
                )
            return arr

        X_train = _to_float_arr(tr, "train")
        y_train = _to_y_array(tr["quantity_actual"], "train")
        X_hold  = _to_float_arr(ho, "holdout")
        y_hold  = _to_y_array(ho["quantity_actual"], "holdout")

        # ── Train ─────────────────────────────────────────────────────────────
        assert y_train.ndim == 1 and np.issubdtype(y_train.dtype, np.floating), (
            f"[sensing] y_train not 1-D float for tier={tier}: "
            f"ndim={y_train.ndim}, dtype={y_train.dtype}"
        )
        base_score = float(np.mean(y_train))
        print(f"[sensing] Training tier={tier}: {len(tr)} train rows ...")
        model = xgb.XGBRegressor(**_XGB_PARAMS, base_score=base_score)
        model.fit(X_train, y_train)

        # ── Holdout predictions (no negatives) ────────────────────────────────
        y_pred = np.maximum(0.0, model.predict(X_hold))

        # ── Feature contributions via XGBoost native pred_contribs ──────────────
        # pred_contribs=True returns (n_holdout, n_features + 1): each column
        # 0..n_features-1 is the per-feature contribution (same semantics as
        # SHAP TreeExplainer values); the LAST column is the bias/base term.
        # shap.TreeExplainer is incompatible with XGBoost 3.x: it reads
        # base_score from the booster JSON config in the format '[1.185077E3]'
        # (changed in 3.x) and calls float() on the raw string, raising
        # ValueError before any prediction happens.
        contribs  = model.get_booster().predict(
            xgb.DMatrix(X_hold), pred_contribs=True
        )
        shap_mat  = contribs[:, :-1]           # drop bias column → (n_holdout, n_features)
        shap_mean_abs += np.mean(np.abs(shap_mat), axis=0)
        n_tiers_trained += 1

        # ── Holdout APE per row ───────────────────────────────────────────────
        row_apes = np.abs(y_hold - y_pred) / np.maximum(y_hold, 1.0)
        all_row_apes.extend(row_apes.tolist())
        tier_mape = float(np.mean(row_apes))
        tier_bias = float(np.mean(y_pred - y_hold))

        # ── D-014 check: per-series MAPE on holdout ───────────────────────────
        n_over_30 = 0
        for (_, _), grp in ho.groupby(["sku_id", "state_code"]):
            idx          = grp.index.tolist()
            series_mape  = float(np.mean(row_apes[idx]))
            if series_mape >= MAPE_THRESHOLD:
                n_over_30 += 1
        n_over_30_total += n_over_30

        # ── model_registry + audit_log (same transaction) ─────────────────────
        model_id = f"xgboost_{tier}_v1"
        reg_row  = {
            "model_id":         model_id,
            "model_type":       "xgboost",
            "scope":            f"tier_{tier}",
            "status":           "champion",
            "trained_at":       _now_iso(),
            "train_window":     (f"{weeks_list[4]}.."
                                 f"{weeks_list[holdout_start_ord - 1]}"),
            "hyperparams_json": json.dumps(_XGB_PARAMS),
            "val_mape":         round(tier_mape, 6),
            "val_bias":         round(tier_bias, 2),
            "feature_set_json": json.dumps(FEATURE_COLS),
            "artifact_path":    None,
            "parent_model_id":  None,
        }
        audit_row = {
            "timestamp":   _now_iso(),
            "actor":       "sensing_module",
            "action":      "MODEL_CHAMPION_SELECTED",
            "entity":      "model_registry",
            "detail_json": json.dumps({
                "model_id": model_id,
                "tier":     tier,
                "val_mape": round(tier_mape, 6),
            }),
        }
        with repo.transaction():
            repo.upsert("model_registry", [reg_row])
            repo.upsert("audit_log",      [audit_row])

        # ── demand_sensing_output rows ─────────────────────────────────────────
        for i in range(len(ho)):
            shap_dict = {FEATURE_COLS[j]: round(float(shap_mat[i, j]), 4)
                         for j in range(len(FEATURE_COLS))}
            all_output_rows.append({
                "sku_id":       ho.iloc[i]["sku_id"],
                "state_code":   ho.iloc[i]["state_code"],
                "week_index":   ho.iloc[i]["week_index"],
                "forecast_qty": round(float(y_pred[i]), 2),
                "model_id":     model_id,
                "shap_json":    json.dumps(shap_dict),
            })

        tier_summaries[tier] = {
            "mape_pct":   round(tier_mape * 100, 1),
            "n_train":    int(len(tr)),
            "n_holdout":  int(len(ho)),
            "n_over_30":  int(n_over_30),
        }
        print(f"[sensing] tier={tier}  holdout MAPE={tier_mape*100:.1f}%  "
              f"n_series_over_30%={n_over_30}")

    # ── Persist demand_sensing_output ──────────────────────────────────────────
    if all_output_rows:
        repo.upsert("demand_sensing_output", all_output_rows)

    # ── Baseline MAPE for comparison (from model_registry champion rows) ───────
    baseline_champ_rows = [
        r for r in repo.query("model_registry", filters={"status": "champion"})
        if r.get("model_type") not in ("xgboost", None)
        and r.get("val_mape") is not None
    ]
    baseline_mapes   = [r["val_mape"] for r in baseline_champ_rows]
    baseline_median  = (round(float(np.median(baseline_mapes)) * 100, 1)
                        if baseline_mapes else None)
    overall_sensing  = (round(float(np.mean(all_row_apes)) * 100, 1)
                        if all_row_apes else None)

    # ── Top SHAP features (mean |shap| across tiers, normalised) ──────────────
    if n_tiers_trained > 0:
        avg_shap = shap_mean_abs / n_tiers_trained
        shap_ranking = sorted(zip(FEATURE_COLS, avg_shap), key=lambda x: -x[1])
        top_shap = [(f, round(float(v), 4)) for f, v in shap_ranking[:5]]
    else:
        top_shap = []

    print(f"\n[sensing] ── Summary ────────────────────────────────────────────────")
    print(f"[sensing] Overall sensing MAPE  : {overall_sensing} %")
    print(f"[sensing] Baseline median MAPE  : {baseline_median} %")
    print(f"[sensing] D-014 series > 30 %   : {n_over_30_total}")
    print(f"[sensing] Top SHAP features     : {top_shap}")
    print(f"[sensing] demand_sensing_output : {len(all_output_rows)} rows")

    if overall_sensing is not None and overall_sensing < 5.0:
        print("[sensing] !! WARNING: sensing MAPE < 5 % — "
              "investigate possible target leakage before declaring success !!")

    return {
        "sensing_output_rows":      len(all_output_rows),
        "overall_sensing_mape_pct": overall_sensing,
        "baseline_median_mape_pct": baseline_median,
        "n_series_over_30pct":      n_over_30_total,
        "tier_summaries":           tier_summaries,
        "top_shap_features":        top_shap,
    }
