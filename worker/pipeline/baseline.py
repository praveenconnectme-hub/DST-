"""
Module 2 — Baseline with Champion-Challenger Selection (BRD §4.2, Phase 2).

Three model families compete per series:
  • Holt-Winters (HW)  — additive seasonal; handles regular demand patterns
  • Auto-ARIMA         — statsmodels grid-search over 8 candidate orders;
                         captures trend / autocorrelation without seasonal ARIMA
                         (D-015: pmdarima used when installed; statsmodels fallback
                         for builds without network access)
  • Croston            — intermittent demand only (zero_freq > SPARSE_THRESHOLD)

Selection rule: lowest holdout MAPE on leave-last-BACKTEST_WEEKS-out split.
HOLDOUT MAPE only — training error is never used for model selection (BRD §4.2).

Status lifecycle per series (in one transaction):
  • Winner  → status = 'champion'  + audit_log row (action = MODEL_CHAMPION_SELECTED)
  • Losers  → status = 'retired'   + audit_log row (action = MODEL_RETIRED)

All reads/writes go through the repository (Rule 1 / Standing Rule).
No sqlite3, no file I/O, no inline SQL in this file.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── pmdarima (preferred) or statsmodels fallback (D-015) ─────────────────────
try:
    from pmdarima import auto_arima as _pmd_auto_arima  # type: ignore
    _HAS_PMDARIMA = True
except ImportError:
    _HAS_PMDARIMA = False

# ── Constants ─────────────────────────────────────────────────────────────────

HORIZON              = 12     # weeks to forecast forward
BACKTEST_WEEKS       = 12     # leave-last-N-out holdout for champion selection
SEASONAL_PERIODS     = 52     # annual seasonality (weekly data)
MIN_TRAIN_WEEKS      = 16     # min rows required after setting aside BACKTEST_WEEKS
MAPE_FLAG_THRESHOLD  = 0.30   # flag accuracy row if mean MAPE >= 30 %
SPARSE_THRESHOLD     = 0.40   # zero_freq above this → Croston only

# ARIMA order grid (used when pmdarima unavailable)
_ARIMA_GRID = [
    (1, 1, 0), (0, 1, 1), (1, 1, 1),
    (2, 1, 0), (0, 1, 2), (0, 1, 0),
    (1, 0, 1), (2, 0, 0),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_id(family: str, sku_id: str, state_code: str) -> str:
    return f"{family}_{sku_id.lower()}_{state_code.lower()}_v1"


def _next_week_index(last_week: str, offset: int) -> str:
    import datetime as dt
    y, w   = last_week.split("-W")
    base   = dt.date.fromisocalendar(int(y), int(w), 1)
    future = base + dt.timedelta(weeks=offset)
    iso    = future.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _zero_freq(series: pd.Series) -> float:
    return float((series == 0).sum()) / max(1, len(series))


def _holdout_mape(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    return float(np.mean([
        abs(float(a) - float(f)) / max(1.0, float(a))
        for a, f in zip(actuals, forecasts)
    ]))


# ── Model fitters — return (bt_preds_on_train, hparams_dict) ─────────────────
# Each fitter is called ONCE on the train split.
# The champion is then refit separately on the full series via _refit_champion.

def _fit_hw(series: pd.Series) -> tuple:
    """
    Fit HW on `series`; return (bt_forecast_fn(n)->np.ndarray, hyperparams_dict).
    bt_forecast_fn is used to produce BACKTEST_WEEKS predictions.
    """
    seasonal = "add" if len(series) >= SEASONAL_PERIODS * 2 else None
    try:
        mdl = ExponentialSmoothing(
            series, trend="add", seasonal=seasonal,
            seasonal_periods=SEASONAL_PERIODS if seasonal else None,
            initialization_method="estimated",
        )
        fit = mdl.fit(optimized=True, remove_bias=True)
    except Exception:
        mdl = ExponentialSmoothing(series, trend="add", seasonal=None,
                                   initialization_method="estimated")
        fit = mdl.fit(optimized=True)
        seasonal = None

    hparams = {"trend": "add", "seasonal": seasonal,
               "seasonal_periods": SEASONAL_PERIODS if seasonal else None}
    return lambda n: np.maximum(0.0, fit.forecast(n).values), hparams


def _fit_arima(series: pd.Series) -> tuple:
    """
    Fit Auto-ARIMA on `series`.  Uses pmdarima if available, else grid-search (D-015).
    Returns (forecast_fn(n)->np.ndarray, hyperparams_dict).
    hyperparams_dict always contains 'order' so _refit_arima_full can reuse it.
    """
    if _HAS_PMDARIMA:
        fit = _pmd_auto_arima(
            series, seasonal=False, stepwise=True,
            max_p=3, max_q=3, max_d=2,
            information_criterion="aic",
            error_action="ignore", suppress_warnings=True,
        )
        order   = list(fit.order)
        hparams = {"order": order, "method": "pmdarima"}
        return lambda n: np.maximum(0.0, np.array(fit.predict(n_periods=n))), hparams

    # statsmodels grid-search fallback
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    best_aic, best_fit, best_order = float("inf"), None, None
    for order in _ARIMA_GRID:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = ARIMA(series, order=order).fit()
            if f.aic < best_aic:
                best_aic, best_fit, best_order = f.aic, f, order
        except Exception:
            continue
    if best_fit is None:
        raise ValueError("All ARIMA orders failed to converge")
    hparams = {"order": list(best_order), "aic": round(best_aic, 2),
               "method": "statsmodels_grid"}
    return lambda n: np.maximum(0.0, best_fit.forecast(n).values), hparams


def _croston_forecast(series: np.ndarray, n_steps: int,
                      alpha: float = 0.1) -> np.ndarray:
    """
    Classic Croston (1972) for intermittent demand series.
    Returns a constant forecast array of length n_steps.
    """
    nz_idx = np.where(series > 0)[0]
    if len(nz_idx) == 0:
        return np.zeros(n_steps)

    a    = float(series[nz_idx[0]])   # smoothed demand magnitude
    p    = 1.0                         # smoothed inter-demand interval
    last = nz_idx[0]

    for t in range(nz_idx[0] + 1, len(series)):
        if series[t] > 0:
            a    = alpha * float(series[t]) + (1 - alpha) * a
            p    = alpha * (t - last)       + (1 - alpha) * p
            last = t

    return np.full(n_steps, max(0.0, a / max(0.01, p)))


def _refit_champion_full(family: str, hparams: dict,
                         full_series: pd.Series) -> np.ndarray:
    """
    Refit the champion model on the FULL series and return its HORIZON-step forecast.
    Uses saved hyperparams to avoid re-running model selection.
    """
    if family == "holt_winters":
        seasonal = hparams.get("seasonal")
        try:
            mdl = ExponentialSmoothing(
                full_series, trend="add", seasonal=seasonal,
                seasonal_periods=SEASONAL_PERIODS if seasonal else None,
                initialization_method="estimated",
            )
            fit = mdl.fit(optimized=True, remove_bias=True)
        except Exception:
            mdl = ExponentialSmoothing(full_series, trend="add", seasonal=None,
                                       initialization_method="estimated")
            fit = mdl.fit(optimized=True)
        return np.maximum(0.0, fit.forecast(HORIZON).values)

    elif family == "auto_arima":
        order = tuple(hparams["order"])
        if _HAS_PMDARIMA:
            fit = _pmd_auto_arima(
                full_series, seasonal=False, stepwise=True,
                max_p=3, max_q=3, max_d=2,
                information_criterion="aic",
                error_action="ignore", suppress_warnings=True,
            )
            return np.maximum(0.0, np.array(fit.predict(n_periods=HORIZON)))
        else:
            from statsmodels.tsa.arima.model import ARIMA
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ARIMA(full_series, order=order).fit()
            return np.maximum(0.0, fit.forecast(HORIZON).values)

    elif family == "croston":
        return _croston_forecast(full_series.values, HORIZON,
                                 alpha=hparams.get("alpha", 0.1))

    raise ValueError(f"Unknown family: {family}")


# ── Main entry point ──────────────────────────────────────────────────────────

def run(repo) -> dict:
    """
    Run champion-challenger baseline selection for all active sku×state series.

    For each series:
      • sparse (zero_freq > SPARSE_THRESHOLD): Croston only → champion by default.
      • non-sparse: HW + Auto-ARIMA compete; champion = lowest holdout MAPE.

    Each model status write (champion / retired) is paired with an audit_log
    row in the same transaction per BRD Standing Rule 1.
    """
    skus   = repo.query("sku_master",  filters={"is_active": 1})
    states = repo.query("geo_master",  filters={"is_reporting": 1})

    champion_mix      = {"holt_winters": 0, "auto_arima": 0, "croston": 0}
    per_model_mapes   = {"holt_winters": [], "auto_arima": [], "croston": []}
    champ_mapes_list  = []
    skip_log          = []
    n_skipped         = 0
    n_forecast_rows   = 0

    for sku in skus:
        for state in states:
            sku_id     = sku["sku_id"]
            state_code = state["state_code"]

            rows = repo.query(
                "sales_history",
                filters={"sku_id": sku_id, "state_code": state_code},
                order_by=["week_index"],
            )

            min_rows = MIN_TRAIN_WEEKS + BACKTEST_WEEKS
            if len(rows) < min_rows:
                reason = (f"insufficient history "
                          f"({len(rows)} weeks < {min_rows} required)")
                skip_log.append({"sku_id": sku_id, "state_code": state_code,
                                  "reason": reason})
                n_skipped += 1
                print(f"[baseline] SKIP {sku_id}×{state_code}: {reason}")
                continue

            df   = pd.DataFrame(rows).sort_values("week_index").reset_index(drop=True)
            full = df["quantity_actual"].astype(float)
            n    = len(full)

            train_series = full.iloc[:n - BACKTEST_WEEKS]
            test_vals    = full.iloc[n - BACKTEST_WEEKS:].values
            bt_weeks     = df["week_index"].iloc[n - BACKTEST_WEEKS:].tolist()
            zf           = _zero_freq(train_series)
            train_win    = f"{df['week_index'].iloc[0]}..{df['week_index'].iloc[-1]}"

            # ── Fit candidates on train split ─────────────────────────────────
            # Each entry: (family, holdout_mape, bt_preds, hparams)
            candidates: list = []

            if zf > SPARSE_THRESHOLD:
                # Intermittent series: Croston only
                try:
                    bt = _croston_forecast(train_series.values, BACKTEST_WEEKS)
                    m  = _holdout_mape(test_vals, bt)
                    hparams = {"alpha": 0.1, "zero_freq": round(zf, 3)}
                    candidates.append(("croston", m, bt, hparams))
                    per_model_mapes["croston"].append(m)
                except Exception as exc:
                    skip_log.append({"sku_id": sku_id, "state_code": state_code,
                                     "reason": f"Croston failed: {exc}"})
                    n_skipped += 1
                    print(f"[baseline] SKIP {sku_id}×{state_code}: Croston: {exc}")
                    continue
            else:
                # Regular series: HW + ARIMA
                for family, fit_fn in [("holt_winters", _fit_hw),
                                       ("auto_arima",   _fit_arima)]:
                    try:
                        fcast_fn, hparams = fit_fn(train_series)
                        bt   = fcast_fn(BACKTEST_WEEKS)
                        mape = _holdout_mape(test_vals, bt)
                        candidates.append((family, mape, bt, hparams))
                        per_model_mapes[family].append(mape)
                    except Exception as exc:
                        print(f"[baseline] {family} failed for "
                              f"{sku_id}×{state_code}: {exc}")

                if not candidates:
                    skip_log.append({"sku_id": sku_id, "state_code": state_code,
                                     "reason": "all non-sparse models failed"})
                    n_skipped += 1
                    continue

            # ── Select champion: lowest holdout MAPE ──────────────────────────
            candidates.sort(key=lambda c: c[1])
            champ_family, champ_mape, champ_bt, champ_hp = candidates[0]
            retired = candidates[1:]
            champ_mid = _model_id(champ_family, sku_id, state_code)

            champion_mix[champ_family] += 1
            champ_mapes_list.append(champ_mape)
            flagged = 1 if champ_mape >= MAPE_FLAG_THRESHOLD else 0

            # ── Accuracy metrics (all candidates, each tagged with model_id) ──
            acc_rows = []
            for family, _, bt_preds, _ in candidates:
                mid = _model_id(family, sku_id, state_code)
                for wk, actual, pred in zip(bt_weeks, test_vals, bt_preds):
                    denom  = max(1.0, float(actual))
                    ape    = abs(float(actual) - float(pred)) / denom
                    bias_v = (float(pred) - float(actual)) / denom
                    acc_rows.append({
                        "sku_id":              sku_id,
                        "state_code":          state_code,
                        "week_index":          wk,
                        "model_id":            mid,
                        "mape":                round(ape, 6),
                        "bias":                round(bias_v, 6),
                        "flagged_for_retrain": flagged if family == champ_family else 0,
                    })

            # ── model_registry rows ───────────────────────────────────────────
            champ_bias = float(np.mean([
                (float(p) - float(a)) / max(1.0, float(a))
                for a, p in zip(test_vals, champ_bt)
            ]))
            reg_rows = [{
                "model_id":         champ_mid,
                "model_type":       champ_family,
                "scope":            f"{sku_id}_{state_code}",
                "status":           "champion",
                "trained_at":       _now_iso(),
                "train_window":     train_win,
                "hyperparams_json": json.dumps(champ_hp),
                "val_mape":         round(champ_mape, 6),
                "val_bias":         round(champ_bias, 6),
                "feature_set_json": json.dumps(["quantity_actual"]),
                "artifact_path":    None,
                "parent_model_id":  None,
            }] + [
                {
                    "model_id":         _model_id(f, sku_id, state_code),
                    "model_type":       f,
                    "scope":            f"{sku_id}_{state_code}",
                    "status":           "retired",
                    "trained_at":       _now_iso(),
                    "train_window":     train_win,
                    "hyperparams_json": json.dumps(hp),
                    "val_mape":         round(m, 6),
                    "val_bias":         None,
                    "feature_set_json": json.dumps(["quantity_actual"]),
                    "artifact_path":    None,
                    "parent_model_id":  None,
                }
                for f, m, _, hp in retired
            ]

            # ── Audit log: one row per status change ──────────────────────────
            audit_rows = [
                {
                    "timestamp":   _now_iso(),
                    "actor":       "system",
                    "action":      "MODEL_CHAMPION_SELECTED",
                    "entity":      "model_registry",
                    "detail_json": json.dumps({
                        "model_id":   champ_mid,
                        "model_type": champ_family,
                        "scope":      f"{sku_id}_{state_code}",
                        "val_mape":   round(champ_mape, 6),
                    }),
                }
            ] + [
                {
                    "timestamp":   _now_iso(),
                    "actor":       "system",
                    "action":      "MODEL_RETIRED",
                    "entity":      "model_registry",
                    "detail_json": json.dumps({
                        "model_id":   _model_id(f, sku_id, state_code),
                        "model_type": f,
                        "scope":      f"{sku_id}_{state_code}",
                        "val_mape":   round(m, 6),
                    }),
                }
                for f, m, _, _ in retired
            ]

            # ── Refit champion on FULL series; generate forward forecast ──────
            try:
                fwd_vals = _refit_champion_full(champ_family, champ_hp, full)
            except Exception as exc:
                print(f"[baseline] Full refit failed for {sku_id}×{state_code}: {exc}, "
                      "using backtest fit extrapolation")
                fwd_vals = np.full(HORIZON, max(0.0, float(np.mean(champ_bt))))

            last_week  = df["week_index"].iloc[-1]
            fcast_rows = [
                {
                    "sku_id":       sku_id,
                    "state_code":   state_code,
                    "week_index":   _next_week_index(last_week, i),
                    "forecast_qty": float(v),
                    "model_id":     champ_mid,
                }
                for i, v in enumerate(fwd_vals, start=1)
            ]
            n_forecast_rows += len(fcast_rows)

            # ── Per-series transaction: model_registry (FK anchor first) ─────
            with repo.transaction():
                repo.upsert("model_registry", reg_rows)
                repo.upsert("audit_log",      audit_rows)
                if acc_rows:
                    repo.upsert("accuracy_metrics", acc_rows)
                repo.upsert("baseline_forecast", fcast_rows)

            print(f"[baseline] {sku_id}×{state_code} → champion={champ_family} "
                  f"mape={champ_mape*100:.1f}%")

    # ── Champion MAPE distribution ────────────────────────────────────────────
    mape_pct     = np.array(champ_mapes_list) * 100.0
    distribution = {}
    if len(mape_pct) > 0:
        distribution = {
            "n_series":   int(len(mape_pct)),
            "min_pct":    round(float(np.min(mape_pct)),    1),
            "median_pct": round(float(np.median(mape_pct)), 1),
            "p90_pct":    round(float(np.percentile(mape_pct, 90)), 1),
            "max_pct":    round(float(np.max(mape_pct)), 1),
            "n_over_30":  int(np.sum(mape_pct >= 30.0)),
        }

    per_model_medians = {
        k: round(float(np.median(v)) * 100, 1) if v else None
        for k, v in per_model_mapes.items()
    }

    print(f"\n[baseline] Champion mix:                {champion_mix}")
    print(f"[baseline] Per-model holdout MAPE (%):  {per_model_medians}")
    print(f"[baseline] Champion MAPE distribution:  {distribution}")

    with repo.transaction():
        repo.upsert("audit_log", [{
            "timestamp":   _now_iso(),
            "actor":       "system",
            "action":      "BASELINE_COMPLETE",
            "entity":      "baseline_forecast",
            "detail_json": json.dumps({
                "n_forecasted":       len(champ_mapes_list),
                "n_skipped":          n_skipped,
                "champion_mix":       champion_mix,
                "per_model_medians":  per_model_medians,
                **distribution,
            }),
        }])

    return {
        "forecasted_series":          len(champ_mapes_list),
        "skipped_series":             n_skipped,
        "total_forecast_rows":        n_forecast_rows,
        "champion_mix":               champion_mix,
        "per_model_mape_median_pct":  per_model_medians,
        "champion_mape_distribution": distribution,
        "skip_log":                   skip_log,
    }
