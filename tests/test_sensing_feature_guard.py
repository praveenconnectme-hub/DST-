"""
Regression tests for the numeric guard in assemble_features() — sensing.py.

Root cause of the guarded bug:
  Under macOS Docker bind-mounts (VirtioFS / osxfs), concurrent api+worker
  SQLite WAL access produced ERRNO 35 (Resource deadlock avoided) which could
  corrupt a partial read so that a REAL column value arrived in Python as an
  object (string) instead of a float.  The specific observed value was
  '[1.185077E3]' — a string that pandas cannot silently coerce, causing XGBoost
  training to fail at X_train = tr[FEATURE_COLS].values.astype(float) with no
  indication of which column was corrupt.

  Root-cause fix: move SQLite to a Docker named volume (D-023).
  Defensive fix (this guard): assemble_features() now validates every feature
  column that is present at assembly time and raises ValueError with the
  offending column name and sample values.

Run:  python -m pytest tests/test_sensing_feature_guard.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

import numpy as np
import pandas as pd
import pytest

from migrations.migration_001 import run as apply_migration
from repository.factory import RepositoryFactory
from pipeline.sensing import assemble_features, FEATURE_COLS
from pipeline import ingestion as ingestion_module
from pipeline import signals as signals_module


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _make_repo(tmp_path):
    db = str(tmp_path / "guard_test.db")
    apply_migration(db)
    return RepositoryFactory.create({"type": "sqlite", "db_path": db})


def _write_fixture_csvs(tmp_path, n_weeks=40):
    """Write minimal CSV fixtures for ingestion + signals (D-009 carve-out)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    SKUS = [
        {"sku_id": "E1", "sku_name": "E1", "product_tier": "entry",
         "base_cost_inr": 1000, "is_active": 1},
        {"sku_id": "M1", "sku_name": "M1", "product_tier": "mid",
         "base_cost_inr": 2000, "is_active": 1},
    ]
    STATES = [
        {"state_code": "MH", "state_name": "Maharashtra",
         "commercial_zone": "West", "is_reporting": 1},
        {"state_code": "DL", "state_name": "Delhi",
         "commercial_zone": "North", "is_reporting": 1},
    ]
    pd.DataFrame(SKUS).to_csv(data_dir / "sku_master.csv", index=False)
    pd.DataFrame(STATES).to_csv(data_dir / "geo_master.csv", index=False)

    weeks = [f"2023-W{w:02d}" for w in range(1, n_weeks + 1)]
    rng   = np.random.default_rng(seed=99)
    holdout_start = n_weeks - 12

    sales_rows, actuals_rows = [], []
    for sku in SKUS:
        for state in STATES:
            for w_idx, wk in enumerate(weeks):
                qty = max(1, int(200 + 10 * w_idx + rng.normal(0, 10)))
                sales_rows.append({
                    "sku_id": sku["sku_id"], "state_code": state["state_code"],
                    "week_index": wk, "quantity_actual": qty,
                })
                if w_idx >= holdout_start:
                    actuals_rows.append({
                        "sku_id": sku["sku_id"], "state_code": state["state_code"],
                        "week_index": wk, "quantity_actual": qty,
                        "loaded_at": "2023-10-01T00:00:00+00:00",
                    })

    pd.DataFrame(sales_rows).to_csv(data_dir / "sales_history.csv", index=False)
    pd.DataFrame(actuals_rows).to_csv(data_dir / "actuals_holdout.csv", index=False)

    sig_rows = []
    for state in STATES:
        for w_idx, wk in enumerate(weeks):
            sig_rows.append({
                "state_code": state["state_code"], "week_index": wk,
                "temp_deviation": float(w_idx + 1),
                "competitor_price_index": 1.0 + (w_idx + 1) * 0.01,
                "search_trend_index": 50.0 + (w_idx + 1),
            })

    sig_df = pd.DataFrame(sig_rows)
    sig_df[["state_code", "week_index", "temp_deviation"]].to_csv(
        data_dir / "weather_data.csv", index=False)
    sig_df[["state_code", "week_index", "competitor_price_index"]].to_csv(
        data_dir / "competitor_scrapes.csv", index=False)
    sig_df[["state_code", "week_index", "search_trend_index"]].to_csv(
        data_dir / "google_trends_export.csv", index=False)

    return str(data_dir)


def _setup_repo_with_data(tmp_path):
    """Full setup: repo, ingestion run, and signals run."""
    repo     = _make_repo(tmp_path)
    data_dir = _write_fixture_csvs(tmp_path)
    ingestion_module.run(repo, data_dir)
    signals_module.run(repo, data_dir)
    return repo


# ── Guard tests ────────────────────────────────────────────────────────────────

def test_clean_data_assembles_without_error(tmp_path):
    """Baseline: assemble_features succeeds on clean numeric signal data."""
    repo = _setup_repo_with_data(tmp_path)
    df, weeks_list, w2i = assemble_features(repo)

    assert len(df) > 0
    assert len(weeks_list) == 40
    # All assembly-stage feature columns must be numeric (not object)
    assembly_cols = [c for c in FEATURE_COLS if c in df.columns]
    for col in assembly_cols:
        assert df[col].dtype != object, (
            f"Column '{col}' has object dtype after assemble_features — "
            "XGBoost training will fail at .values.astype(float)"
        )


def test_bracketed_string_in_signal_raises_with_temp_lag1_column_name(tmp_path):
    """
    Core regression guard: the exact bracketed string '[1.185077E3]' stored as
    a temp_deviation signal value must raise ValueError naming 'temp_lag1'.

    Without the guard, this propagates all the way to XGBoost training as:
      ValueError: could not convert string to float: '[1.185077E3]'
    with no indication of which column is corrupt.

    The string is stored as TEXT in SQLite (brackets prevent REAL coercion).
    pandas read_frame returns it as object dtype in the temp_deviation column,
    which after _shift_signal_col(lag=1) becomes temp_lag1 in the sales frame.
    """
    repo = _setup_repo_with_data(tmp_path)

    # Overwrite one signal row with the exact bad value from the production crash.
    # SQLite stores it as TEXT because brackets prevent REAL affinity coercion.
    # signal at W10 → after lag-1 shift → temp_lag1 for W11 in the feature matrix.
    repo.upsert("signal_data", [{
        "signal_name":      "temp_deviation",
        "state_code":       "MH",
        "week_index":       "2023-W10",
        "value":            "[1.185077E3]",
        "source_connector": "test_injection",
    }])

    with pytest.raises(ValueError, match="temp_lag1"):
        assemble_features(repo)


def test_bracketed_string_in_comp_signal_raises_with_comp_lag2_column_name(tmp_path):
    """A non-numeric string in competitor_price_index raises naming 'comp_lag2'."""
    repo = _setup_repo_with_data(tmp_path)

    repo.upsert("signal_data", [{
        "signal_name":      "competitor_price_index",
        "state_code":       "DL",
        "week_index":       "2023-W10",
        "value":            "CORRUPT",
        "source_connector": "test_injection",
    }])

    with pytest.raises(ValueError, match="comp_lag2"):
        assemble_features(repo)


def test_bracketed_string_in_search_signal_raises_with_search_lag1_column_name(tmp_path):
    """A bracketed string in search_trend_index raises naming 'search_lag1'."""
    repo = _setup_repo_with_data(tmp_path)

    repo.upsert("signal_data", [{
        "signal_name":      "search_trend_index",
        "state_code":       "MH",
        "week_index":       "2023-W15",
        "value":            "[42.0]",
        "source_connector": "test_injection",
    }])

    with pytest.raises(ValueError, match="search_lag1"):
        assemble_features(repo)


def test_error_message_includes_bad_value_sample(tmp_path):
    """ValueError message must include the offending sample values for diagnosis."""
    repo = _setup_repo_with_data(tmp_path)

    bad_val = "[1.185077E3]"
    repo.upsert("signal_data", [{
        "signal_name":      "temp_deviation",
        "state_code":       "MH",
        "week_index":       "2023-W10",
        "value":            bad_val,
        "source_connector": "test_injection",
    }])

    with pytest.raises(ValueError) as exc_info:
        assemble_features(repo)

    msg = str(exc_info.value)
    assert bad_val in msg, (
        f"Expected bad value '{bad_val}' in error message, got: {msg}"
    )


def test_feature_matrix_is_float_convertible_after_assembly(tmp_path):
    """
    After clean assembly, all present feature columns must be float64-convertible.
    This is the XGBoost training pre-condition — .values.astype(float) must not raise.
    """
    repo = _setup_repo_with_data(tmp_path)
    df, _, _ = assemble_features(repo)

    assembly_cols = [c for c in FEATURE_COLS if c in df.columns]
    clean = df.dropna(subset=assembly_cols)

    # Must not raise — this is the exact call that fails in XGBoost training
    X = clean[assembly_cols].values.astype(float)
    assert X.shape[1] == len(assembly_cols)
    assert X.dtype == float


def test_model_base_score_is_python_float_scalar_and_fit_succeeds(tmp_path):
    """
    base_score passed to XGBRegressor must be a plain Python float scalar.
    Computing it with float(np.mean(y_train)) and passing it explicitly is
    defensive against any internal auto-computation that could produce a
    non-scalar value and cause fit() to fail.
    """
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from pipeline.sensing import HOLDOUT_WEEKS, _XGB_PARAMS

    repo = _setup_repo_with_data(tmp_path)
    df, weeks_list, _ = assemble_features(repo)

    sku_le   = LabelEncoder().fit(sorted(df["sku_id"].unique()))
    state_le = LabelEncoder().fit(sorted(df["state_code"].unique()))
    df = df.copy()
    df["sku_id_enc"]     = sku_le.transform(df["sku_id"])
    df["state_code_enc"] = state_le.transform(df["state_code"])

    holdout_start_ord = len(weeks_list) - HOLDOUT_WEEKS
    df_clean  = df.dropna(subset=FEATURE_COLS).copy()
    train_df  = df_clean[df_clean["week_ord"] < holdout_start_ord].copy()

    tr = train_df[train_df["product_tier"] == "entry"].reset_index(drop=True)
    assert len(tr) >= 20

    y_train    = tr["quantity_actual"].values.astype(float)
    base_score = float(np.mean(y_train))

    # Must be a Python float — numpy scalars and numpy arrays both fail
    # inside XGBoost 2.x's DMatrix construction via the same code path.
    assert type(base_score) is float, (
        f"base_score must be Python float, got {type(base_score).__name__}: "
        f"{base_score!r}"
    )

    # All static _XGB_PARAMS values must also be Python scalars (sanity check)
    for k, v in _XGB_PARAMS.items():
        assert not isinstance(v, np.ndarray), (
            f"_XGB_PARAMS[{k!r}] is a numpy array: {v!r}"
        )

    # model.fit() must not raise — previously crashed with the bracketed-string
    # error when base_score was not explicitly supplied as a Python float.
    X_train = tr[FEATURE_COLS].values.astype(float)
    model = xgb.XGBRegressor(**_XGB_PARAMS, base_score=base_score)
    model.fit(X_train, y_train)


def test_pred_contribs_replaces_shap_tree_explainer(tmp_path):
    """
    Regression guard: XGBoost native pred_contribs=True must replace
    shap.TreeExplainer, which is incompatible with XGBoost 3.x.

    shap.TreeExplainer reads base_score from the booster JSON config.
    XGBoost 3.x serialises it as '[1.185077E3]' (bracketed scientific notation).
    SHAP 0.49.1 calls float() on that raw string and raises ValueError before
    any prediction happens.  pred_contribs=True is XGBoost's own C++ SHAP
    implementation and is always version-compatible.

    pred_contribs returns (n_samples, n_features + 1): the last column is the
    bias/base term.  After slicing [:, :-1] the shape is (n_samples, n_features)
    with per-feature contributions in FEATURE_COLS order.
    """
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from pipeline.sensing import HOLDOUT_WEEKS, _XGB_PARAMS

    repo = _setup_repo_with_data(tmp_path)
    df, weeks_list, _ = assemble_features(repo)

    sku_le   = LabelEncoder().fit(sorted(df["sku_id"].unique()))
    state_le = LabelEncoder().fit(sorted(df["state_code"].unique()))
    df = df.copy()
    df["sku_id_enc"]     = sku_le.transform(df["sku_id"])
    df["state_code_enc"] = state_le.transform(df["state_code"])

    holdout_start_ord = len(weeks_list) - HOLDOUT_WEEKS
    df_clean   = df.dropna(subset=FEATURE_COLS).copy()
    train_df   = df_clean[df_clean["week_ord"] < holdout_start_ord].copy()
    holdout_df = df_clean[df_clean["week_ord"] >= holdout_start_ord].copy()

    tr = train_df[train_df["product_tier"]   == "entry"].reset_index(drop=True)
    ho = holdout_df[holdout_df["product_tier"] == "entry"].reset_index(drop=True)
    assert len(tr) >= 20 and len(ho) > 0

    X_train    = tr[FEATURE_COLS].values.astype(float)
    y_train    = tr["quantity_actual"].values.astype(float)
    X_hold     = ho[FEATURE_COLS].values.astype(float)
    base_score = float(np.mean(y_train))

    model = xgb.XGBRegressor(**_XGB_PARAMS, base_score=base_score)
    model.fit(X_train, y_train)

    # pred_contribs via booster API (sklearn predict() doesn't expose pred_contribs in 3.x)
    contribs = model.get_booster().predict(xgb.DMatrix(X_hold), pred_contribs=True)
    assert contribs.ndim == 2, f"contribs must be 2-D, got shape={contribs.shape}"
    assert contribs.shape[1] == len(FEATURE_COLS) + 1, (
        f"Expected {len(FEATURE_COLS)+1} columns (features + bias), "
        f"got {contribs.shape[1]}"
    )

    shap_mat = contribs[:, :-1]
    assert shap_mat.shape == (len(ho), len(FEATURE_COLS)), (
        f"Expected ({len(ho)}, {len(FEATURE_COLS)}), got {shap_mat.shape}"
    )
    assert np.issubdtype(shap_mat.dtype, np.floating), (
        f"shap_mat dtype not float: {shap_mat.dtype}"
    )

    # Must round-trip through JSON as feature-keyed dict (downstream consumer shape)
    import json
    shap_dict = {FEATURE_COLS[j]: round(float(shap_mat[0, j]), 4)
                 for j in range(len(FEATURE_COLS))}
    parsed = json.loads(json.dumps(shap_dict))
    assert set(parsed.keys()) == set(FEATURE_COLS), (
        f"Contribution keys don't match FEATURE_COLS: {set(parsed.keys())}"
    )


def test_training_path_x_and_y_are_numeric_and_fit_succeeds(tmp_path):
    """
    Integration guard: for the 'entry' tier, both the feature matrix X and the
    target vector y must be 1-D/2-D numeric float arrays, and model.fit(X, y)
    must succeed without raising.

    Regression guard against array-wrapped scalars in y_train producing
    'could not convert string to float: [value]' inside XGBoost's DMatrix
    even when all 10 FEATURE_COLS report clean numeric dtypes.
    """
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from pipeline.sensing import HOLDOUT_WEEKS

    repo = _setup_repo_with_data(tmp_path)
    df, weeks_list, _ = assemble_features(repo)

    # Replicate the label-encoding step from sensing.run()
    sku_le   = LabelEncoder().fit(sorted(df["sku_id"].unique()))
    state_le = LabelEncoder().fit(sorted(df["state_code"].unique()))
    df = df.copy()
    df["sku_id_enc"]     = sku_le.transform(df["sku_id"])
    df["state_code_enc"] = state_le.transform(df["state_code"])

    holdout_start_ord = len(weeks_list) - HOLDOUT_WEEKS
    df_clean  = df.dropna(subset=FEATURE_COLS).copy()
    train_df  = df_clean[df_clean["week_ord"] < holdout_start_ord].copy()

    tr = train_df[train_df["product_tier"] == "entry"].reset_index(drop=True)
    assert len(tr) >= 20, f"Not enough training rows for tier=entry: {len(tr)}"

    X_train = tr[FEATURE_COLS].values.astype(float)
    y_raw   = tr["quantity_actual"].values

    # X must be 2-D float (n_samples × n_features)
    assert X_train.ndim == 2, f"X_train is not 2-D: shape={X_train.shape}"
    assert np.issubdtype(X_train.dtype, np.floating), (
        f"X_train dtype not float: {X_train.dtype}"
    )

    # y must be 1-D and NOT object dtype.
    # object dtype means cells are arrays or strings — the crash condition.
    assert y_raw.ndim == 1, (
        f"y_train is not 1-D: shape={y_raw.shape}. "
        "Likely cause: quantity_actual column cells contain arrays, not scalars."
    )
    assert y_raw.dtype != object, (
        f"y_train has object dtype — cells may be arrays or strings: "
        f"sample={y_raw[:3].tolist()}  "
        f"types={[type(v).__name__ for v in y_raw[:3]]}"
    )

    # XGBoost DMatrix construction and model.fit() must not raise
    y_train = y_raw.astype(float)
    model = xgb.XGBRegressor(n_estimators=5, max_depth=3, verbosity=0,
                              objective="reg:squarederror")
    model.fit(X_train, y_train)  # raises if y contains non-float-convertible values
