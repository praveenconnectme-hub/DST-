"""
Isolation diagnostic — phase 3: post-fit path (predict + SHAP).

model.fit() passes cleanly for all base_score configs.
Now test: predict, shap.TreeExplainer, shap_values, shap_mean_abs accumulation.

Run: docker cp worker/_diag_xgb.py <worker>:/app/_diag_xgb.py && docker exec <worker> python /app/_diag_xgb.py
"""
import sys, os, traceback
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/api")

import numpy as np
import xgboost as xgb
import shap

print(f"=== XGBoost/SHAP post-fit isolation diagnostic ===")
print(f"xgboost version : {xgb.__version__}")
print(f"shap    version : {shap.__version__}")
print(f"python  version : {sys.version}")
print(f"numpy   version : {np.__version__}")
print()

from repository.factory import RepositoryFactory
from pipeline.sensing import assemble_features, FEATURE_COLS, HOLDOUT_WEEKS
from sklearn.preprocessing import LabelEncoder

repo = RepositoryFactory.create({"type": "sqlite", "db_path": "/data/dst.db"})
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

X_train = tr[FEATURE_COLS].values.astype(float)
y_train = tr["quantity_actual"].values.astype(float)
X_hold  = ho[FEATURE_COLS].values.astype(float)
y_hold  = ho["quantity_actual"].values.astype(float)
base_score = float(np.mean(y_train))

print(f"Train: {X_train.shape}, Holdout: {X_hold.shape}, base_score={base_score}")

_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
               reg_alpha=0.1, reg_lambda=1.0,
               random_state=42, objective="reg:squarederror", verbosity=0)

# ── Step 1: fit ───────────────────────────────────────────────────────────────
print("\n--- Step 1: model.fit(X_train, y_train) ---")
try:
    model = xgb.XGBRegressor(**_PARAMS, base_score=base_score)
    model.fit(X_train, y_train)
    print("    OK")
except Exception:
    print("    CRASH"); traceback.print_exc(file=sys.stdout)

# ── Step 2: predict ───────────────────────────────────────────────────────────
print("\n--- Step 2: model.predict(X_hold) ---")
try:
    y_pred = np.maximum(0.0, model.predict(X_hold))
    print(f"    OK  y_pred[:3]={y_pred[:3]}")
except Exception:
    print("    CRASH"); traceback.print_exc(file=sys.stdout)

# ── Step 3: TreeExplainer construction ───────────────────────────────────────
print("\n--- Step 3: shap.TreeExplainer(model) ---")
try:
    explainer = shap.TreeExplainer(model)
    print(f"    OK  expected_value={explainer.expected_value!r}  "
          f"type={type(explainer.expected_value).__name__}")
except Exception:
    print("    CRASH"); traceback.print_exc(file=sys.stdout)

# ── Step 4: shap_values ───────────────────────────────────────────────────────
print("\n--- Step 4: explainer.shap_values(X_hold) ---")
try:
    shap_mat = explainer.shap_values(X_hold)
    print(f"    OK  type={type(shap_mat).__name__}  ", end="")
    if isinstance(shap_mat, list):
        print(f"list len={len(shap_mat)}  elem shape={np.array(shap_mat[0]).shape}")
    else:
        print(f"shape={np.array(shap_mat).shape}")
except Exception:
    print("    CRASH"); traceback.print_exc(file=sys.stdout)

# ── Step 5: shap_mean_abs accumulation ───────────────────────────────────────
print("\n--- Step 5: shap_mean_abs accumulation ---")
try:
    shap_mean_abs = np.zeros(len(FEATURE_COLS))
    contrib = np.mean(np.abs(shap_mat), axis=0)
    print(f"    contrib type={type(contrib).__name__}  shape={np.array(contrib).shape}")
    shap_mean_abs += contrib
    print(f"    OK  shap_mean_abs={shap_mean_abs}")
except Exception:
    print("    CRASH"); traceback.print_exc(file=sys.stdout)

# ── Step 6: demand_sensing_output row construction ────────────────────────────
print("\n--- Step 6: shap_mat indexing for output rows ---")
try:
    import json
    row = {"sku_id": ho.iloc[0]["sku_id"],
           "state_code": ho.iloc[0]["state_code"],
           "week_index": ho.iloc[0]["week_index"],
           "forecast_qty": round(float(y_pred[0]), 2),
           "model_id": "xgboost_entry_v1",
           "shap_json": json.dumps({FEATURE_COLS[j]: round(float(shap_mat[0, j]), 4)
                                    for j in range(len(FEATURE_COLS))})}
    print(f"    OK  shap_json len={len(row['shap_json'])}")
except Exception:
    print("    CRASH"); traceback.print_exc(file=sys.stdout)

print("\n=== done ===")
