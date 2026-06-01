import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


MODELS_DIR = "models"
APP_DATA_DIR = Path("app_data")
APP_DATA_DIR.mkdir(exist_ok=True)


def load_bundle(path):
    return joblib.load(path)


def predict_bundle(bundle, rows, feature_cols):
    x_scaled = bundle["scaler"].transform(rows[feature_cols].values)
    pred = bundle["joint"].predict(x_scaled)
    do_pred = np.exp(pred[:, 0])
    ph_pred = pred[:, 1]
    do_lo = np.exp(bundle["quantiles"]["DO_log_q10"].predict(x_scaled))
    do_hi = np.exp(bundle["quantiles"]["DO_log_q90"].predict(x_scaled))
    ph_lo = bundle["quantiles"]["pH_q10"].predict(x_scaled)
    ph_hi = bundle["quantiles"]["pH_q90"].predict(x_scaled)
    return do_lo, do_pred, do_hi, ph_lo, ph_pred, ph_hi


def feature_importance(bundle, feature_cols, model_name):
    booster = bundle["joint"].get_booster()
    score = booster.get_score(importance_type="gain")
    fmap = {f"f{i}": name for i, name in enumerate(feature_cols)}
    rows = []
    for key, value in score.items():
        rows.append(
            {
                "model": model_name,
                "feature": fmap.get(key, key),
                "importance": value,
            }
        )
    return rows


with open(os.path.join(MODELS_DIR, "meta.json")) as handle:
    meta = json.load(handle)

history = pd.read_parquet(os.path.join(MODELS_DIR, "history.parquet"))
history["Datetime"] = pd.to_datetime(history["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
history[["Datetime", "PondID", "PondCode", "Temp", "Turb", "DO", "pH"]].to_csv(
    APP_DATA_DIR / "history.csv", index=False
)

global_bundle = load_bundle(os.path.join(MODELS_DIR, "global.joblib"))
bundles = {"Global": global_bundle}
for pond_name in meta["pond_names"]:
    path = os.path.join(MODELS_DIR, f"pond_{pond_name}.joblib")
    if os.path.exists(path):
        bundles[f"Per-pond specialist ({pond_name})"] = load_bundle(path)

forecast_rows = []
backtest_rows = []
metric_rows = []
importance_rows = []

for model_name, bundle in bundles.items():
    importance_rows.extend(feature_importance(bundle, meta["feature_cols"], model_name))

for pond_name in meta["pond_names"]:
    pond_code = meta["pond_map"][pond_name]
    pond_history = history[history["PondCode"] == pond_code].copy()
    pond_history = pond_history.sort_values("Datetime").reset_index(drop=True)
    latest = pond_history.iloc[[-1]]
    candidates = ["Global"]
    specialist_name = f"Per-pond specialist ({pond_name})"
    if specialist_name in bundles:
        candidates.insert(0, specialist_name)
    for model_name in candidates:
        bundle = bundles[model_name]
        do_lo, do_pred, do_hi, ph_lo, ph_pred, ph_hi = predict_bundle(
            bundle, latest, meta["feature_cols"]
        )
        anchor_dt = pd.to_datetime(latest["Datetime"].iloc[0])
        forecast_rows.append(
            {
                "pond": pond_name,
                "model": model_name,
                "anchor_dt": anchor_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "target_dt": (anchor_dt + pd.Timedelta(hours=meta["horizon"])).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "anchor_do": latest["DO"].iloc[0],
                "anchor_ph": latest["pH"].iloc[0],
                "do_lo": do_lo[0],
                "do_pred": do_pred[0],
                "do_hi": do_hi[0],
                "ph_lo": ph_lo[0],
                "ph_pred": ph_pred[0],
                "ph_hi": ph_hi[0],
            }
        )
        split = int(len(pond_history) * 0.8)
        test = pond_history.iloc[split:].copy()
        do_lo, do_pred, do_hi, ph_lo, ph_pred, ph_hi = predict_bundle(
            bundle, test, meta["feature_cols"]
        )
        test_output = pd.DataFrame(
            {
                "Datetime": test["Datetime"],
                "pond": pond_name,
                "model": model_name,
                "DO_true": test["DO_future"],
                "DO_pred": do_pred,
                "DO_lo": do_lo,
                "DO_hi": do_hi,
                "pH_true": test["pH_future"],
                "pH_pred": ph_pred,
                "pH_lo": ph_lo,
                "pH_hi": ph_hi,
            }
        )
        backtest_rows.append(test_output)
        metric_rows.append(
            {
                "pond": pond_name,
                "model": model_name,
                "DO_MAE": mean_absolute_error(test_output["DO_true"], test_output["DO_pred"]),
                "DO_R2": r2_score(test_output["DO_true"], test_output["DO_pred"]),
                "pH_MAE": mean_absolute_error(test_output["pH_true"], test_output["pH_pred"]),
                "pH_R2": r2_score(test_output["pH_true"], test_output["pH_pred"]),
            }
        )

pd.DataFrame(forecast_rows).to_csv(APP_DATA_DIR / "forecasts.csv", index=False)
pd.concat(backtest_rows, ignore_index=True).to_csv(APP_DATA_DIR / "backtests.csv", index=False)
pd.DataFrame(metric_rows).to_csv(APP_DATA_DIR / "backtest_metrics.csv", index=False)
pd.DataFrame(importance_rows).to_csv(APP_DATA_DIR / "feature_importance.csv", index=False)
with open(APP_DATA_DIR / "meta.json", "w") as handle:
    json.dump(
        {
            "pond_names": meta["pond_names"],
            "pond_map": meta["pond_map"],
            "horizon": meta["horizon"],
            "feature_cols": meta["feature_cols"],
        },
        handle,
        indent=2,
    )
