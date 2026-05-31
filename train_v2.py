import warnings, glob, os, json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore")
SEED = 42
HORIZON = 24
MODELS_DIR = "models"
MIN_POND_ROWS = 600
BOUNDS = {
    "Temp": (18.0, 38.0),
    "Turb": (0.0, 250.0),
    "DO": (0.5, 14.0),
    "pH": (5.0, 10.0),
}
FEATURE_COLS = [
    "Temp",
    "Turb",
    "HourSin",
    "HourCos",
    "DayOfYearSin",
    "DayOfYearCos",
    "Temp_lag1",
    "Temp_lag3",
    "Temp_lag6",
    "Temp_lag12",
    "Turb_lag1",
    "Turb_lag3",
    "Turb_lag6",
    "Turb_lag12",
    "DO_lag1",
    "DO_lag3",
    "DO_lag6",
    "DO_lag12",
    "DO_lag24",
    "pH_lag1",
    "pH_lag3",
    "pH_lag6",
    "pH_lag12",
    "pH_lag24",
    "Temp_roll6_mean",
    "Temp_roll6_std",
    "DO_roll6_mean",
    "pH_roll6_mean",
    "PondCode",
]
os.makedirs(MODELS_DIR, exist_ok=True)


def find_col(df, kws):
    for kw in kws:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None


def load_pond(path):
    name = os.path.basename(path).replace(".csv", "")
    raw = pd.read_csv(path, low_memory=False)
    cols = {
        "Datetime": find_col(raw, ["created_at"]),
        "Temp": find_col(raw, ["temp"]),
        "Turb": find_col(raw, ["turb"]),
        "DO": find_col(raw, ["dissolved", "oxygen"]),
        "pH": find_col(raw, ["ph"]),
    }
    if not all(cols.values()):
        return (None, name)
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(
                raw[cols["Datetime"]], errors="coerce", utc=True
            ),
            "Temp": pd.to_numeric(raw[cols["Temp"]], errors="coerce"),
            "Turb": pd.to_numeric(raw[cols["Turb"]], errors="coerce"),
            "DO": pd.to_numeric(raw[cols["DO"]], errors="coerce"),
            "pH": pd.to_numeric(raw[cols["pH"]], errors="coerce"),
        }
    ).dropna()
    df["Datetime"] = df["Datetime"].dt.tz_localize(None)
    df = df.sort_values("Datetime").reset_index(drop=True)
    for col, (lo, hi) in BOUNDS.items():
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    if len(df) < 200:
        return (None, name)
    df = df.set_index("Datetime").resample("1h").median().dropna()
    if len(df) < 200 or df["DO"].std() < 0.05 or df["pH"].std() < 0.01:
        return (None, name)
    df["PondID"] = name
    return (df.reset_index(), name)


print("Loading ponds...")
ponds = []
for f in sorted(glob.glob("data2/**/*.csv", recursive=True)):
    p, n = load_pond(f)
    if p is not None:
        ponds.append(p)
        print(f"  {n}: {len(p)} rows")
    else:
        print(f"  {n}: SKIP")


def engineer(df):
    df = df.sort_values("Datetime").reset_index(drop=True)
    h, doy = (df["Datetime"].dt.hour, df["Datetime"].dt.dayofyear)
    df["HourSin"] = np.sin(2 * np.pi * h / 24)
    df["HourCos"] = np.cos(2 * np.pi * h / 24)
    df["DayOfYearSin"] = np.sin(2 * np.pi * doy / 365)
    df["DayOfYearCos"] = np.cos(2 * np.pi * doy / 365)
    for lag in [1, 3, 6, 12]:
        for c in ["Temp", "Turb", "DO", "pH"]:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
    df["DO_lag24"] = df["DO"].shift(24)
    df["pH_lag24"] = df["pH"].shift(24)
    df["Temp_roll6_mean"] = df["Temp"].rolling(6).mean()
    df["Temp_roll6_std"] = df["Temp"].rolling(6).std()
    df["DO_roll6_mean"] = df["DO"].rolling(6).mean()
    df["pH_roll6_mean"] = df["pH"].rolling(6).mean()
    df["DO_future"] = df["DO"].shift(-HORIZON)
    df["pH_future"] = df["pH"].shift(-HORIZON)
    return df.dropna()


engineered = [engineer(p.copy()) for p in ponds]
pond_names = [p["PondID"].iloc[0] for p in engineered]
pond_map = {n: i for (i, n) in enumerate(pond_names)}
for df in engineered:
    df["PondCode"] = df["PondID"].map(pond_map)
combined = (
    pd.concat(engineered, ignore_index=True)
    .sort_values("Datetime")
    .reset_index(drop=True)
)
print(f"\nPooled rows: {len(combined)}  |  Ponds: {len(pond_names)}")


def metrics(y_true, y_pred, label):
    return {
        "Model": label,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": float(
            np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-08))) * 100
        ),
    }


def build_xy(df):
    X = df[FEATURE_COLS].values
    y_do_log = np.log(df["DO_future"].values)
    y_ph = df["pH_future"].values
    return (X, y_do_log, y_ph)


def compute_sample_weights(df, n_bins=20):
    w = np.ones(len(df))
    if "PondCode" in df.columns:
        counts = df["PondCode"].value_counts()
        w *= df["PondCode"].map(lambda c: 1.0 / counts[c]).values
    do = df["DO_future"].values
    hist, edges = np.histogram(do, bins=n_bins)
    idx = np.clip(np.digitize(do, edges[1:-1]), 0, n_bins - 1)
    density = hist[idx].astype(float) + 1.0
    w *= 1.0 / density
    ph = df["pH_future"].values
    hist, edges = np.histogram(ph, bins=n_bins)
    idx = np.clip(np.digitize(ph, edges[1:-1]), 0, n_bins - 1)
    density = hist[idx].astype(float) + 1.0
    w *= 1.0 / density
    w = w / w.mean()
    return w


def train_global(df_train):
    X, y_do_log, y_ph = build_xy(df_train)
    sw = compute_sample_weights(df_train)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    Y = np.column_stack([y_do_log, y_ph])
    joint = xgb.XGBRegressor(
        tree_method="hist",
        multi_strategy="multi_output_tree",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    joint.fit(Xs, Y, sample_weight=sw)
    quantiles = {}
    for tname, ytrain in [("DO_log", y_do_log), ("pH", y_ph)]:
        for q in (0.1, 0.9):
            m = xgb.XGBRegressor(
                tree_method="hist",
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
            )
            m.fit(Xs, ytrain, sample_weight=sw)
            quantiles[f"{tname}_q{int(q * 100)}"] = m
    return {"scaler": sc, "joint": joint, "quantiles": quantiles}


def train_per_pond(df_pond):
    X, y_do_log, y_ph = build_xy(df_pond)
    sw = compute_sample_weights(df_pond)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    Y = np.column_stack([y_do_log, y_ph])
    joint = xgb.XGBRegressor(
        tree_method="hist",
        multi_strategy="multi_output_tree",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    joint.fit(Xs, Y, sample_weight=sw)
    quantiles = {}
    for tname, ytrain in [("DO_log", y_do_log), ("pH", y_ph)]:
        for q in (0.1, 0.9):
            m = xgb.XGBRegressor(
                tree_method="hist",
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
                n_jobs=-1,
            )
            m.fit(Xs, ytrain, sample_weight=sw)
            quantiles[f"{tname}_q{int(q * 100)}"] = m
    return {"scaler": sc, "joint": joint, "quantiles": quantiles}


split = int(len(combined) * 0.8)
train_df = combined.iloc[:split].copy()
test_df = combined.iloc[split:].copy()
print("\nTraining GLOBAL model (XGBoost joint + quantiles, log-DO)...")
global_bundle = train_global(train_df)
print("Training BASELINE sklearn MO-GBR (for comparison)...")
X_tr, y_do_log_tr, y_ph_tr = build_xy(train_df)
X_te, y_do_log_te, y_ph_te = build_xy(test_df)
sc_base = StandardScaler().fit(X_tr)
Xtr_s, Xte_s = (sc_base.transform(X_tr), sc_base.transform(X_te))
baseline = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=SEED,
    ),
    n_jobs=-1,
)
baseline.fit(Xtr_s, np.column_stack([np.exp(y_do_log_tr), y_ph_tr]))
print("\nEvaluating on held-out 20% (pooled, all ponds)...")
results = []
sc_g = global_bundle["scaler"]
Xte_g = sc_g.transform(X_te)
pred_global = global_bundle["joint"].predict(Xte_g)
do_pred_global = np.exp(pred_global[:, 0])
ph_pred_global = pred_global[:, 1]
pred_base = baseline.predict(Xte_s)
do_pred_base = pred_base[:, 0]
ph_pred_base = pred_base[:, 1]
y_do_te = np.exp(y_do_log_te)
results.append(metrics(y_do_te, do_pred_global, "Global XGB-joint(log) → DO"))
results.append(metrics(y_ph_te, ph_pred_global, "Global XGB-joint     → pH"))
results.append(metrics(y_do_te, do_pred_base, "Baseline sklearn MO-GBR → DO"))
results.append(metrics(y_ph_te, ph_pred_base, "Baseline sklearn MO-GBR → pH"))
print("\nTraining PER-POND specialists...")
per_pond = {}
per_pond_use = {}
for name, df_pond in zip(pond_names, engineered):
    sp = int(len(df_pond) * 0.8)
    tr_p = df_pond.iloc[:sp]
    te_p = df_pond.iloc[sp:]
    if len(tr_p) < MIN_POND_ROWS:
        print(f"  {name}: too few rows ({len(tr_p)}) → fall back to global")
        per_pond_use[name] = False
        continue
    bundle = train_per_pond(tr_p)
    per_pond[name] = bundle
    per_pond_use[name] = True
    Xp, ydp_log, yp = build_xy(te_p)
    Xps = bundle["scaler"].transform(Xp)
    pp = bundle["joint"].predict(Xps)
    do_pp = np.exp(pp[:, 0])
    ph_pp = pp[:, 1]
    yd_true = np.exp(ydp_log)
    results.append(metrics(yd_true, do_pp, f"Per-pond {name} → DO"))
    results.append(metrics(yp, ph_pp, f"Per-pond {name} → pH"))
    Xpsg = global_bundle["scaler"].transform(Xp)
    pg = global_bundle["joint"].predict(Xpsg)
    do_g = np.exp(pg[:, 0])
    ph_g = pg[:, 1]
    results.append(metrics(yd_true, do_g, f"Global on {name} → DO"))
    results.append(metrics(yp, ph_g, f"Global on {name} → pH"))
joblib.dump(global_bundle, os.path.join(MODELS_DIR, "global.joblib"))
for name, bundle in per_pond.items():
    joblib.dump(bundle, os.path.join(MODELS_DIR, f"pond_{name}.joblib"))
joblib.dump(
    {"baseline_model": baseline, "baseline_scaler": sc_base},
    os.path.join(MODELS_DIR, "baseline.joblib"),
)
meta = {
    "feature_cols": FEATURE_COLS,
    "horizon": HORIZON,
    "pond_names": pond_names,
    "pond_map": pond_map,
    "per_pond_available": [n for (n, ok) in per_pond_use.items() if ok],
    "bounds": BOUNDS,
}
with open(os.path.join(MODELS_DIR, "meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
combined.to_parquet(os.path.join(MODELS_DIR, "history.parquet"))
results_df = pd.DataFrame(results).round(4)
results_df.to_csv("model_comparison_v2.csv", index=False)
print("\n=== Results ===")
print(results_df.to_string(index=False))
print(f"\nSaved models to {MODELS_DIR}/, results to model_comparison_v2.csv")
