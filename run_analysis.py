import warnings, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 130
DATA_PATH = "data/IoT Monitoring Dataset of Water Quality and Tilapia (Oreochromis niloticus) Health in Aquaculture Ponds in Montería, Colombia (2024))/Data_Model_IoTMLCQ_2024.xlsx"
SEED = 42
HORIZON = 24
print("=" * 70)
print("LOADING & CLEANING DATA")
print("=" * 70)
raw = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
raw["Datetime"] = pd.to_datetime(raw["Datetime"])
raw = raw.sort_values("Datetime").reset_index(drop=True)
raw["DO"] = raw["Dissolved Oxygen (mg/L)"]
raw["pH"] = raw["pH"]
raw["Temp"] = raw["Temperature (°C)"]
raw["Turb"] = raw["Turbidity (NTU)"]
raw["Month_num"] = raw["Datetime"].dt.month
raw["hour"] = raw["Datetime"].dt.hour
print("Monthly DO & pH variance diagnostics:")
monthly = raw.groupby("Month_num")[["DO", "pH"]].agg(["mean", "std", "min", "max"])
print(monthly.round(4))
VALID_MONTHS = [1, 2, 3, 4, 5]
df = raw[raw["Month_num"].isin(VALID_MONTHS)].copy().reset_index(drop=True)
print(
    f"\nRows after dropping frozen months (6-7): {len(df)}  ({df['Datetime'].min().date()} → {df['Datetime'].max().date()})"
)
df["HourSin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["HourCos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["DayOfYear"] = df["Datetime"].dt.dayofyear
df["DayOfYearSin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
df["DayOfYearCos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)
for lag in [1, 3, 6, 12]:
    df[f"Temp_lag{lag}"] = df["Temp"].shift(lag)
    df[f"Turb_lag{lag}"] = df["Turb"].shift(lag)
for lag in [1, 3, 6, 12, 24]:
    df[f"DO_lag{lag}"] = df["DO"].shift(lag)
    df[f"pH_lag{lag}"] = df["pH"].shift(lag)
df["Temp_roll6_mean"] = df["Temp"].rolling(6).mean()
df["Temp_roll6_std"] = df["Temp"].rolling(6).std()
df["DO_roll6_mean"] = df["DO"].rolling(6).mean()
df["pH_roll6_mean"] = df["pH"].rolling(6).mean()
df["DO_future"] = df["DO"].shift(-HORIZON)
df["pH_future"] = df["pH"].shift(-HORIZON)
df.dropna(inplace=True)
print(f"Rows after lag/shift engineering: {len(df)}")
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
]
TARGET_COLS = ["DO_future", "pH_future"]
X = df[FEATURE_COLS].values
y = df[TARGET_COLS].values
print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Targets  ({len(TARGET_COLS)}): {TARGET_COLS}")
print("\n" + "=" * 70)
print("EDA")
print("=" * 70)
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
axes = axes.flatten()
cols_plot = ["Temp", "Turb", "DO", "pH"]
labels_plot = ["Temperature (°C)", "Turbidity (NTU)", "DO (mg/L)", "pH"]
for ax, col, lab, c in zip(axes, cols_plot, labels_plot, sns.color_palette("muted", 4)):
    ax.plot(df["Datetime"], df[col], lw=0.6, color=c)
    ax.set_ylabel(lab, fontsize=9)
axes[-1].set_xlabel("Datetime")
fig.suptitle("Sensor Time Series (Jan–May 2024, frozen months removed)", fontsize=12)
plt.tight_layout()
plt.savefig("01_timeseries.png", bbox_inches="tight")
plt.close()
corr_cols = ["Temp", "Turb", "DO", "pH"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Pearson Correlation (Jan–May)")
plt.tight_layout()
plt.savefig("02_correlation.png", bbox_inches="tight")
plt.close()
print("EDA plots saved.")


def compute_metrics(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-08))) * 100
    return {"Model": label, "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE%": mape}


def train_and_eval(X_tr, y_tr, X_te, y_te, gbr_params, tag=""):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    mo = MultiOutputRegressor(GradientBoostingRegressor(**gbr_params), n_jobs=-1)
    mo.fit(Xtr, y_tr)
    pred_mo = mo.predict(Xte)
    gbr_do = GradientBoostingRegressor(**gbr_params)
    gbr_ph = GradientBoostingRegressor(**gbr_params)
    gbr_do.fit(Xtr, y_tr[:, 0])
    gbr_ph.fit(Xtr, y_tr[:, 1])
    pred_so = np.column_stack([gbr_do.predict(Xte), gbr_ph.predict(Xte)])
    rows = [
        compute_metrics(y_te[:, 0], pred_mo[:, 0], f"[{tag}] MO-GBR → DO"),
        compute_metrics(y_te[:, 0], pred_so[:, 0], f"[{tag}] SO-GBR → DO"),
        compute_metrics(y_te[:, 1], pred_mo[:, 1], f"[{tag}] MO-GBR → pH"),
        compute_metrics(y_te[:, 1], pred_so[:, 1], f"[{tag}] SO-GBR → pH"),
    ]
    return (rows, mo, gbr_do, gbr_ph, pred_mo, pred_so, sc)


GBR_CONFIGS = {
    "Default  (n=300, lr=0.05, d=4)": dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=SEED,
    ),
    "Shallow  (n=100, lr=0.10, d=3)": dict(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=SEED,
    ),
    "Deep     (n=500, lr=0.02, d=5)": dict(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        subsample=0.7,
        min_samples_leaf=5,
        random_state=SEED,
    ),
}
SPLITS = {"80/20": 0.8, "70/30": 0.7, "60/40": 0.6}
all_results = []
print("\n" + "=" * 70)
print("SCENARIO RUNS")
print("=" * 70)
for split_name, ratio in SPLITS.items():
    si = int(len(X) * ratio)
    X_tr, X_te = (X[:si], X[si:])
    y_tr, y_te = (y[:si], y[si:])
    for cfg_name, params in GBR_CONFIGS.items():
        tag = f"{split_name} | {cfg_name}"
        print(f"\n  ▶ {tag}")
        rows, *_ = train_and_eval(X_tr, y_tr, X_te, y_te, params, tag)
        all_results.extend(rows)
        for r in rows:
            print(
                f"     {r['Model']:<55}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  R²={r['R²']:.4f}  MAPE={r['MAPE%']:.2f}%"
            )
print("\n" + "=" * 70)
print("TIME-SERIES CV (5-fold, Default config)")
print("=" * 70)
GBR_BEST = dict(
    n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=SEED
)
tscv = TimeSeriesSplit(n_splits=5)
cv_rows = []
for fold, (tri, tei) in enumerate(tscv.split(X), start=1):
    rows, *_ = train_and_eval(X[tri], y[tri], X[tei], y[tei], GBR_BEST, f"Fold-{fold}")
    cv_rows.extend(rows)
    for r in rows:
        print(
            f"  {r['Model']:<45}  MAE={r['MAE']:.4f}  R²={r['R²']:.4f}  MAPE={r['MAPE%']:.2f}%"
        )
cv_df = pd.DataFrame(cv_rows)
cv_df["ModelType"] = cv_df["Model"].str.extract("\\] (.+)")
cv_agg = (
    cv_df.groupby("ModelType")[["MAE", "RMSE", "R²", "MAPE%"]]
    .agg(["mean", "std"])
    .round(4)
)
print("\nCV Aggregated (mean ± std across 5 folds):")
print(cv_agg.to_string())
print("\n" + "=" * 70)
print("FINAL PLOTS — 80/20, Default config")
print("=" * 70)
si = int(len(X) * 0.8)
X_tr, X_te = (X[:si], X[si:])
y_tr, y_te = (y[:si], y[si:])
test_dates = df["Datetime"].values[si:]
rows, mo_f, gbr_do_f, gbr_ph_f, pred_mo_f, pred_so_f, sc_f = train_and_eval(
    X_tr, y_tr, X_te, y_te, GBR_BEST, "Final(80/20)"
)
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for i, (ax, lab) in enumerate(zip(axes, ["DO (mg/L) — 24 h ahead", "pH — 24 h ahead"])):
    ax.plot(test_dates, y_te[:, i], color="black", lw=1.2, label="Actual", zorder=3)
    ax.plot(
        test_dates,
        pred_mo_f[:, i],
        color="steelblue",
        lw=0.9,
        ls="--",
        label="Multi-Output GBR",
        zorder=2,
    )
    ax.plot(
        test_dates,
        pred_so_f[:, i],
        color="tomato",
        lw=0.9,
        ls=":",
        label="Single-Output GBR",
        zorder=2,
    )
    ax.set_ylabel(lab, fontsize=9)
    ax.legend(fontsize=8)
axes[-1].set_xlabel("Datetime")
fig.suptitle("24-Hour Ahead Predictions — Test Set (80/20, Jan–May 2024)", fontsize=12)
plt.tight_layout()
plt.savefig("03_predictions.png", bbox_inches="tight")
plt.close()
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
cases = [
    (axes[0, 0], y_te[:, 0], pred_mo_f[:, 0], "MO-GBR → DO"),
    (axes[0, 1], y_te[:, 0], pred_so_f[:, 0], "SO-GBR → DO"),
    (axes[1, 0], y_te[:, 1], pred_mo_f[:, 1], "MO-GBR → pH"),
    (axes[1, 1], y_te[:, 1], pred_so_f[:, 1], "SO-GBR → pH"),
]
for ax, yt, yp, title in cases:
    ax.scatter(yt, yp, alpha=0.3, s=10, edgecolors="none")
    mn, mx = (min(yt.min(), yp.min()), max(yt.max(), yp.max()))
    ax.plot([mn, mx], [mn, mx], "r--", lw=1)
    r2 = r2_score(yt, yp)
    mae = mean_absolute_error(yt, yp)
    ax.set_title(f"{title}\nR²={r2:.4f}  MAE={mae:.4f}", fontsize=9)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
plt.suptitle("Actual vs Predicted — Test Set", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("04_actual_vs_predicted.png", bbox_inches="tight")
plt.close()
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, est, tname in zip(axes, mo_f.estimators_, ["DO", "pH"]):
    imp = est.feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    ax.barh(
        np.array(FEATURE_COLS)[idx][::-1],
        imp[idx][::-1],
        color=sns.color_palette("muted")[0],
    )
    ax.set_title(f"Feature Importance — MO-GBR → {tname}", fontsize=10)
    ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("05_feature_importance.png", bbox_inches="tight")
plt.close()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, i, tname in zip(axes, [0, 1], ["DO", "pH"]):
    res_mo = y_te[:, i] - pred_mo_f[:, i]
    res_so = y_te[:, i] - pred_so_f[:, i]
    sns.kdeplot(res_mo, ax=ax, label="MO-GBR", fill=True, alpha=0.4)
    sns.kdeplot(res_so, ax=ax, label="SO-GBR", fill=True, alpha=0.4)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_title(f"Residual Distribution — {tname}")
    ax.set_xlabel("Actual − Predicted")
    ax.legend()
plt.tight_layout()
plt.savefig("06_residuals.png", bbox_inches="tight")
plt.close()
print("All plots saved.")
print("\n" + "=" * 70)
print("FULL RESULTS TABLE — All Scenarios")
print("=" * 70)
results_df = pd.DataFrame(all_results).round(4)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 160)
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n─── Best Model per Target (highest R²) ───")
for target in ["DO", "pH"]:
    sub = results_df[results_df["Model"].str.endswith(target)]
    best = sub.loc[sub["R²"].idxmax()]
    print(f"  {target}: {best['Model']}")
    print(
        f"       R²={best['R²']:.4f}  MAE={best['MAE']:.4f}  RMSE={best['RMSE']:.4f}  MAPE={best['MAPE%']:.2f}%"
    )
print("\n─── CV Mean ± Std (5-fold, default config) ───")
for mt in cv_agg.index:
    r2m = cv_agg.loc[mt, ("R²", "mean")]
    r2s = cv_agg.loc[mt, ("R²", "std")]
    maem = cv_agg.loc[mt, ("MAE", "mean")]
    maes = cv_agg.loc[mt, ("MAE", "std")]
    r2_display = (
        f"{r2m:.4f} ± {r2s:.4f}" if abs(r2m) < 1000 else "unstable (near-zero var fold)"
    )
    print(f"  {mt:<25}  MAE={maem:.4f}±{maes:.4f}  R²={r2_display}")
print("\nSaved: model_comparison_results.csv + 6 PNG plots")
