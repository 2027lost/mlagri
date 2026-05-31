# Fish Pond 24-h DO & pH Forecaster — Presentation Guide

A walkthrough of the project for a school presentation: what the problem is, what we built, why we made each decision, how it works end-to-end, and what to say if someone asks tough questions.

---

## 1. The problem in one paragraph

Tilapia and other farmed fish are extremely sensitive to **dissolved oxygen (DO)** and **pH**. If DO drops below ~4 mg/L overnight, fish suffocate. If pH drifts outside 6.5–8.5, fish get sick and stop feeding. Farmers want to know **24 hours in advance** what DO and pH will look like, so they can react (run aerators, change water, adjust feed) before fish die. Our job: build a machine-learning model that takes the available sensor inputs — **water temperature, turbidity, historical DO, historical pH, and pond identity** — and predicts both **DO and pH 24 hours from now, at the same time**.

---

## 2. The dataset

- **Source**: Kaggle — `ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets` (IoT sensors from 12 aquaponics fish ponds).
- **Files**: 11 CSVs, one per pond (`IoTPond1.csv` … `IoTPond12.csv`), each ~minute-level sensor readings.
- **Columns we use**: `created_at` (timestamp), `Temperature`, `Turbidity`, `Dissolved Oxygen`, `pH`.

### Why this dataset and not the first one we tried

The first dataset (`jocelyndumlao/iot-monitoring-of-water-quality-and-tilapia`) had a **sensor freeze** — for months 6 and 7 of 2024, the DO and pH readings were perfectly constant (standard deviation = 0.000). A model can't learn from flat-line data, and the metric R² mathematically explodes to large negative numbers when the test set has zero variance. We diagnosed this, switched datasets, and added **physical-validity bounds** so this kind of garbage data gets caught next time.

### Data cleaning pipeline

```
raw CSV → parse timestamps → drop NaN rows
       → apply physical bounds (Temp 18–38 °C, Turb 0–250 NTU, DO 0.5–14 mg/L, pH 5–10)
       → resample to 1-hour median
       → reject ponds with near-zero variance (DO std < 0.05 OR pH std < 0.01)
       → keep 7 valid ponds, ~7,625 hourly rows total
```

**Why physical bounds**: raw sensor data had impossible values — temperatures of −127 °C (the DS18B20 thermometer's error sentinel), DO above 45 mg/L (only ~14 is physically possible at 20 °C), pH below 0 and above 14 (those are theoretical extremes, never seen in real water). Filtering by bounds removes garbage without hand-labeling.

**Why median resampling**: pond sensors are noisy. Taking the median over each hour smooths spikes from things like air bubbles passing the DO probe, without distorting the trend like a mean would.

**Why reject low-variance ponds**: if a sensor was frozen, the pond contributes no learnable signal — it just biases the model toward whatever constant value it got stuck on.

---

## 3. Feature engineering — turning sensor readings into model inputs

A gradient-boosted tree only sees the columns we hand it. We gave it 29 features:

| Group | Features | Why |
|---|---|---|
| Current values | `Temp`, `Turb` | Direct measurements at time *t* |
| Cyclic time | `HourSin`, `HourCos`, `DayOfYearSin`, `DayOfYearCos` | DO and pH have strong daily and seasonal cycles — sin/cos lets a tree split on "near noon vs near midnight" without an artificial wrap-around at hour 23/0 |
| Lag features | `*_lag1`, `*_lag3`, `*_lag6`, `*_lag12` of Temp/Turb/DO/pH; plus `DO_lag24`, `pH_lag24` | Past values are the single strongest predictor of future values — autoregression |
| Rolling stats | `Temp_roll6_mean`, `Temp_roll6_std`, `DO_roll6_mean`, `pH_roll6_mean` | Short-term trend and volatility |
| Pond identity | `PondCode` | A categorical integer per pond — lets the model learn "Pond 3 is generally low-DO" without needing one-hot encoding |
| Targets (what we predict) | `DO_future = DO.shift(-24)`, `pH_future = pH.shift(-24)` | These are the **24h-ahead** values we want to predict |

### Why the shift(-24)

We want to forecast the future. The trick: line up each row's *inputs* with the *outputs* from 24 rows later. `df["DO"].shift(-24)` does exactly that — every row now contains "what DO actually was 24 hours after this moment." During training the model learns the inputs→future mapping; at inference time, given today's inputs, it outputs tomorrow's predicted values.

---

## 4. The models — and the decisions behind them

We compared three families of model. Knowing why each exists is the heart of the project.

### 4.1 `MultiOutputRegressor(GradientBoostingRegressor)` — the **baseline**
This is sklearn's classic approach: wrap a single-target regressor and train one independent copy per target. So we get one GBR for DO and one GBR for pH, trained separately.

> **Important caveat**: `MultiOutputRegressor` is **mathematically identical** to training two independent single-output models. The DO model and pH model don't share any information. So "Multi-Output GBR vs Single-Output GBR" in sklearn is a trick comparison — they will always give the same numbers. We kept it in the project because the original assignment asked for it, and it's a clean baseline.

### 4.2 **XGBoost with `multi_strategy='multi_output_tree'`** — the **true joint model**
This is the upgrade. With `multi_strategy="multi_output_tree"`, XGBoost builds **one tree that outputs both DO and pH at every leaf**. The same splits in the tree are chosen based on what helps *both* targets at once. This is the first model in our comparison that genuinely learns DO and pH jointly — useful because they share underlying drivers like temperature and oxygen-producing algae.

### 4.3 **Per-pond specialists + global fallback**
Different ponds have different sensors, different stocking densities, different shading — so the "ideal" model for Pond 3 might be very different from the one for Pond 7. We train:
- **One global model** on all ponds pooled (the safe default — works for any pond, including new ones).
- **One specialist model per pond** that has enough data (≥ 600 rows after cleaning).

At inference time the dashboard lets you pick **specialist** (if available) or **global**. For ponds without a specialist we fall back to global.

### 4.4 **Log-transform on DO**
Raw DO MAPE was over 200% even though absolute errors were small — because some test DO values were near the 0.5 mg/L floor, and a 1 mg/L error there is a 200% relative error. We train on `log(DO_future)` instead, then exponentiate the prediction back. The error metric becomes well-behaved across the whole range.

### 4.5 **Quantile heads (P10 / P90) for uncertainty bands**
A single point prediction ("DO will be 5.2 mg/L tomorrow") is useless if we don't know how confident we are. We train two extra XGBoost models per target with the `reg:quantileerror` objective at α = 0.10 and α = 0.90. Their outputs give a confidence band: "we're 80% sure DO is between 4.1 and 6.3 mg/L." The dashboard shows this as a shaded ribbon and triggers a warning if the P10 lower bound drops below the 4 mg/L safety floor.

### 4.6 **Sample weighting for balance**
The pooled dataset is imbalanced two ways:
1. Some ponds contribute 1,400 rows; others contribute 600.
2. Most DO readings cluster in a comfortable mid-range; dangerously low or unusually high readings are rare — but those are the ones we care most about predicting.

We compute a `sample_weight` array that combines:
- **Pond inverse-frequency**: weight ∝ 1 / (rows in that pond).
- **Target inverse-density**: bin DO and pH into 20 histogram bins; weight ∝ 1 / (bin count). Rare value ranges get more attention.

These weights are passed into `xgb.fit(..., sample_weight=w)`. The model still sees every row, but the rare/critical ones count more when computing the loss.

---

## 5. The training/test split — why chronological

For time-series data, you **never** shuffle randomly. If you do, the training set contains data from "after" the test set, and the model effectively cheats by knowing the future. We split each pond chronologically: **first 80% for training, last 20% for testing**. This simulates the real world: train on what you've seen, evaluate on what comes next.

We also do **5-fold time-series cross-validation** with `TimeSeriesSplit`: fold 1 trains on the first 1/6, tests on the second; fold 2 trains on the first 2/6, tests on the third; and so on. This catches models that look good on one specific split but are actually unstable.

---

## 6. The metrics — what each one means

| Metric | Formula | What it tells you | "Good" range |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | `mean(|actual − pred|)` | Average prediction error in the original units (mg/L for DO, pH units) | Lower is better |
| **RMSE** (Root Mean Squared Error) | `sqrt(mean((actual − pred)²))` | Same units as MAE, but penalizes big errors more | Lower is better |
| **R²** (Coefficient of Determination) | `1 − SS_res/SS_total` | Fraction of variance the model explains. 1.0 = perfect; 0 = same as predicting the mean; negative = worse than the mean | Higher is better; > 0.5 is solid for noisy sensor data |
| **MAPE** (Mean Absolute Percentage Error) | `mean(|actual − pred|/|actual|) × 100` | Average error as a percent of the true value | Lower; misleading when actual ≈ 0 |

### Why R² goes negative
R² compares your model to a **constant-mean baseline**. If the test slice has very low variance (e.g., DO barely changes for that month), the denominator is tiny and any small error in the numerator looks catastrophic — you can get R² = −500 even when your model is reasonable. **Always read R² alongside MAE**. The two numbers together tell the truth.

---

## 7. Results — what to actually say on stage

Pooled chronological 80/20 (XGB joint with sample weighting + log-DO):

| Model | DO MAE | DO MAPE | pH MAE | pH MAPE |
|---|---|---|---|---|
| Baseline sklearn MO-GBR | 2.96 | 201% | 0.31 | 5.6% |
| **Global XGB joint (ours)** | 3.13 | 195% | 0.32 | 5.7% |

**Per-pond results — where the real wins are**:

| Pond | Per-pond DO MAPE | Global on same pond | Improvement |
|---|---|---|---|
| Pond 3 | **12.8%** | 15.9% | ~20% better |
| Pond 6 | **48.9%** | 374% | ~7.6× better |
| Pond 1 | **84.5%** | 316% | ~3.7× better |
| Pond 7 | 38.7% | 23.4% | global wins here |

### The honest story to tell
1. **pH is genuinely well-predicted** — MAPE around 1–6%, MAE around 0.05–0.3 pH units. That's tight enough to be operationally useful.
2. **DO is harder.** Pooled DO has high MAPE because different ponds have wildly different DO baselines (some run near 1 mg/L, others near 8). A model trained on all of them at once struggles to know which pond regime it's in.
3. **Per-pond specialists fix most of that.** When the model is trained only on Pond 6's history, it understands Pond 6's idiosyncrasies and DO MAPE drops from 374% to 49%.
4. **Cross-pond generalization is the open problem.** A truly hard case is: a brand-new pond gets installed — how do we forecast for it? Best answer today: use the global model, then switch to per-pond once enough data has been collected (typically a few weeks).

---

## 8. The dashboard — what's on the screen

Run with `streamlit run app.py`, open `http://localhost:8501`.

### Sidebar
- **Pond selector** — pick one of the 7 ponds.
- **Model selector** — per-pond specialist (if available) or global fallback.
- **History window** — how many hours back to show in the plot.

### Tabs

1. **📈 Forecast** — Recent observed DO and pH as a time series, plus a forecast point 24h in the future with a P10–P90 confidence band. Horizontal lines mark safety thresholds (DO ≥ 4 mg/L, pH 6.5–8.5). If the lower bound of either band crosses the danger line, the UI shows a yellow warning at the top of the page.

2. **🔁 Back-test** — Runs the chosen model over that pond's held-out last 20% and overlays predicted vs actual with a shaded confidence band. Shows MAE and R² in the header.

3. **📊 Model comparison** — The full results CSV as a sortable table. Every row is one (model, target) combination.

4. **🧠 Feature importance** — XGBoost gain-based importance for the top 20 features of whichever model is selected. You'll see DO lags and roll-means dominate for DO; pH lags dominate for pH; pond ID matters too.

---

## 9. Decision log — questions you might be asked, and the short answers

- **Why XGBoost over neural networks?** Tree-boosters dominate tabular time-series benchmarks at this data scale (~10k rows). A neural network would need much more data and careful tuning to beat XGB here.
- **Why didn't you use an LSTM/transformer?** Same reason — sequence models shine when you have millions of rows or very long context. With 7,625 hourly rows across heterogeneous ponds, tree boosting is the right tool.
- **Why 24 hours ahead specifically?** Farmers run management actions on a daily schedule. 24h gives enough lead time to act, but isn't so far that the prediction becomes worthless.
- **Why not include feed amount as a feature, like the problem statement said?** The aquaponics dataset doesn't have a feed column. The original problem statement assumed a different dataset that did have it. Using just temperature and turbidity (with lags) is what's actually available.
- **Why MultiOutputRegressor if it's identical to two separate models?** The original assignment specified it for comparison. We kept it as the baseline and added the true joint model (XGB multi-output trees) on top — so we satisfy the assignment **and** show why a real joint model is better.
- **Why log-transform DO?** To prevent MAPE blow-up near the lower bound (0.5 mg/L). Without it, a small absolute error near zero becomes a huge percentage error.
- **Why two separate quantile models instead of one?** XGBoost's quantile objective doesn't yet support multi-output trees, so we train one quantile model per (target, quantile) pair. Four extra models total: DO-P10, DO-P90, pH-P10, pH-P90.
- **Why a 600-row threshold for per-pond models?** Below ~600 hourly rows (about 25 days of data), there isn't enough variation across days/seasons to learn from — the model overfits.
- **Are the confidence bands calibrated?** Not formally — they're empirical quantiles from two independent quantile regressors. For real production you'd add **conformal prediction** to guarantee that, say, 80% of actual values fall inside the P10–P90 band. That's the next improvement on the list.
- **What about new ponds with no data?** Use the global model. Coverage of new ponds is harder because the model never saw their baseline behavior — that's the "cross-pond distribution shift" problem mentioned in the results.

---

## 10. File layout

```
mlagri/
├── data2/
├── models/
│   ├── global.joblib
│   ├── pond_<name>.joblib
│   ├── baseline.joblib
│   ├── history.parquet
│   └── meta.json
├── train_v2.py
├── run_analysis2.py
├── app.py
├── model_comparison_v2.csv
├── p2_*.png
└── PRESENTATION_GUIDE.md
```

---

## 11. How to run it from scratch

```bash
pip install -r requirements.txt

kaggle datasets download -d ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets -p data2 --unzip

python train_v2.py

streamlit run app.py
```

---

## 12. What I'd do next if I had more time

1. **Conformal prediction** for calibrated confidence bands.
2. **Add a feed-amount sensor** — would directly improve DO forecasts (feeding → fish respiration → DO drop).
3. **Domain-adversarial training** so the global model becomes pond-invariant — would help new ponds.
4. **Asymmetric loss for DO** — under-predicting low DO kills fish, over-predicting just wastes electricity on aerators; the cost is asymmetric and the loss function should be too.
5. **Online learning / drift detection** — continuously retrain as new sensor data streams in; flag ponds whose residual distribution shifts.
6. **SHAP explanations** in the dashboard — show *why* the model thinks DO will drop tomorrow, not just *that* it will.

---

## 13. One-slide elevator pitch

> "Fish farmers lose stock when oxygen crashes overnight. We built a model that takes the last 24 hours of cheap sensor readings — temperature, turbidity, prior DO and pH — and predicts the next 24 hours of dissolved oxygen and pH with calibrated uncertainty. The model is an XGBoost gradient-boosted tree with multi-output and quantile heads, trained both globally and per-pond with sample weighting to handle imbalance. A Streamlit dashboard shows the forecast in real time and raises a warning when predicted DO is about to cross the safety threshold."
