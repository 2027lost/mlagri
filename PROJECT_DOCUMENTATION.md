# Project Documentation

## Objective

The objective is to forecast fish pond water quality 24 hours ahead. The two target variables are dissolved oxygen and pH. These targets matter because low dissolved oxygen and unsafe pH ranges can cause stress, poor feeding, disease, or mortality in fish.

## Dataset Used

The active training pipeline uses the CSV files in `data2/`. Each file represents one pond. The pipeline automatically detects columns for timestamp, temperature, turbidity, dissolved oxygen, and pH by matching partial column names.

The earlier `data/` Excel dataset is only used by `run_analysis.py`. That script is retained as an earlier analysis path, but the final app and saved models come from `train_v2.py`.

## Cleaning Process

The cleaning flow in `train_v2.py` is:

1. Read every CSV file from `data2/`.
2. Auto-detect required columns.
3. Convert timestamp values to timezone-naive `Datetime`.
4. Convert sensor columns to numeric values.
5. Drop missing rows.
6. Sort rows chronologically.
7. Apply physical validity bounds:
   - Temperature: 18.0 to 38.0 degrees C
   - Turbidity: 0.0 to 250.0 NTU
   - Dissolved oxygen: 0.5 to 14.0 mg/L
   - pH: 5.0 to 10.0
8. Resample readings to 1-hour medians.
9. Remove ponds with fewer than 200 valid hourly rows.
10. Remove ponds with near-constant DO or pH sensors.

The final metadata lists these usable ponds:

- `IoTPond2`
- `IoTPond3`
- `IoTPond4`
- `IoTPond6`
- `IoTPond7`
- `IoTPond9`
- `IoTpond1`

## Feature Engineering

For each valid pond, rows are sorted by time and converted into supervised learning rows.

Time features:

- `HourSin`
- `HourCos`
- `DayOfYearSin`
- `DayOfYearCos`

Lag features:

- `Temp_lag1`, `Temp_lag3`, `Temp_lag6`, `Temp_lag12`
- `Turb_lag1`, `Turb_lag3`, `Turb_lag6`, `Turb_lag12`
- `DO_lag1`, `DO_lag3`, `DO_lag6`, `DO_lag12`, `DO_lag24`
- `pH_lag1`, `pH_lag3`, `pH_lag6`, `pH_lag12`, `pH_lag24`

Rolling features:

- `Temp_roll6_mean`
- `Temp_roll6_std`
- `DO_roll6_mean`
- `pH_roll6_mean`

Pond identity:

- `PondCode`

Targets:

- `DO_future = DO.shift(-24)`
- `pH_future = pH.shift(-24)`

Because the data is hourly after resampling, a 24-row shift represents 24 hours.

## Training Flow

The final training flow is:

1. Engineer features per pond.
2. Encode each pond name as an integer `PondCode`.
3. Pool all engineered pond rows.
4. Sort the pooled data chronologically.
5. Split the pooled data into first 80% training and last 20% testing.
6. Train the global XGBoost model.
7. Train the sklearn baseline.
8. Train per-pond specialist models where enough data exists.
9. Evaluate global, baseline, and per-pond models.
10. Save model artifacts, metadata, engineered history, and comparison CSVs.

## Algorithms

### Global XGBoost Joint Model

The main model is `xgboost.XGBRegressor` with:

- `tree_method="hist"`
- `multi_strategy="multi_output_tree"`
- `n_estimators=600`
- `learning_rate=0.05`
- `max_depth=6`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `random_state=42`
- `n_jobs=-1`

This model predicts two outputs together:

- log-transformed DO
- raw pH

The DO prediction is transformed back with `np.exp()`.

### Per-Pond Specialist Models

Each pond with at least 600 training rows gets its own specialist model.

Specialist parameters:

- `tree_method="hist"`
- `multi_strategy="multi_output_tree"`
- `n_estimators=400`
- `learning_rate=0.05`
- `max_depth=5`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `random_state=42`
- `n_jobs=-1`

If a pond has too little data, the app uses the global model.

### Quantile Regressors

The project trains separate XGBoost quantile models to estimate uncertainty bands.

Targets and quantiles:

- `DO_log_q10`
- `DO_log_q90`
- `pH_q10`
- `pH_q90`

Global quantile parameters:

- `objective="reg:quantileerror"`
- `quantile_alpha=0.10` or `0.90`
- `n_estimators=400`
- `learning_rate=0.05`
- `max_depth=5`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `random_state=42`
- `n_jobs=-1`

Per-pond quantile models use `n_estimators=300` and `max_depth=4`.

### Baseline Model

The comparison baseline is:

- `MultiOutputRegressor(GradientBoostingRegressor(...))`
- `n_estimators=300`
- `learning_rate=0.05`
- `max_depth=4`
- `subsample=0.8`
- `random_state=42`
- `n_jobs=-1`

This baseline predicts DO directly and pH directly.

## Sample Weighting

The XGBoost models use sample weights to reduce imbalance.

Weights combine:

- Inverse pond frequency, so ponds with many rows do not dominate.
- Inverse DO target-bin density, so rare DO ranges receive more attention.
- Inverse pH target-bin density, so rare pH ranges receive more attention.

The final weights are normalized to mean 1.

## Evaluation Metrics

The project reports:

- MAE: average absolute error in original units
- RMSE: square-rooted average squared error
- R2: variance explained compared with predicting the mean
- MAPE: average percentage error

R2 can be negative when the test slice has low variance or when the model is worse than a mean baseline. For this dataset, MAE and MAPE are more useful for explaining practical error.

## Final Results Summary

From `model_comparison_v2.csv`:

| Model | Target | MAE | RMSE | R2 | MAPE |
|---|---:|---:|---:|---:|---:|
| Global XGB joint | DO | 3.1345 | 3.9732 | -1.1104 | 195.6175 |
| Global XGB joint | pH | 0.3244 | 0.6508 | 0.1618 | 5.7732 |
| Baseline sklearn MO-GBR | DO | 2.9648 | 4.1686 | -1.3232 | 201.5371 |
| Baseline sklearn MO-GBR | pH | 0.3111 | 0.6397 | 0.1901 | 5.5617 |

Important per-pond outcomes:

- `IoTPond3` DO MAPE improves from 15.9784 with the global model to 12.7806 with the specialist model.
- `IoTPond6` DO MAPE improves from 374.6701 with the global model to 48.9741 with the specialist model.
- `IoTpond1` DO MAPE improves from 316.3149 with the global model to 84.4657 with the specialist model.
- `IoTPond7` performs better with the global model than its specialist model for DO.

The honest interpretation is that pH is more stable and easier to predict, while DO is harder because pond-specific behavior varies strongly.

## Dashboard Flow

At runtime `app.py`:

1. Loads `models/meta.json`.
2. Loads `models/history.parquet`.
3. Loads `models/global.joblib`.
4. Loads available `models/pond_<pond>.joblib` specialists.
5. Lets the user choose a pond and model.
6. Takes the latest engineered row for that pond.
7. Scales features using the saved scaler.
8. Predicts median DO and pH.
9. Predicts P10 and P90 bounds.
10. Shows safety alerts when DO lower bound is below 4.0 mg/L or pH bounds leave 6.5 to 8.5.

## Limitations

- Feed amount is not available in the active dataset.
- Quantile bands are not formally calibrated.
- Pooled DO prediction is difficult because ponds have different baselines.
- New ponds should start with the global model and switch to a specialist model after enough history is collected.
- The Streamlit app is not directly deployable to Vercel without conversion to a Vercel-compatible frontend and API.

## Reproducibility

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Retrain:

```bash
python train_v2.py
```

Run dashboard:

```bash
streamlit run app.py
```
