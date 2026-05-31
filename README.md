# Fish Pond 24-Hour Water Quality Forecasting

This project predicts dissolved oxygen and pH in fish ponds 24 hours ahead from IoT water-quality sensor data. It includes training scripts, saved model artifacts, comparison results, plots, a notebook, and a Streamlit dashboard for pond-level forecasting.

## What The Project Does

The app forecasts:

- `DO_future`: dissolved oxygen 24 hours after the current reading
- `pH_future`: pH 24 hours after the current reading

The dashboard loads trained models from `models/`, lets the user select a pond, shows the latest 24-hour-ahead forecast, displays P10-P90 uncertainty bands, and runs a back-test on the final 20% chronological slice for the selected pond.

## Main Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard for forecast, back-test, comparison table, and feature importance |
| `train_v2.py` | Final training pipeline using XGBoost joint multi-output models and quantile heads |
| `run_analysis2.py` | Pooled multi-pond sklearn baseline analysis with plots and CSV results |
| `run_analysis.py` | Earlier single-dataset sklearn analysis |
| `fish_pond_prediction.ipynb` | Notebook version of the exploratory workflow |
| `models/` | Saved scalers, XGBoost models, per-pond models, metadata, and engineered history |
| `model_comparison_v2.csv` | Final global/per-pond model comparison results |
| `PROJECT_DOCUMENTATION.md` | Full process, flow, algorithms, parameters, and deployment notes |

## Data Flow

```text
raw pond CSV files
-> timestamp parsing and numeric conversion
-> physical sensor validity filtering
-> 1-hour median resampling
-> near-constant sensor rejection
-> time, lag, rolling, and pond-code feature engineering
-> chronological train/test split
-> model training and evaluation
-> saved model artifacts
-> Streamlit dashboard inference
```

## Features Used

The final model uses 29 input features:

- Current values: `Temp`, `Turb`
- Cyclic time: `HourSin`, `HourCos`, `DayOfYearSin`, `DayOfYearCos`
- Lags: `Temp`, `Turb`, `DO`, and `pH` at 1, 3, 6, and 12 hours; `DO` and `pH` also at 24 hours
- Rolling statistics: 6-hour temperature mean/std, 6-hour DO mean, 6-hour pH mean
- Pond identity: `PondCode`

The dataset does not contain feed amount, so feed is not used as a model feature.

## Algorithms

The final pipeline in `train_v2.py` trains:

- A global XGBoost regressor with `multi_strategy="multi_output_tree"` for joint DO and pH prediction
- Per-pond XGBoost specialist models when a pond has at least 600 training rows
- Four quantile regressors for P10 and P90 bounds for DO and pH
- A sklearn `MultiOutputRegressor(GradientBoostingRegressor)` baseline for comparison

DO is trained in log space and converted back with `exp()` during prediction.

## How To Run Locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

To retrain the final model:

```bash
python train_v2.py
```

## Deployment Status

The project is ready for GitHub after running `git init`, committing the files, and pushing to a remote repository.

The app is a Streamlit app. Streamlit Community Cloud is the natural deployment target. Vercel's official Python runtime is designed for serverless HTTP handlers, not persistent Streamlit apps, so direct Vercel deployment would require converting the dashboard to a separate web frontend plus API.

See `DEPLOYMENT.md` for exact GitHub, Streamlit Cloud, and Vercel options.
