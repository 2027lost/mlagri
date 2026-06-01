import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


APP_DATA_DIR = Path("app_data")

st.set_page_config(page_title="Pond Forecaster", layout="wide", page_icon="🐟")


@st.cache_data
def load_data():
    with open(APP_DATA_DIR / "meta.json") as handle:
        meta = json.load(handle)
    history = pd.read_csv(APP_DATA_DIR / "history.csv", parse_dates=["Datetime"])
    forecasts = pd.read_csv(
        APP_DATA_DIR / "forecasts.csv", parse_dates=["anchor_dt", "target_dt"]
    )
    backtests = pd.read_csv(APP_DATA_DIR / "backtests.csv", parse_dates=["Datetime"])
    backtest_metrics = pd.read_csv(APP_DATA_DIR / "backtest_metrics.csv")
    feature_importance = pd.read_csv(APP_DATA_DIR / "feature_importance.csv")
    comparison = pd.read_csv("model_comparison_v2.csv")
    return meta, history, forecasts, backtests, backtest_metrics, feature_importance, comparison


meta, history, forecasts, backtests, backtest_metrics, feature_importance, comparison = load_data()

st.sidebar.title("🐟 Pond Forecaster")
st.sidebar.caption("24-hour ahead DO & pH prediction")

pond_choice = st.sidebar.selectbox("Pond", options=meta["pond_names"], index=0)

available_models = forecasts.loc[forecasts["pond"] == pond_choice, "model"].tolist()
model_choice = st.sidebar.radio("Model", available_models)
window_hours = st.sidebar.slider("History window (hours)", 48, 720, 240, step=24)

st.sidebar.markdown("---")
st.sidebar.markdown("**Inputs**: Temp, Turb, historical DO/pH, pond ID")
st.sidebar.markdown("**Targets**: DO(t+24h), pH(t+24h)")
st.sidebar.markdown("**Bands**: P10-P90 quantile heads")
st.sidebar.caption("Deployment uses precomputed model outputs to avoid cloud pickle/joblib issues.")

forecast = forecasts[
    (forecasts["pond"] == pond_choice) & (forecasts["model"] == model_choice)
].iloc[0]

st.title("🐟 Fish Pond 24-h Water Quality Forecast")
st.caption(
    f"Predicting DO and pH 24 hours ahead using saved XGBoost model outputs. Pond: **{pond_choice}** | Model: **{model_choice}**"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Anchor DO (now)", f"{forecast['anchor_do']:.2f} mg/L")
col2.metric(
    "Forecast DO (+24h)",
    f"{forecast['do_pred']:.2f} mg/L",
    delta=f"{forecast['do_pred'] - forecast['anchor_do']:+.2f}",
)
col3.metric("Anchor pH (now)", f"{forecast['anchor_ph']:.2f}")
col4.metric(
    "Forecast pH (+24h)",
    f"{forecast['ph_pred']:.2f}",
    delta=f"{forecast['ph_pred'] - forecast['anchor_ph']:+.2f}",
)

alerts = []
if forecast["do_lo"] < 4.0:
    alerts.append(
        f"DO P10 = {forecast['do_lo']:.2f} mg/L is below safe threshold (4 mg/L) within 24h."
    )
if forecast["ph_lo"] < 6.5 or forecast["ph_hi"] > 8.5:
    alerts.append(
        f"pH band [{forecast['ph_lo']:.2f}, {forecast['ph_hi']:.2f}] is outside safe window (6.5-8.5)."
    )
for alert in alerts:
    st.warning(alert)
if not alerts:
    st.success("Predicted water quality stays within safe operating range.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast", "🔁 Back-test", "📊 Model comparison", "🧠 Feature importance"]
)

with tab1:
    pond_hist = history[history["PondID"] == pond_choice].tail(window_hours)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("DO (mg/L)", "pH"))
    fig.add_trace(
        go.Scatter(
            x=pond_hist["Datetime"],
            y=pond_hist["DO"],
            name="DO observed",
            line=dict(color="#1f77b4", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast["target_dt"]],
            y=[forecast["do_pred"]],
            name="DO P50 forecast",
            mode="markers",
            marker=dict(color="#1f77b4", size=12, symbol="diamond"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast["anchor_dt"], forecast["target_dt"]],
            y=[forecast["anchor_do"], forecast["do_pred"]],
            name="DO trajectory",
            line=dict(color="#1f77b4", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast["target_dt"], forecast["target_dt"]],
            y=[forecast["do_lo"], forecast["do_hi"]],
            name="DO P10-P90",
            line=dict(color="#1f77b4", width=8),
            opacity=0.3,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=4.0,
        line_dash="dot",
        line_color="red",
        row=1,
        col=1,
        annotation_text="safe DO floor (4 mg/L)",
    )
    fig.add_trace(
        go.Scatter(
            x=pond_hist["Datetime"],
            y=pond_hist["pH"],
            name="pH observed",
            line=dict(color="#2ca02c", width=1),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast["target_dt"]],
            y=[forecast["ph_pred"]],
            name="pH P50 forecast",
            mode="markers",
            marker=dict(color="#2ca02c", size=12, symbol="diamond"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast["anchor_dt"], forecast["target_dt"]],
            y=[forecast["anchor_ph"], forecast["ph_pred"]],
            name="pH trajectory",
            line=dict(color="#2ca02c", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast["target_dt"], forecast["target_dt"]],
            y=[forecast["ph_lo"], forecast["ph_hi"]],
            name="pH P10-P90",
            line=dict(color="#2ca02c", width=8),
            opacity=0.3,
        ),
        row=2,
        col=1,
    )
    fig.add_hrect(y0=6.5, y1=8.5, fillcolor="green", opacity=0.05, line_width=0, row=2, col=1)
    fig.update_layout(height=600, hovermode="x unified", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Anchor time: {forecast['anchor_dt']} | Forecast target: {forecast['target_dt']} (+{meta['horizon']}h)"
    )

with tab2:
    selected_backtest = backtests[
        (backtests["pond"] == pond_choice) & (backtests["model"] == model_choice)
    ]
    selected_metrics = backtest_metrics[
        (backtest_metrics["pond"] == pond_choice)
        & (backtest_metrics["model"] == model_choice)
    ].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DO MAE", f"{selected_metrics['DO_MAE']:.3f}")
    col2.metric("DO R²", f"{selected_metrics['DO_R2']:.3f}")
    col3.metric("pH MAE", f"{selected_metrics['pH_MAE']:.3f}")
    col4.metric("pH R²", f"{selected_metrics['pH_R2']:.3f}")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("DO back-test (last 20%)", "pH back-test (last 20%)"),
    )
    fig.add_trace(
        go.Scatter(
            x=selected_backtest["Datetime"],
            y=selected_backtest["DO_true"],
            name="DO actual",
            line=dict(color="black", width=1.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_backtest["Datetime"],
            y=selected_backtest["DO_pred"],
            name="DO pred",
            line=dict(color="#1f77b4", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_backtest["Datetime"],
            y=selected_backtest["DO_hi"],
            name="DO P90",
            line=dict(color="#1f77b4", width=0),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_backtest["Datetime"],
            y=selected_backtest["DO_lo"],
            name="DO P10",
            line=dict(color="#1f77b4", width=0),
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.2)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_backtest["Datetime"],
            y=selected_backtest["pH_true"],
            name="pH actual",
            line=dict(color="black", width=1.2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_backtest["Datetime"],
            y=selected_backtest["pH_pred"],
            name="pH pred",
            line=dict(color="#2ca02c", width=1),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_backtest["Datetime"],
            y=selected_backtest["pH_hi"],
            name="pH P90",
            line=dict(color="#2ca02c", width=0),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=selected_backtest["Datetime"],
            y=selected_backtest["pH_lo"],
            name="pH P10",
            line=dict(color="#2ca02c", width=0),
            fill="tonexty",
            fillcolor="rgba(44,160,44,0.2)",
        ),
        row=2,
        col=1,
    )
    fig.update_layout(height=600, hovermode="x unified", legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, width="stretch")

with tab3:
    st.subheader("Pooled & per-pond comparison")
    st.dataframe(
        comparison.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}", "MAPE": "{:.2f}"}),
        width="stretch",
        height=600,
    )
    st.caption("Lower MAE/RMSE/MAPE is better. R² near 1 is best; negative = worse than mean.")

with tab4:
    selected_importance = feature_importance[feature_importance["model"] == model_choice]
    selected_importance = selected_importance.sort_values("importance", ascending=False).head(20)
    selected_importance = selected_importance.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=selected_importance["importance"],
            y=selected_importance["feature"],
            orientation="h",
            marker=dict(color="#9467bd"),
        )
    )
    fig.update_layout(
        title=f"Top-20 feature importance (gain) - {model_choice}",
        height=600,
        xaxis_title="Gain",
    )
    st.plotly_chart(fig, width="stretch")
