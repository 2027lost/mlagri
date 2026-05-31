import os, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODELS_DIR = "models"
META_PATH = os.path.join(MODELS_DIR, "meta.json")
HIST_PATH = os.path.join(MODELS_DIR, "history.parquet")
st.set_page_config(page_title="Pond Forecaster", layout="wide", page_icon="🐟")


@st.cache_resource
def load_assets():
    with open(META_PATH) as fh:
        meta = json.load(fh)
    history = pd.read_parquet(HIST_PATH)
    global_bundle = joblib.load(os.path.join(MODELS_DIR, "global.joblib"))
    per_pond = {}
    for name in meta["pond_names"]:
        path = os.path.join(MODELS_DIR, f"pond_{name}.joblib")
        if os.path.exists(path):
            per_pond[name] = joblib.load(path)
    results = None
    if os.path.exists("model_comparison_v2.csv"):
        results = pd.read_csv("model_comparison_v2.csv")
    return (meta, history, global_bundle, per_pond, results)


meta, history, global_bundle, per_pond, results = load_assets()
st.sidebar.title("🐟 Pond Forecaster")
st.sidebar.caption("24-hour ahead DO & pH prediction")
pond_choice = st.sidebar.selectbox("Pond", options=meta["pond_names"], index=0)
available_models = ["Global (XGB joint + quantiles, log-DO)"]
if pond_choice in per_pond:
    available_models.insert(0, f"Per-pond specialist ({pond_choice})")
model_choice = st.sidebar.radio("Model", available_models)
use_per_pond = model_choice.startswith("Per-pond")
window_hours = st.sidebar.slider("History window (hours)", 48, 720, 240, step=24)
st.sidebar.markdown("---")
st.sidebar.markdown("**Inputs**: Temp, Turb, historical DO/pH, pond ID")
st.sidebar.markdown("**Targets**: DO(t+24h), pH(t+24h)")
st.sidebar.markdown("**Bands**: P10–P90 quantile heads")


def forecast_for(pond_name, bundle, history):
    pond_hist = history[history["PondCode"] == meta["pond_map"][pond_name]].copy()
    pond_hist = pond_hist.sort_values("Datetime").reset_index(drop=True)
    if pond_hist.empty:
        return None
    last_row = pond_hist.iloc[[-1]]
    X = last_row[meta["feature_cols"]].values
    Xs = bundle["scaler"].transform(X)
    pred = bundle["joint"].predict(Xs)[0]
    do_med = float(np.exp(pred[0]))
    ph_med = float(pred[1])
    qd_lo = float(np.exp(bundle["quantiles"]["DO_log_q10"].predict(Xs)[0]))
    qd_hi = float(np.exp(bundle["quantiles"]["DO_log_q90"].predict(Xs)[0]))
    qp_lo = float(bundle["quantiles"]["pH_q10"].predict(Xs)[0])
    qp_hi = float(bundle["quantiles"]["pH_q90"].predict(Xs)[0])
    return {
        "anchor_dt": pond_hist["Datetime"].iloc[-1],
        "anchor_do": float(pond_hist["DO"].iloc[-1]),
        "anchor_ph": float(pond_hist["pH"].iloc[-1]),
        "do": (qd_lo, do_med, qd_hi),
        "ph": (qp_lo, ph_med, qp_hi),
        "history": pond_hist,
    }


def backtest_pond(pond_name, bundle, history):
    pond_hist = history[history["PondCode"] == meta["pond_map"][pond_name]].copy()
    pond_hist = pond_hist.sort_values("Datetime").reset_index(drop=True)
    split = int(len(pond_hist) * 0.8)
    test = pond_hist.iloc[split:].copy()
    if test.empty:
        return None
    X = test[meta["feature_cols"]].values
    Xs = bundle["scaler"].transform(X)
    pred = bundle["joint"].predict(Xs)
    do_pred = np.exp(pred[:, 0])
    ph_pred = pred[:, 1]
    qd_lo = np.exp(bundle["quantiles"]["DO_log_q10"].predict(Xs))
    qd_hi = np.exp(bundle["quantiles"]["DO_log_q90"].predict(Xs))
    qp_lo = bundle["quantiles"]["pH_q10"].predict(Xs)
    qp_hi = bundle["quantiles"]["pH_q90"].predict(Xs)
    test["DO_pred"] = do_pred
    test["DO_lo"] = qd_lo
    test["DO_hi"] = qd_hi
    test["pH_pred"] = ph_pred
    test["pH_lo"] = qp_lo
    test["pH_hi"] = qp_hi
    test["DO_true"] = test["DO_future"]
    test["pH_true"] = test["pH_future"]
    return test


st.title("🐟 Fish Pond 24-h Water Quality Forecast")
st.caption(
    f"Predicting DO and pH 24 hours ahead using XGBoost joint multi-output trees with quantile bands. Pond: **{pond_choice}** | Model: **{model_choice}**"
)
bundle = (
    per_pond[pond_choice] if use_per_pond and pond_choice in per_pond else global_bundle
)
fc = forecast_for(pond_choice, bundle, history)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Anchor DO (now)", f"{fc['anchor_do']:.2f} mg/L")
col2.metric(
    "Forecast DO (+24h)",
    f"{fc['do'][1]:.2f} mg/L",
    delta=f"{fc['do'][1] - fc['anchor_do']:+.2f}",
)
col3.metric("Anchor pH (now)", f"{fc['anchor_ph']:.2f}")
col4.metric(
    "Forecast pH (+24h)",
    f"{fc['ph'][1]:.2f}",
    delta=f"{fc['ph'][1] - fc['anchor_ph']:+.2f}",
)
alerts = []
if fc["do"][0] < 4.0:
    alerts.append(
        f"⚠️ DO P10 = {fc['do'][0]:.2f} mg/L — below safe threshold (4 mg/L) within 24h."
    )
if fc["ph"][0] < 6.5 or fc["ph"][2] > 8.5:
    alerts.append(
        f"⚠️ pH band [{fc['ph'][0]:.2f}, {fc['ph'][2]:.2f}] outside safe window (6.5–8.5)."
    )
for a in alerts:
    st.warning(a)
if not alerts:
    st.success("✅ Predicted water quality stays within safe operating range.")
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Forecast", "🔁 Back-test", "📊 Model comparison", "🧠 Feature importance"]
)
with tab1:
    hist = fc["history"].tail(window_hours)
    anchor_dt = fc["anchor_dt"]
    target_dt = anchor_dt + pd.Timedelta(hours=meta["horizon"])
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=("DO (mg/L)", "pH")
    )
    fig.add_trace(
        go.Scatter(
            x=hist["Datetime"],
            y=hist["DO"],
            name="DO observed",
            line=dict(color="#1f77b4", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[target_dt],
            y=[fc["do"][1]],
            name="DO P50 forecast",
            mode="markers",
            marker=dict(color="#1f77b4", size=12, symbol="diamond"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[anchor_dt, target_dt],
            y=[fc["anchor_do"], fc["do"][1]],
            name="DO trajectory",
            line=dict(color="#1f77b4", dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[target_dt, target_dt],
            y=[fc["do"][0], fc["do"][2]],
            name="DO P10–P90",
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
            x=hist["Datetime"],
            y=hist["pH"],
            name="pH observed",
            line=dict(color="#2ca02c", width=1),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[target_dt],
            y=[fc["ph"][1]],
            name="pH P50 forecast",
            mode="markers",
            marker=dict(color="#2ca02c", size=12, symbol="diamond"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[anchor_dt, target_dt],
            y=[fc["anchor_ph"], fc["ph"][1]],
            name="pH trajectory",
            line=dict(color="#2ca02c", dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[target_dt, target_dt],
            y=[fc["ph"][0], fc["ph"][2]],
            name="pH P10–P90",
            line=dict(color="#2ca02c", width=8),
            opacity=0.3,
        ),
        row=2,
        col=1,
    )
    fig.add_hrect(
        y0=6.5, y1=8.5, fillcolor="green", opacity=0.05, line_width=0, row=2, col=1
    )
    fig.update_layout(
        height=600,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Anchor time: {anchor_dt} | Forecast target: {target_dt} (+{meta['horizon']}h)"
    )
with tab2:
    bt = backtest_pond(pond_choice, bundle, history)
    if bt is None or bt.empty:
        st.info("Not enough data for back-test.")
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        do_mae = mean_absolute_error(bt["DO_true"], bt["DO_pred"])
        ph_mae = mean_absolute_error(bt["pH_true"], bt["pH_pred"])
        do_r2 = r2_score(bt["DO_true"], bt["DO_pred"])
        ph_r2 = r2_score(bt["pH_true"], bt["pH_pred"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("DO MAE", f"{do_mae:.3f}")
        c2.metric("DO R²", f"{do_r2:.3f}")
        c3.metric("pH MAE", f"{ph_mae:.3f}")
        c4.metric("pH R²", f"{ph_r2:.3f}")
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("DO back-test (last 20%)", "pH back-test (last 20%)"),
        )
        fig.add_trace(
            go.Scatter(
                x=bt["Datetime"],
                y=bt["DO_true"],
                name="DO actual",
                line=dict(color="black", width=1.2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bt["Datetime"],
                y=bt["DO_pred"],
                name="DO pred",
                line=dict(color="#1f77b4", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bt["Datetime"],
                y=bt["DO_hi"],
                name="DO P90",
                line=dict(color="#1f77b4", width=0),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bt["Datetime"],
                y=bt["DO_lo"],
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
                x=bt["Datetime"],
                y=bt["pH_true"],
                name="pH actual",
                line=dict(color="black", width=1.2),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bt["Datetime"],
                y=bt["pH_pred"],
                name="pH pred",
                line=dict(color="#2ca02c", width=1),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bt["Datetime"],
                y=bt["pH_hi"],
                name="pH P90",
                line=dict(color="#2ca02c", width=0),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bt["Datetime"],
                y=bt["pH_lo"],
                name="pH P10",
                line=dict(color="#2ca02c", width=0),
                fill="tonexty",
                fillcolor="rgba(44,160,44,0.2)",
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            height=600, hovermode="x unified", legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
with tab3:
    if results is None:
        st.info("No comparison CSV found.")
    else:
        st.subheader("Pooled & per-pond comparison")
        st.dataframe(
            results.style.format(
                {"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}", "MAPE": "{:.2f}"}
            ),
            use_container_width=True,
            height=600,
        )
        st.caption(
            "Lower MAE/RMSE/MAPE is better. R² near 1 is best; negative = worse than mean."
        )
with tab4:
    booster = bundle["joint"].get_booster()
    score = booster.get_score(importance_type="gain")
    fmap = {f"f{i}": n for (i, n) in enumerate(meta["feature_cols"])}
    items = sorted(
        [(fmap.get(k, k), v) for (k, v) in score.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:20]
    names = [x[0] for x in items][::-1]
    vals = [x[1] for x in items][::-1]
    fig = go.Figure(
        go.Bar(x=vals, y=names, orientation="h", marker=dict(color="#9467bd"))
    )
    fig.update_layout(
        title=f"Top-20 feature importance (gain) — {model_choice}",
        height=600,
        xaxis_title="Gain",
    )
    st.plotly_chart(fig, use_container_width=True)
