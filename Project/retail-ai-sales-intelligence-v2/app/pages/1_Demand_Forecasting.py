import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.retail_ai.data import default_data_path, load_transactions
from src.retail_ai.forecasting import forecast_product

st.set_page_config(page_title="Demand Forecasting", page_icon="🔮", layout="wide")
st.title("🔮 Demand Forecasting")
st.caption("Daily product-level forecasts with lagged demand, rolling averages, calendar seasonality, chronological backtesting, and empirical error bands.")

if not default_data_path().exists():
    st.warning("Synthetic data not found. From the project root, run `python generate_data.py`.")
    st.stop()

df = load_transactions(default_data_path())
with st.sidebar:
    st.subheader("Forecast settings")
    products = sorted(df["product_id"].astype(str).unique())
    product = st.selectbox("Product", products)
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1)
    backtest_days = st.slider("Backtest holdout (days)", min_value=7, max_value=56, value=28, step=7)
    interval_level = st.select_slider("Prediction band", options=[0.80, 0.90, 0.95], value=0.90, format_func=lambda x: f"{x:.0%}")
    run = st.button("Train model and forecast", type="primary", use_container_width=True)

if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
    st.session_state.forecast_signature = None

signature = (product, horizon, backtest_days, interval_level)
if run or st.session_state.forecast_signature != signature or st.session_state.forecast_result is None:
    with st.spinner("Training and backtesting the demand model..."):
        try:
            result = forecast_product(df, product, horizon, backtest_days, interval_level)
            st.session_state.forecast_result = result
            st.session_state.forecast_signature = signature
        except Exception as exc:
            st.error(f"Forecast could not be generated: {exc}")
            st.stop()

result = st.session_state.forecast_result
m1, m2, m3, m4 = st.columns(4)
m1.metric("Backtest MAE", f"{result.metrics['MAE']:.2f} units/day")
m2.metric("Backtest RMSE", f"{result.metrics['RMSE']:.2f} units/day")
wape = result.metrics["WAPE_pct"]
m3.metric("Backtest WAPE", f"{wape:.1f}%" if pd.notna(wape) else "N/A")
m4.metric("Forecast total", f"{result.forecast['forecast_units'].sum():,.0f} units")
st.caption(f"Evaluation: last {int(result.metrics['backtest_days'])} days held out in time order. Interval method: {result.interval_method}. These bands are empirical, not guaranteed coverage.")

tab_forecast, tab_backtest, tab_features = st.tabs(["Forecast", "Backtest", "Model drivers"])
with tab_forecast:
    fig = go.Figure()
    hist = result.history.tail(120)
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["actual_units"], name="Historical demand", mode="lines", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=result.forecast["date"], y=result.forecast["upper_units"], name="Upper band", mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=result.forecast["date"], y=result.forecast["lower_units"], name=f"{interval_level:.0%} empirical band", mode="lines", line=dict(width=0), fill="tonexty", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=result.forecast["date"], y=result.forecast["forecast_units"], name="Forecast", mode="lines+markers", line=dict(width=3, dash="dash")))
    fig.update_layout(height=460, xaxis_title="Date", yaxis_title="Units sold", hovermode="x unified", legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(result.forecast.round(2), use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", data=result.forecast.to_csv(index=False).encode("utf-8"), file_name=f"{product}_demand_forecast.csv", mime="text/csv")
with tab_backtest:
    st.markdown("The model trains on earlier dates and predicts the later holdout recursively. No holdout actual values are used as lag inputs.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.backtest["date"], y=result.backtest["actual_units"], name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=result.backtest["date"], y=result.backtest["predicted_units"], name="Predicted", mode="lines+markers"))
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Units sold", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(result.backtest.round(2), use_container_width=True, hide_index=True)
    st.markdown("**Metric guide**")
    st.write("- **MAE:** average absolute error, in units per day.")
    st.write("- **RMSE:** penalizes larger misses more strongly, in units per day.")
    st.write("- **WAPE:** total absolute error divided by total actual demand; useful when some days have zero demand.")
    st.write("- **MAPE (nonzero days):** excludes zero-demand days and should be interpreted cautiously.")
with tab_features:
    st.markdown("Feature importance indicates which inputs the fitted random forest relied on most. It describes model behavior, not causation.")
    imp = result.feature_importance.head(12).sort_values("importance")
    fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h"))
    fig.update_layout(height=420, xaxis_title="Feature importance", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(result.feature_importance.round(4), use_container_width=True, hide_index=True)
