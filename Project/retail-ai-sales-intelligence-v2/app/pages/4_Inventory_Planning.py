import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.retail_ai.data import default_data_path, load_transactions
from src.retail_ai.inventory import optimize_inventory

st.set_page_config(page_title="Inventory Planning", page_icon="📦", layout="wide")
st.title("📦 Inventory Optimization")
st.caption("Estimate reorder points, safety stock, suggested replenishment quantities and stockout risk from recent product demand.")

uploaded = st.sidebar.file_uploader("Optional: upload retail CSV", type=["csv"])
if uploaded is not None:
    try:
        df = load_transactions(uploaded)
    except Exception as exc:
        st.error(f"CSV could not be loaded: {exc}")
        st.stop()
else:
    if not default_data_path().exists():
        st.warning("Synthetic data not found. From the project root, run `python generate_data.py`.")
        st.stop()
    df = load_transactions(default_data_path())

with st.sidebar:
    st.subheader("Planning assumptions")
    lead_time = st.slider("Supplier lead time (days)", min_value=1, max_value=30, value=7)
    service_level = st.select_slider("Target service level", options=[0.80, 0.85, 0.90, 0.95, 0.975, 0.99], value=0.95, format_func=lambda x: f"{x:.1%}")
    review_period = st.slider("Review period (days)", min_value=0, max_value=30, value=7)
    demand_window = st.slider("Recent demand window (days)", min_value=14, max_value=180, value=56, step=7)
    run = st.button("Optimize inventory", type="primary", use_container_width=True)

signature = (len(df), str(df["order_date"].max()), lead_time, service_level, review_period, demand_window)
if "inventory_result" not in st.session_state:
    st.session_state.inventory_result = None
    st.session_state.inventory_signature = None
if run or st.session_state.inventory_result is None or st.session_state.inventory_signature != signature:
    with st.spinner("Calculating safety stock, reorder points and stockout risk..."):
        try:
            result = optimize_inventory(
                df,
                lead_time_days=lead_time,
                service_level=service_level,
                review_period_days=review_period,
                demand_window_days=demand_window,
            )
            st.session_state.inventory_result = result
            st.session_state.inventory_signature = signature
        except Exception as exc:
            st.error(f"Inventory analysis could not be generated: {exc}")
            st.stop()

result = st.session_state.inventory_result
inventory = result.inventory
urgent = int(inventory["status"].isin(["Reorder now", "Stockout risk during lead time"]).sum())
risk_count = int((inventory["stockout_risk_pct"] >= 50).sum())
suggested_units = int(inventory["suggested_order_qty"].sum())
a, b, c, d = st.columns(4)
a.metric("Products analyzed", f"{len(inventory):,}")
b.metric("Reorder / lead-time alerts", f"{urgent:,}")
c.metric("Products with ≥50% estimated risk", f"{risk_count:,}")
d.metric("Suggested replenishment", f"{suggested_units:,} units")
st.caption(f"Assumptions: {lead_time}-day lead time, {service_level:.1%} target service level, {review_period}-day review period, last {demand_window} days. Risk is an approximate normal-demand estimate, not a calibrated probability.")

tab_overview, tab_table, tab_guide = st.tabs(["Risk overview", "Product planner", "Method & assumptions"])
with tab_overview:
    left, right = st.columns(2)
    with left:
        status_counts = inventory["status"].value_counts().rename_axis("status").reset_index(name="products")
        fig = px.bar(status_counts, x="status", y="products", color="status", title="Products by stock status", text="products")
        fig.update_layout(height=360, showlegend=False, xaxis_title="", yaxis_title="Products")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(inventory, x="current_stock", y="reorder_point", size="suggested_order_qty", color="status", hover_data=["product_id", "avg_daily_demand", "days_of_cover", "stockout_risk_pct"], title="Current stock vs. reorder point")
        fig.add_shape(type="line", x0=0, y0=0, x1=max(float(inventory["current_stock"].max()), float(inventory["reorder_point"].max()), 1), y1=max(float(inventory["current_stock"].max()), float(inventory["reorder_point"].max()), 1), line=dict(dash="dash"))
        fig.update_layout(height=360, xaxis_title="Current stock (units)", yaxis_title="Reorder point (units)")
        st.plotly_chart(fig, use_container_width=True)
    top_risk = inventory.sort_values("stockout_risk_pct", ascending=False).head(10)
    fig = px.bar(top_risk.sort_values("stockout_risk_pct"), x="stockout_risk_pct", y="product_id", orientation="h", color="status", title="Top 10 products by estimated stockout risk")
    fig.update_layout(height=400, xaxis_title="Estimated risk (%)", yaxis_title="Product")
    st.plotly_chart(fig, use_container_width=True)
with tab_table:
    status_filter = st.multiselect("Filter by stock status", options=sorted(inventory["status"].unique()), default=sorted(inventory["status"].unique()))
    filtered = inventory[inventory["status"].isin(status_filter)].copy()
    min_risk = st.slider("Minimum estimated stockout risk (%)", 0, 100, 0)
    filtered = filtered[filtered["stockout_risk_pct"] >= min_risk]
    st.dataframe(filtered.round(2), use_container_width=True, hide_index=True)
    st.download_button("Download inventory plan CSV", filtered.to_csv(index=False).encode("utf-8"), "inventory_optimization_plan.csv", "text/csv")
with tab_guide:
    st.markdown("""
### Definitions
- **Average daily demand:** mean units sold per calendar day in the selected recent window; days without sales count as zero.
- **Safety stock:** buffer for variability in demand during supplier lead time.
- **Reorder point:** expected demand during lead time plus safety stock.
- **Target stock:** lead-time demand plus demand over the review period, with an uncertainty buffer.
- **Suggested order quantity:** units needed to bring current stock up to the target stock level.
- **Days of cover:** current stock divided by average daily demand.

### Assumptions and limitations
- Daily demand is approximated as independent with a stable mean and standard deviation over the selected window.
- Supplier lead time is treated as fixed; supplier delays, minimum order quantities, pack sizes, backorders and perishability are not modeled yet.
- The dataset's `inventory_on_hand` is recorded on transaction rows, so this demo uses the last recorded value per product as its current stock estimate. In production, stock should come from an authoritative inventory snapshot.
- Estimated stockout risk is a normal-approximation probability that lead-time demand exceeds current stock. It is not calibrated on actual stockout outcomes.
""")
    st.info("The service-level setting controls the safety-stock buffer. A higher target generally means more buffer inventory and fewer expected stockouts, at the cost of holding more stock.")
