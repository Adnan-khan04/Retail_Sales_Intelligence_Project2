import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.retail_ai.data import default_data_path, load_transactions
from src.retail_ai.segmentation import segment_customers

st.set_page_config(page_title="Customer Intelligence", page_icon="👥", layout="wide")
st.title("👥 Customer Intelligence")
st.caption("RFM segmentation groups customers by purchase recency, order frequency, and observed revenue. Cluster names are interpretable heuristics, not business truths.")

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
    st.subheader("Segmentation settings")
    n_clusters = st.slider("Number of customer groups", min_value=2, max_value=8, value=4)
    run = st.button("Run customer segmentation", type="primary", use_container_width=True)

signature = (len(df), n_clusters, str(df["order_date"].max()), str(df["customer_id"].nunique()))
if "rfm_result" not in st.session_state:
    st.session_state.rfm_result = None
    st.session_state.rfm_signature = None
if run or st.session_state.rfm_result is None or st.session_state.rfm_signature != signature:
    with st.spinner("Building RFM features and clustering customers..."):
        try:
            result = segment_customers(df, n_clusters=n_clusters)
            st.session_state.rfm_result = result
            st.session_state.rfm_signature = signature
        except Exception as exc:
            st.error(f"Segmentation could not be generated: {exc}")
            st.stop()

result = st.session_state.rfm_result
customers = result.customers
profile = result.profile
total_customers = len(customers)
total_revenue = customers["monetary"].sum()
avg_recency = customers["recency_days"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers analyzed", f"{total_customers:,}")
c2.metric("Observed revenue", f"{total_revenue:,.0f}")
c3.metric("Median recency", f"{customers['recency_days'].median():.0f} days")
c4.metric("Silhouette score", f"{result.silhouette:.3f}" if result.silhouette is not None else "N/A")
st.caption(f"RFM reference date: {result.reference_date.date()}. Revenue reflects only the transactions present in the dataset. Silhouette is a rough clustering-separation score, not a measure of business usefulness.")

tab_segments, tab_customers, tab_method = st.tabs(["Segment profiles", "Customer explorer", "Method & actions"])
with tab_segments:
    left, right = st.columns([1.15, 1])
    with left:
        fig = px.bar(profile.sort_values("total_revenue", ascending=True), x="total_revenue", y="segment", orientation="h", text_auto=".2s", title="Observed revenue by segment")
        fig.update_layout(height=390, xaxis_title="Revenue", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.scatter(customers, x="recency_days", y="monetary", color="segment", size="frequency", hover_data=["customer_id", "frequency", "total_units"], title="Customer recency vs. spend")
        fig.update_layout(height=390, xaxis_title="Days since last purchase (lower is better)", yaxis_title="Total observed revenue")
        st.plotly_chart(fig, use_container_width=True)
    display_profile = profile.rename(columns={
        "segment": "Segment", "customers": "Customers", "median_recency_days": "Median recency (days)",
        "median_orders": "Median orders", "median_revenue": "Median customer revenue",
        "total_revenue": "Segment revenue", "avg_customer_age_days": "Avg customer age (days)",
        "median_units": "Median units"
    })
    st.subheader("Interpretable segment profiles")
    st.dataframe(display_profile.round(2), use_container_width=True, hide_index=True)
    st.download_button("Download segment profiles CSV", profile.to_csv(index=False).encode("utf-8"), "customer_segment_profiles.csv", "text/csv")
with tab_customers:
    options = ["All segments"] + sorted(customers["segment"].unique().tolist())
    selected_segment = st.selectbox("Filter segment", options)
    search = st.text_input("Search customer ID", placeholder="Type a customer ID")
    filtered = customers.copy()
    if selected_segment != "All segments":
        filtered = filtered[filtered["segment"] == selected_segment]
    if search.strip():
        filtered = filtered[filtered["customer_id"].astype(str).str.contains(search.strip(), case=False, regex=False)]
    shown = filtered[["customer_id", "segment", "recency_days", "frequency", "monetary", "total_units", "last_purchase", "first_purchase"]].copy()
    shown = shown.rename(columns={
        "customer_id": "Customer ID", "segment": "Segment", "recency_days": "Recency (days)",
        "frequency": "Orders", "monetary": "Observed revenue", "total_units": "Units",
        "last_purchase": "Last purchase", "first_purchase": "First purchase"
    })
    st.write(f"Showing **{len(shown):,}** customers.")
    st.dataframe(shown.round(2), use_container_width=True, hide_index=True)
    st.download_button("Download filtered customers CSV", shown.to_csv(index=False).encode("utf-8"), "customer_rfm_segments.csv", "text/csv")
with tab_method:
    st.markdown("""
### RFM definitions
- **Recency:** calendar days between the reference date and the customer's most recent purchase. Lower is better.
- **Frequency:** number of distinct orders in the observed data. Higher often indicates repeat behavior.
- **Monetary:** total observed sales revenue (`quantity × unit_price`). It is not profit or customer lifetime value.

### How clustering works
1. Aggregate transactions to one row per customer.
2. Log-transform frequency and monetary values to reduce skew.
3. Standardize RFM features and fit K-means.
4. Describe each cluster using medians and transparent relative-rank rules.

### How to act on the groups
- **Champions / Loyal Customers:** consider loyalty rewards and early access.
- **Recent Customers:** encourage a second purchase with relevant onboarding offers.
- **At-Risk High Value:** review recent service issues and consider a targeted win-back campaign.
- **Needs Nurturing / High-Spend Customers:** test tailored recommendations and measure incremental response.

These are suggested campaign ideas, not automatic decisions. Validate segment stability and campaign lift before spending money.
""")
    st.subheader("Feature scale contribution (standardized cluster centers)")
    fig = px.bar(result.feature_importance.sort_values("cluster_center_spread"), x="cluster_center_spread", y="feature", orientation="h")
    fig.update_layout(height=300, xaxis_title="Spread of standardized cluster centers", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.info("This plot is a descriptive measure of how much cluster centers differ on each standardized feature; it is not causal feature importance.")
