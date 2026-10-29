import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
from src.retail_ai.data import default_data_path, load_transactions
from src.retail_ai.analytics import sales_summary, daily_sales

st.set_page_config(page_title="Retail AI Sales Intelligence", page_icon="📈", layout="wide")
st.title("📈 Retail AI Sales Intelligence")
st.caption("Understand sales performance, explore demand, and turn retail data into actionable decisions.")

uploaded = st.sidebar.file_uploader("Upload retail CSV", type=["csv"])
try:
    if uploaded:
        df = load_transactions(uploaded)
        st.sidebar.success(f"Loaded {len(df):,} rows from upload")
    elif default_data_path().exists():
        df = load_transactions(default_data_path())
        st.sidebar.info("Using synthetic demo data")
    else:
        st.warning("Demo data not found. Run `python generate_data.py` from the project folder.")
        st.stop()
except Exception as exc:
    st.error(str(exc))
    st.stop()

min_date, max_date = df["order_date"].min().date(), df["order_date"].max().date()
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if start_date > end_date:
    st.error("Start date must be before end date.")
    st.stop()
filtered = df[df["order_date"].dt.date.between(start_date, end_date)]
summary = sales_summary(filtered)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Revenue", f"${summary['revenue']:,.0f}")
c2.metric("Orders", f"{summary['orders']:,}")
c3.metric("Customers", f"{summary['customers']:,}")
c4.metric("Gross profit", f"${summary['gross_profit']:,.0f}")

left, right = st.columns([1.6, 1])
with left:
    st.subheader("Revenue over time")
    daily = daily_sales(filtered)
    if not daily.empty:
        st.plotly_chart(px.line(daily, x="order_date", y="revenue", markers=True, labels={"order_date": "Date", "revenue": "Revenue ($)"}), use_container_width=True)
    else:
        st.info("No sales in this date range.")
with right:
    st.subheader("Revenue by category")
    category = filtered.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    st.plotly_chart(px.bar(category, x="category", y="revenue", labels={"category": "Category", "revenue": "Revenue ($)"}), use_container_width=True)

st.subheader("Top products")
top_products = (filtered.groupby("product_id", as_index=False).agg(revenue=("revenue", "sum"), units=("quantity", "sum"))
                .sort_values("revenue", ascending=False).head(10))
st.dataframe(top_products, use_container_width=True, hide_index=True)
st.caption("Demo data is synthetic. Forecasts and customer risk scores will be added with validation in subsequent steps.")
