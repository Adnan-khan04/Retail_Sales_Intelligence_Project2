from fastapi import FastAPI, HTTPException, Query
import pandas as pd

from src.retail_ai.data import default_data_path, load_transactions
from src.retail_ai.analytics import sales_summary, daily_sales
from src.retail_ai.forecasting import forecast_product
from src.retail_ai.segmentation import segment_customers
from src.retail_ai.inventory import optimize_inventory

app = FastAPI(
    title="Retail AI Sales Intelligence API",
    version="0.4.0",
    description="Sales KPIs, demand forecasts, RFM segments, and inventory planning."
)


def get_data():
    path = default_data_path()
    if not path.exists():
        raise HTTPException(status_code=503, detail="Dataset not found. Run `python generate_data.py` first.")
    try:
        return load_transactions(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load dataset: {exc}") from exc


@app.get("/health")
def health():
    return {"status": "ok", "service": "retail-ai-sales-intelligence", "version": "0.4.0"}


@app.get("/sales/summary")
def get_sales_summary():
    return sales_summary(get_data())


@app.get("/sales/daily")
def get_daily_sales(limit: int = Query(default=90, ge=1, le=730)):
    df = daily_sales(get_data()).tail(limit)
    return {"data": df.assign(order_date=df["order_date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")}


@app.get("/forecast/demand")
def demand_forecast(
    product_id: str,
    horizon_days: int = Query(default=30, ge=1, le=90),
    backtest_days: int = Query(default=28, ge=7, le=56),
):
    df = get_data()
    if product_id not in set(df["product_id"].astype(str)):
        raise HTTPException(status_code=404, detail=f"Product {product_id!r} was not found.")
    try:
        result = forecast_product(df, product_id, horizon=horizon_days, backtest_days=backtest_days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "product_id": product_id,
        "horizon_days": horizon_days,
        "method": "Random forest with lagged demand, rolling statistics and calendar seasonality",
        "metrics": {k: (float(v) if pd.notna(v) else None) for k, v in result.metrics.items()},
        "forecast": result.forecast.assign(date=result.forecast["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "interval_method": result.interval_method,
    }


@app.get("/customers/segments")
def customer_segments(n_clusters: int = Query(default=4, ge=2, le=8)):
    try:
        result = segment_customers(get_data(), n_clusters=n_clusters)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "n_clusters": result.n_clusters,
        "silhouette": result.silhouette,
        "reference_date": str(result.reference_date.date()),
        "profiles": result.profile.to_dict(orient="records"),
        "customers": result.customers[["customer_id", "recency_days", "frequency", "monetary", "cluster_id", "segment"]].to_dict(orient="records"),
    }


@app.get("/inventory/plan")
def inventory_plan(
    lead_time_days: int = Query(default=7, ge=1, le=60),
    service_level: float = Query(default=0.95, gt=0.5, lt=1.0),
    review_period_days: int = Query(default=7, ge=0, le=60),
    demand_window_days: int = Query(default=56, ge=14, le=180),
):
    try:
        result = optimize_inventory(
            get_data(),
            lead_time_days=lead_time_days,
            service_level=service_level,
            review_period_days=review_period_days,
            demand_window_days=demand_window_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "assumptions": result.assumptions,
        "products": result.inventory.replace({float("inf"): None}).to_dict(orient="records"),
    }
