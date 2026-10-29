"""End-to-end retail intelligence pipeline.

Order: validate/load -> sales KPIs -> RFM segmentation -> demand forecast
(backtested per selected product) -> inventory plan -> export artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from src.retail_ai.data import default_data_path, load_transactions
from src.retail_ai.analytics import sales_summary
from src.retail_ai.segmentation import segment_customers
from src.retail_ai.inventory import optimize_inventory
from src.retail_ai.forecasting import forecast_product


def run_pipeline(
    input_path: str | Path | None = None,
    output_dir: str | Path = "artifacts",
    forecast_products: int = 5,
    forecast_horizon: int = 30,
    backtest_days: int = 28,
    n_clusters: int = 4,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> dict[str, Any]:
    """Run the integrated pipeline and write CSV/JSON artifacts."""
    source = Path(input_path) if input_path else default_data_path()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Load and validate.
    df = load_transactions(source)
    if df.empty:
        raise ValueError("Input dataset contains no valid transactions.")

    # 2) Descriptive sales summary.
    summary = sales_summary(df)
    summary.update({
        "rows_loaded": int(len(df)),
        "date_min": str(df["order_date"].min().date()),
        "date_max": str(df["order_date"].max().date()),
        "products": int(df["product_id"].nunique()),
    })
    (out / "sales_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 3) RFM segmentation.
    segmentation = segment_customers(df, n_clusters=n_clusters)
    segmentation.customers.to_csv(out / "customer_segments.csv", index=False)
    segmentation.profile.to_csv(out / "customer_segment_profiles.csv", index=False)

    # 4) Demand forecast: rank products by recent observed units and forecast top N.
    product_demand = (
        df.groupby("product_id", as_index=False)["quantity"].sum()
        .sort_values("quantity", ascending=False)
    )
    selected_products = product_demand["product_id"].astype(str).head(max(1, forecast_products)).tolist()
    forecasts = []
    backtests = []
    forecast_metrics = []
    for product_id in selected_products:
        try:
            result = forecast_product(df, product_id, horizon=forecast_horizon, backtest_days=backtest_days)
            f = result.forecast.copy()
            f.insert(0, "product_id", product_id)
            forecasts.append(f)
            b = result.backtest.copy()
            b.insert(0, "product_id", product_id)
            backtests.append(b)
            forecast_metrics.append({"product_id": product_id, **result.metrics})
        except ValueError as exc:
            forecast_metrics.append({"product_id": product_id, "error": str(exc)})
    forecast_df = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(
        columns=["product_id", "date", "forecast_units", "lower_units", "upper_units"]
    )
    backtest_df = pd.concat(backtests, ignore_index=True) if backtests else pd.DataFrame()
    forecast_df.to_csv(out / "demand_forecasts.csv", index=False)
    backtest_df.to_csv(out / "forecast_backtests.csv", index=False)
    pd.DataFrame(forecast_metrics).to_csv(out / "forecast_metrics.csv", index=False)

    # 5) Inventory planning uses the same validated transactions and explicit assumptions.
    inventory = optimize_inventory(
        df,
        lead_time_days=lead_time_days,
        service_level=service_level,
    )
    inventory.inventory.to_csv(out / "inventory_plan.csv", index=False)

    # 6) Pipeline manifest for traceability.
    manifest = {
        "pipeline": "retail-ai-sales-intelligence",
        "status": "completed",
        "input_path": str(source),
        "output_dir": str(out),
        "transactions": int(len(df)),
        "customers": int(df["customer_id"].nunique()),
        "products": int(df["product_id"].nunique()),
        "forecast_products_requested": int(forecast_products),
        "forecast_products_completed": int(forecast_df["product_id"].nunique()) if not forecast_df.empty else 0,
        "n_customer_clusters": int(n_clusters),
        "inventory_lead_time_days": int(lead_time_days),
        "inventory_service_level": float(service_level),
        "outputs": [
            "sales_summary.json", "customer_segments.csv", "customer_segment_profiles.csv",
            "demand_forecasts.csv", "forecast_backtests.csv", "forecast_metrics.csv", "inventory_plan.csv"
        ],
        "limitations": [
            "Forecasts are product-level models trained independently.",
            "Forecast bands are empirical holdout-error bands, not guaranteed coverage.",
            "Inventory risk is an approximate normal-demand calculation.",
            "RFM segments are descriptive and are not churn probabilities."
        ]
    }
    (out / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"summary": summary, "manifest": manifest, "output_dir": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Retail AI Sales Intelligence pipeline.")
    parser.add_argument("--input", default=None, help="Input retail CSV path; defaults to the synthetic demo CSV.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for generated outputs.")
    parser.add_argument("--forecast-products", type=int, default=5, help="Number of highest-volume products to forecast.")
    parser.add_argument("--forecast-horizon", type=int, default=30)
    parser.add_argument("--backtest-days", type=int, default=28)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--lead-time-days", type=int, default=7)
    parser.add_argument("--service-level", type=float, default=0.95)
    args = parser.parse_args()
    result = run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        forecast_products=args.forecast_products,
        forecast_horizon=args.forecast_horizon,
        backtest_days=args.backtest_days,
        n_clusters=args.clusters,
        lead_time_days=args.lead_time_days,
        service_level=args.service_level,
    )
    print(json.dumps({"status": result["manifest"]["status"], "summary": result["summary"], "output_dir": result["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
