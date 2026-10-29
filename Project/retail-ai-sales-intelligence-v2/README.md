# Retail AI Sales Intelligence

A starter project for retail sales analytics, demand forecasting, churn risk, customer segmentation, pricing insights, and inventory planning.

## Features in this starter
- Synthetic retail transaction data generator
- CSV upload with basic schema validation
- Streamlit dashboard with sales overview and navigation
- FastAPI endpoints for health, sales summary, and forecast placeholder
- Modular `src/retail_ai` package for analytics/model code

## Requirements
- Python 3.11+ recommended
- Pandas 3.x

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
```

## Run the dashboard
```powershell
streamlit run app/Home.py
```

## Run the API (in another terminal)
```powershell
uvicorn api.main:app --reload
```
Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## CSV schema
Required columns: `order_id`, `order_date`, `customer_id`, `product_id`, `category`, `quantity`, `unit_price`, `unit_cost`, `region`, `inventory_on_hand`.

Each row represents a product line in an order. Revenue is calculated as `quantity * unit_price`.

## Important modeling note
The initial forecast endpoint is deliberately a baseline placeholder. We should add chronological train/test backtesting before presenting forecasts as reliable. Churn scores should likewise be treated as heuristic until trained and validated on labeled repeat-purchase outcomes.


## Demand forecasting module

The Demand Forecasting page trains a product-level random-forest model using daily demand with no-sale days filled with zero; lags (1, 7, 14, 28, and 364 days); rolling statistics; weekly and annual calendar features; chronological holdout backtesting; MAE, RMSE, WAPE, and non-zero-day MAPE; recursive forecasts; and empirical error bands based on holdout errors.

Run `python generate_data.py`, then `streamlit run app/Home.py`. Run tests with `python -m pytest`.

**Important:** Demo data is synthetic. Backtest scores do not establish production accuracy. Empirical bands are not guaranteed statistical coverage. Before real deployment, compare against seasonal-naive baselines and run multiple rolling-origin backtests.


## RFM customer segmentation

The Customer Intelligence page now builds one row per customer with:
- **Recency:** days since the last observed purchase.
- **Frequency:** distinct orders in the dataset.
- **Monetary:** observed revenue (`quantity × unit_price`).

Frequency and monetary values are log-transformed, features are standardized, and K-means groups customers. Segment names (Champions, Recent Customers, Loyal Customers, At-Risk High Value, High-Spend Customers, Needs Nurturing) are transparent heuristic labels based on cluster medians and relative ranks.

Launch from the project root with `streamlit run app/Home.py`. The Customer Intelligence page includes profile charts, a customer explorer, CSV exports, and explanations of suggested actions. Run `python -m pytest` to test both forecasting and segmentation modules.

**Interpretation note:** segments are descriptive and should be checked for stability before being used for marketing. RFM alone does not establish churn probability or causal campaign effects.


## Inventory optimization

The Inventory Planning page estimates product-level:
- Recent average daily demand and standard deviation (including zero-sale days).
- Safety stock and reorder points for a selected supplier lead time and service level.
- Target stock, suggested replenishment quantity and days of cover.
- Approximate stockout risk and prioritized stock status.
- CSV export for purchasing review.

Run `streamlit run app/Home.py` from the project root and open **Inventory Planning**. Run `python -m pytest` for forecasting, segmentation and inventory tests.

**Important assumptions:** demand is treated as approximately independent with stable mean/variance; lead time is fixed. Inventory-on-hand is taken from the last transaction-row value for each product in this demo schema. Production deployments should use a trustworthy stock snapshot, lead-time history, order constraints and validated stockout outcomes.


## Integrated machine-learning pipeline

Run the end-to-end pipeline from the project root:

```powershell
python -m src.retail_ai.pipeline --forecast-products 5 --forecast-horizon 30 --backtest-days 28 --clusters 4 --lead-time-days 7 --service-level 0.95
```

Optional arguments:
- `--input path\to\retail.csv`: use a CSV with the required retail schema.
- `--output-dir artifacts`: choose where generated files are saved.
- `--forecast-products 5`: train forecasts for the five highest-volume products.
- `--forecast-horizon 30`: forecast days ahead.
- `--backtest-days 28`: chronological holdout size.
- `--clusters 4`: number of RFM customer groups.
- `--lead-time-days 7`: supplier lead-time assumption.
- `--service-level 0.95`: target inventory service level.

The pipeline runs in this order: load and validate data, compute sales KPIs, build customer RFM segments, train and backtest demand forecasts for selected products, calculate inventory recommendations, and export a manifest. Outputs are written to `artifacts/` by default:
`sales_summary.json`, `customer_segments.csv`, `customer_segment_profiles.csv`, `demand_forecasts.csv`, `forecast_backtests.csv`, `forecast_metrics.csv`, `inventory_plan.csv`, and `pipeline_manifest.json`.

API endpoints include `/sales/summary`, `/sales/daily`, `/forecast/demand?product_id=P001`, `/customers/segments`, and `/inventory/plan`. Start the API with `uvicorn api.main:app --reload`, then inspect `http://localhost:8000/docs`.

**Scope note:** This pipeline does not yet train a churn classifier because the synthetic schema does not contain a verified churn outcome label. RFM segments are descriptive, not churn probabilities. Price elasticity also requires a suitable price/quantity experiment or enough observed price variation before it should be modeled.
