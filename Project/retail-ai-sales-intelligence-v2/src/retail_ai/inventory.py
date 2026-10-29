"""Inventory planning based on demand estimates and explicit lead-time assumptions.

Uses recent daily product demand to estimate average demand and variability.
If a forecast is supplied, forecast mean and uncertainty can optionally inform
lead-time demand; this first version uses a transparent normal-approximation
safety-stock calculation from recent daily demand.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class InventoryResult:
    inventory: pd.DataFrame
    assumptions: dict[str, float]


def _daily_product_demand(
    transactions: pd.DataFrame,
    product_col: str = "product_id",
    date_col: str = "order_date",
    quantity_col: str = "quantity",
) -> pd.DataFrame:
    required = {product_col, date_col, quantity_col}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing columns required for inventory planning: {sorted(missing)}")
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")
    df = df.dropna(subset=[product_col, date_col, quantity_col])
    df = df[df[quantity_col] >= 0]
    if df.empty:
        raise ValueError("No valid demand rows are available.")
    start = df[date_col].min().normalize()
    end = df[date_col].max().normalize()
    dates = pd.date_range(start, end, freq="D")
    records = []
    for product, group in df.groupby(product_col, sort=True):
        daily = group.assign(_day=group[date_col].dt.normalize()).groupby("_day")[quantity_col].sum()
        daily = daily.reindex(dates, fill_value=0.0)
        records.append(pd.DataFrame({"product_id": str(product), "date": dates, "demand_units": daily.to_numpy(dtype=float)}))
    return pd.concat(records, ignore_index=True)


def optimize_inventory(
    transactions: pd.DataFrame,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    review_period_days: int = 7,
    demand_window_days: int = 56,
    current_stock_col: str = "inventory_on_hand",
) -> InventoryResult:
    """Calculate reorder points, safety stock, target stock and risk labels.

    Safety stock uses z * std(daily demand) * sqrt(lead time). Reorder point is
    mean daily demand * lead time + safety stock. Order-up-to target includes
    the review period. This assumes independent daily demand and a fixed lead
    time; the dashboard makes those assumptions visible.
    """
    if lead_time_days < 1:
        raise ValueError("Lead time must be at least 1 day.")
    if review_period_days < 0:
        raise ValueError("Review period cannot be negative.")
    if demand_window_days < 14:
        raise ValueError("Demand window must be at least 14 days.")
    if not 0.5 < service_level < 1:
        raise ValueError("Service level must be between 0.5 and 1.")
    if current_stock_col not in transactions.columns:
        raise ValueError(f"Missing current stock column: {current_stock_col}")

    df = transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df[current_stock_col] = pd.to_numeric(df[current_stock_col], errors="coerce")
    df = df.dropna(subset=["order_date", "product_id", "quantity"])
    df = df[df["quantity"] >= 0]
    if df.empty:
        raise ValueError("No valid transactions available.")

    latest_date = df["order_date"].max().normalize()
    daily = _daily_product_demand(df)
    z_scores = {0.80: 0.8416, 0.85: 1.0364, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.96, 0.99: 2.3263}
    z = z_scores.get(round(service_level, 3))
    if z is None:
        # Inverse normal approximation via scipy if installed (it is a scikit-learn dependency).
        from scipy.stats import norm
        z = float(norm.ppf(service_level))

    # Current stock is product-level in this starter dataset; take the last recorded value.
    stock = (
        df.sort_values("order_date")
        .groupby("product_id", as_index=False)
        .agg(current_stock=(current_stock_col, "last"))
    )
    rows = []
    for product, group in daily.groupby("product_id", sort=True):
        recent = group[group["date"] > latest_date - pd.Timedelta(days=demand_window_days)]["demand_units"].astype(float)
        if recent.empty:
            recent = group["demand_units"].astype(float)
        avg_demand = float(recent.mean())
        std_demand = float(recent.std(ddof=1)) if len(recent) > 1 else 0.0
        std_demand = max(0.0, std_demand) if np.isfinite(std_demand) else 0.0
        lead_demand = avg_demand * lead_time_days
        safety_stock = z * std_demand * np.sqrt(lead_time_days)
        reorder_point = lead_demand + safety_stock
        target_stock = avg_demand * (lead_time_days + review_period_days) + z * std_demand * np.sqrt(lead_time_days + review_period_days)
        stock_value = stock.loc[stock["product_id"].astype(str) == str(product), "current_stock"]
        current = float(stock_value.iloc[0]) if not stock_value.empty and pd.notna(stock_value.iloc[0]) else 0.0
        days_cover = current / avg_demand if avg_demand > 0 else float("inf")
        projected_stock_lead = current - lead_demand
        if current <= reorder_point:
            status = "Reorder now"
        elif projected_stock_lead < 0:
            status = "Stockout risk during lead time"
        elif current <= target_stock:
            status = "Watch"
        else:
            status = "Healthy"
        # Approximate stockout risk from normal lead-time demand, conditional on current stock.
        lead_std = std_demand * np.sqrt(lead_time_days)
        if lead_std > 0:
            from scipy.stats import norm
            stockout_probability = float(norm.cdf((lead_demand - current) / lead_std))
        else:
            stockout_probability = 1.0 if current < lead_demand else 0.0
        order_qty = max(0, int(np.ceil(target_stock - current)))
        rows.append({
            "product_id": str(product),
            "current_stock": current,
            "avg_daily_demand": avg_demand,
            "daily_demand_std": std_demand,
            "lead_time_days": lead_time_days,
            "lead_time_demand": lead_demand,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "target_stock": target_stock,
            "days_of_cover": days_cover,
            "projected_stock_at_lead_time": projected_stock_lead,
            "stockout_risk_pct": 100.0 * np.clip(stockout_probability, 0, 1),
            "suggested_order_qty": order_qty,
            "status": status,
        })

    output = pd.DataFrame(rows)
    output = output.sort_values(["stockout_risk_pct", "suggested_order_qty"], ascending=[False, False], ignore_index=True)
    assumptions = {
        "lead_time_days": float(lead_time_days),
        "service_level": float(service_level),
        "review_period_days": float(review_period_days),
        "demand_window_days": float(demand_window_days),
        "z_score": float(z),
        "data_through_epoch_days": float(latest_date.toordinal()),
    }
    return InventoryResult(inventory=output, assumptions=assumptions)
