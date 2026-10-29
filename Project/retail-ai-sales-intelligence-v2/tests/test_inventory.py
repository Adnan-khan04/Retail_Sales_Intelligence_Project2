import pandas as pd
import pytest
from src.retail_ai.inventory import optimize_inventory


def sample_transactions(days=90):
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for i, date in enumerate(dates):
        rows.append({
            "order_id": f"O{i}",
            "order_date": date,
            "customer_id": f"C{i % 10}",
            "product_id": "P001" if i % 2 else "P002",
            "category": "General",
            "quantity": 5 + (i % 4),
            "unit_price": 20.0,
            "unit_cost": 10.0,
            "region": "North",
            "inventory_on_hand": 20 if i % 2 else 80,
        })
    return pd.DataFrame(rows)


def test_inventory_outputs_and_nonnegative_safety_stock():
    result = optimize_inventory(sample_transactions(), lead_time_days=7, service_level=0.95)
    inv = result.inventory
    assert len(inv) == 2
    assert (inv["safety_stock"] >= 0).all()
    assert (inv["reorder_point"] >= inv["lead_time_demand"]).all()
    assert (inv["suggested_order_qty"] >= 0).all()
    assert inv["stockout_risk_pct"].between(0, 100).all()


def test_higher_service_level_does_not_reduce_safety_stock():
    df = sample_transactions()
    low = optimize_inventory(df, service_level=0.80).inventory.set_index("product_id")
    high = optimize_inventory(df, service_level=0.99).inventory.set_index("product_id")
    assert (high["safety_stock"] >= low["safety_stock"]).all()


def test_invalid_lead_time():
    with pytest.raises(ValueError, match="Lead time"):
        optimize_inventory(sample_transactions(), lead_time_days=0)
