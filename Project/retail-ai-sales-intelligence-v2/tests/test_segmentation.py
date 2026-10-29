import pandas as pd
import pytest
from src.retail_ai.segmentation import build_rfm, segment_customers


def sample_transactions():
    rows = []
    dates = pd.date_range("2025-01-01", periods=120, freq="D")
    for c in range(12):
        # Distinct customer behaviors create useful variation.
        for j in range(1 + (c % 5)):
            d = dates[max(0, 115 - j * (c % 4 + 1) - c)]
            rows.append({
                "order_id": f"O{c}_{j}",
                "order_date": d,
                "customer_id": f"C{c:02d}",
                "product_id": "P01",
                "category": "General",
                "quantity": 1 + c,
                "unit_price": 10.0 + c * 5,
                "unit_cost": 5.0,
                "region": "North",
                "inventory_on_hand": 20,
            })
    return pd.DataFrame(rows)


def test_build_rfm_counts_orders_and_revenue():
    df = sample_transactions()
    rfm = build_rfm(df, reference_date="2025-05-01")
    row = rfm.loc[rfm["customer_id"] == "C00"].iloc[0]
    assert row["frequency"] == 1
    assert row["monetary"] == 10.0
    assert row["recency_days"] >= 0


def test_segment_customers_returns_profiles_and_labels():
    result = segment_customers(sample_transactions(), n_clusters=3)
    assert len(result.customers) == 12
    assert result.customers["segment"].notna().all()
    assert result.profile["customers"].sum() == 12
    assert set(result.profile["segment"]).issubset({
        "Champions", "Recent Customers", "At-Risk High Value",
        "Loyal Customers", "High-Spend Customers", "Needs Nurturing"
    })


def test_invalid_cluster_count():
    with pytest.raises(ValueError, match="between 2 and 8"):
        segment_customers(sample_transactions(), n_clusters=1)
