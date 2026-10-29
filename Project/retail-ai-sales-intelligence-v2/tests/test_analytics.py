import pandas as pd
from src.retail_ai.analytics import sales_summary


def test_sales_summary():
    df = pd.DataFrame({
        "order_id": ["o1", "o1", "o2"],
        "customer_id": ["c1", "c1", "c2"],
        "revenue": [10.0, 20.0, 5.0],
        "gross_profit": [3.0, 6.0, 1.0],
    })
    assert sales_summary(df) == {"revenue": 35.0, "orders": 2, "customers": 2, "gross_profit": 10.0}
