import pandas as pd
import pytest
from src.retail_ai.forecasting import aggregate_daily_demand, forecast_product


def sample_transactions(days=160):
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    for i, date in enumerate(dates):
        if i % 5 != 0:
            rows.append({"order_date": date, "product_id": "P001", "quantity": float(2 + (i % 7) + (i // 30) % 3)})
        rows.append({"order_date": date, "product_id": "P002", "quantity": 1.0})
    return pd.DataFrame(rows)


def test_aggregate_daily_fills_missing_dates_with_zero():
    df = pd.DataFrame({"order_date": pd.to_datetime(["2025-01-01", "2025-01-03"]), "product_id": ["P001", "P001"], "quantity": [2, 4]})
    daily = aggregate_daily_demand(df, "P001")
    assert len(daily) == 3
    assert daily.iloc[1] == 0
    assert daily.iloc[0] == 2
    assert daily.iloc[2] == 4


def test_forecast_product_shapes_and_nonnegative_predictions():
    result = forecast_product(sample_transactions(), "P001", horizon=14, backtest_days=14)
    assert len(result.forecast) == 14
    assert len(result.backtest) == 14
    assert {"MAE", "RMSE", "WAPE_pct"}.issubset(result.metrics)
    assert (result.forecast["forecast_units"] >= 0).all()
    assert (result.forecast["lower_units"] >= 0).all()
    assert (result.forecast["upper_units"] >= result.forecast["forecast_units"]).all()


def test_short_history_raises_helpful_error():
    with pytest.raises(ValueError, match="60 calendar days"):
        forecast_product(sample_transactions(days=40), "P001")
