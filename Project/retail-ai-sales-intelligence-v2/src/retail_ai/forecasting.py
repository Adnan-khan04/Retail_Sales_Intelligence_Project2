"""Daily product-level demand forecasting with chronological backtesting.

The model uses lagged demand, rolling statistics, calendar seasonality, and a
same-period-last-year lookup. Forecasts are recursive: predictions are fed back
as lag values for future days. Prediction bands are empirical backtest-error
bands, not guaranteed probabilistic intervals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLUMNS = [
    "product_id_code",
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_364",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_28",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "month",
    "day_of_year", "time_index",
]


@dataclass
class ForecastResult:
    history: pd.DataFrame
    forecast: pd.DataFrame
    metrics: dict[str, float]
    backtest: pd.DataFrame
    feature_importance: pd.DataFrame
    interval_method: str


def aggregate_daily_demand(
    transactions: pd.DataFrame,
    product_id: str,
    date_col: str = "order_date",
    product_col: str = "product_id",
    quantity_col: str = "quantity",
) -> pd.Series:
    """Aggregate transactions to a complete daily series, filling no-sale days with 0."""
    required = {date_col, product_col, quantity_col}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing columns required for forecasting: {sorted(missing)}")

    product_rows = transactions.loc[transactions[product_col].astype(str) == str(product_id)].copy()
    if product_rows.empty:
        raise ValueError(f"No transactions found for product {product_id!r}.")

    product_rows[date_col] = pd.to_datetime(product_rows[date_col], errors="coerce")
    product_rows[quantity_col] = pd.to_numeric(product_rows[quantity_col], errors="coerce")
    product_rows = product_rows.dropna(subset=[date_col, quantity_col])
    if product_rows.empty:
        raise ValueError(f"No valid dated demand rows found for product {product_id!r}.")

    product_rows["day"] = product_rows[date_col].dt.floor("D")
    daily = product_rows.groupby("day")[quantity_col].sum().sort_index()
    # Use the complete dataset's date range so zero-sales days are represented.
    all_dates = pd.date_range(
        pd.to_datetime(transactions[date_col], errors="coerce").min().floor("D"),
        pd.to_datetime(transactions[date_col], errors="coerce").max().floor("D"),
        freq="D",
    )
    daily = daily.reindex(all_dates, fill_value=0.0).astype(float)
    daily.index.name = "date"
    daily.name = "units"
    return daily


def _feature_row(values: np.ndarray, date: pd.Timestamp, product_code: int, time_index: int) -> dict[str, float]:
    """Create a feature row using only values strictly before `date`."""
    def lag(n: int) -> float:
        return float(values[-n]) if len(values) >= n else 0.0

    last_7 = values[-7:] if len(values) >= 7 else values
    last_28 = values[-28:] if len(values) >= 28 else values
    doy = int(date.dayofyear)
    dow = int(date.dayofweek)
    return {
        "product_id_code": float(product_code),
        "lag_1": lag(1),
        "lag_7": lag(7),
        "lag_14": lag(14),
        "lag_28": lag(28),
        "lag_364": lag(364),
        "rolling_mean_7": float(np.mean(last_7)) if len(last_7) else 0.0,
        "rolling_mean_28": float(np.mean(last_28)) if len(last_28) else 0.0,
        "rolling_std_28": float(np.std(last_28, ddof=0)) if len(last_28) else 0.0,
        "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "month": float(date.month),
        "day_of_year": float(doy),
        "time_index": float(time_index),
    }


def _make_training_frame(series: pd.Series, product_code: int) -> pd.DataFrame:
    values = series.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(series.index)
    rows: list[dict[str, Any]] = []
    # Start after 28 days so all short lags/rolling features have history.
    for i in range(28, len(series)):
        features = _feature_row(values[:i], pd.Timestamp(dates[i]), product_code, i)
        features["target"] = float(values[i])
        features["date"] = pd.Timestamp(dates[i])
        rows.append(features)
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["target", "date"])
    return pd.DataFrame(rows)


def _new_model(random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=180,
        min_samples_leaf=2,
        max_features=0.9,
        random_state=random_state,
        n_jobs=-1,
    )


def _recursive_predict(
    model: Any,
    initial_values: np.ndarray,
    dates: pd.DatetimeIndex,
    product_code: int,
    start_index: int,
) -> np.ndarray:
    values = list(np.asarray(initial_values, dtype=float))
    predictions = []
    for offset, date in enumerate(dates):
        row = _feature_row(np.asarray(values, dtype=float), pd.Timestamp(date), product_code, start_index + offset)
        x = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        pred = max(0.0, float(model.predict(x)[0]))
        predictions.append(pred)
        values.append(pred)
    return np.asarray(predictions, dtype=float)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denominator = float(np.sum(np.abs(actual)))
    wape = float(np.sum(np.abs(actual - predicted)) / denominator * 100) if denominator > 0 else float("nan")
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "WAPE_pct": wape, "MAPE_nonzero_pct": mape}


def forecast_product(
    transactions: pd.DataFrame,
    product_id: str,
    horizon: int = 30,
    backtest_days: int = 28,
    interval_level: float = 0.90,
) -> ForecastResult:
    """Fit, chronologically backtest, and recursively forecast one product."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1 day.")
    if backtest_days < 1:
        raise ValueError("backtest_days must be at least 1.")
    if not 0.5 < interval_level < 1:
        raise ValueError("interval_level must be between 0.5 and 1.")

    series = aggregate_daily_demand(transactions, product_id)
    if len(series) < 60:
        raise ValueError("At least 60 calendar days are required; provide more history.")

    product_codes = sorted(transactions["product_id"].astype(str).unique())
    product_code = product_codes.index(str(product_id)) + 1
    holdout = min(backtest_days, max(7, len(series) // 5))
    split = len(series) - holdout
    if split <= 35:
        raise ValueError("Not enough pre-backtest history after feature creation.")

    train_series = series.iloc[:split]
    train_frame = _make_training_frame(train_series, product_code)
    if train_frame.empty:
        raise ValueError("Not enough history to create forecasting features.")

    backtest_model = _new_model()
    backtest_model.fit(train_frame[FEATURE_COLUMNS], train_frame["target"])
    backtest_dates = pd.DatetimeIndex(series.index[split:])
    backtest_predictions = _recursive_predict(
        backtest_model,
        train_series.to_numpy(dtype=float),
        backtest_dates,
        product_code,
        split,
    )
    actual = series.iloc[split:].to_numpy(dtype=float)
    metrics = _metrics(actual, backtest_predictions)
    backtest = pd.DataFrame({
        "date": backtest_dates,
        "actual_units": actual,
        "predicted_units": backtest_predictions,
        "absolute_error": np.abs(actual - backtest_predictions),
    })
    # Finite-sample empirical absolute-error quantile, derived only from held-out dates.
    alpha = 1.0 - interval_level
    errors = np.abs(actual - backtest_predictions)
    q = float(np.quantile(errors, min(1.0, np.ceil((len(errors) + 1) * interval_level) / len(errors)), method="higher"))
    metrics["backtest_days"] = float(len(actual))
    metrics["interval_level_pct"] = float(interval_level * 100)
    metrics["backtest_error_quantile"] = q

    full_frame = _make_training_frame(series, product_code)
    final_model = _new_model()
    final_model.fit(full_frame[FEATURE_COLUMNS], full_frame["target"])
    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_values = _recursive_predict(
        final_model,
        series.to_numpy(dtype=float),
        pd.DatetimeIndex(future_dates),
        product_code,
        len(series),
    )
    forecast = pd.DataFrame({
        "date": future_dates,
        "forecast_units": future_values,
        "lower_units": np.maximum(0.0, future_values - q),
        "upper_units": future_values + q,
    })
    importances = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False, ignore_index=True)

    history = series.rename("actual_units").rename_axis("date").reset_index()
    return ForecastResult(
        history=history,
        forecast=forecast,
        metrics=metrics,
        backtest=backtest,
        feature_importance=importances,
        interval_method=f"Empirical {interval_level:.0%} band using chronological holdout absolute errors",
    )
