import pandas as pd


def sales_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"revenue": 0.0, "orders": 0, "customers": 0, "gross_profit": 0.0}
    return {
        "revenue": round(float(df["revenue"].sum()), 2),
        "orders": int(df["order_id"].nunique()),
        "customers": int(df["customer_id"].nunique()),
        "gross_profit": round(float(df["gross_profit"].sum()), 2),
    }


def daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["order_date", "revenue"])
    return (df.assign(order_date=df["order_date"].dt.floor("D"))
            .groupby("order_date", as_index=False)["revenue"].sum()
            .sort_values("order_date"))
