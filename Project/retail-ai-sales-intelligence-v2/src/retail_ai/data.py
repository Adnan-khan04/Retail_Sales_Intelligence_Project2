from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {
    "order_id", "order_date", "customer_id", "product_id", "category",
    "quantity", "unit_price", "unit_cost", "region", "inventory_on_hand"
}


def load_transactions(file_or_path) -> pd.DataFrame:
    """Load a CSV and validate the minimum retail transaction schema."""
    df = pd.read_csv(file_or_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in ["quantity", "unit_price", "unit_cost", "inventory_on_hand"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["order_date", "quantity", "unit_price", "unit_cost"])
    df = df[(df["quantity"] >= 0) & (df["unit_price"] >= 0) & (df["unit_cost"] >= 0)]
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["gross_profit"] = df["quantity"] * (df["unit_price"] - df["unit_cost"])
    return df


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "retail_transactions.csv"
