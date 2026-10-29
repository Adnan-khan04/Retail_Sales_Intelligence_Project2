from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42


def generate_retail_data(rows: int = 5000, output: str = "data/retail_transactions.csv") -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    end = pd.Timestamp.today().normalize()
    dates = end - pd.to_timedelta(rng.integers(0, 730, size=rows), unit="D")
    customers = [f"C{i:05d}" for i in range(1, 701)]
    products = [f"P{i:03d}" for i in range(1, 61)]
    categories = ["Grocery", "Home", "Electronics", "Fashion", "Beauty"]
    regions = ["North", "South", "East", "West", "Central"]
    product_category = {p: categories[(int(p[1:]) - 1) % len(categories)] for p in products}
    product_price = {p: float(rng.uniform(5, 250)) for p in products}
    product_cost = {p: product_price[p] * float(rng.uniform(0.45, 0.82)) for p in products}
    product_ids = rng.choice(products, size=rows)
    quantities = rng.choice([1, 2, 3, 4, 5], size=rows, p=[.48, .25, .14, .08, .05])
    unit_prices = np.array([product_price[p] for p in product_ids]) * rng.normal(1, .04, size=rows)
    unit_costs = np.array([product_cost[p] for p in product_ids])
    df = pd.DataFrame({
        "order_id": [f"O{i:07d}" for i in range(1, rows + 1)],
        "order_date": dates,
        "customer_id": rng.choice(customers, size=rows),
        "product_id": product_ids,
        "category": [product_category[p] for p in product_ids],
        "quantity": quantities,
        "unit_price": np.round(unit_prices, 2),
        "unit_cost": np.round(unit_costs, 2),
        "region": rng.choice(regions, size=rows),
        "inventory_on_hand": rng.integers(0, 250, size=rows),
    })
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = generate_retail_data()
    print(f"Created {len(data):,} rows at data/retail_transactions.csv")
