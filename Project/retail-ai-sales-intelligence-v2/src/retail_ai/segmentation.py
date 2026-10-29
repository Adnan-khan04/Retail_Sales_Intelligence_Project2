"""RFM customer segmentation with K-means and interpretable segment profiles."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class SegmentationResult:
    customers: pd.DataFrame
    profile: pd.DataFrame
    feature_importance: pd.DataFrame
    n_clusters: int
    silhouette: float | None
    reference_date: pd.Timestamp


def build_rfm(transactions: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Build Recency, Frequency, Monetary features at customer level."""
    required = {"order_date", "order_id", "customer_id", "quantity", "unit_price"}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing columns required for RFM: {sorted(missing)}")

    df = transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df = df.dropna(subset=["order_date", "customer_id", "order_id", "quantity", "unit_price"])
    df = df[(df["quantity"] >= 0) & (df["unit_price"] >= 0)]
    if df.empty:
        raise ValueError("No valid transactions available to build RFM features.")

    if reference_date is None:
        reference_date = df["order_date"].max().normalize() + pd.Timedelta(days=1)
    else:
        reference_date = pd.Timestamp(reference_date).normalize()

    df["revenue"] = df["quantity"] * df["unit_price"]
    rfm = df.groupby("customer_id", as_index=False).agg(
        last_purchase=("order_date", "max"),
        frequency=("order_id", "nunique"),
        monetary=("revenue", "sum"),
        total_units=("quantity", "sum"),
        first_purchase=("order_date", "min"),
    )
    rfm["recency_days"] = (reference_date - rfm["last_purchase"].dt.normalize()).dt.days.clip(lower=0)
    rfm["customer_age_days"] = (reference_date - rfm["first_purchase"].dt.normalize()).dt.days.clip(lower=0)
    # Frequency counts distinct orders; monetary is total observed revenue.
    return rfm.sort_values("customer_id", ignore_index=True)


def _safe_log1p(values: pd.Series) -> pd.Series:
    return np.log1p(values.clip(lower=0))


def segment_customers(
    transactions: pd.DataFrame,
    n_clusters: int = 4,
    reference_date=None,
    random_state: int = 42,
) -> SegmentationResult:
    """Cluster customers on transformed RFM features and label clusters by behavior."""
    if n_clusters < 2 or n_clusters > 8:
        raise ValueError("Choose between 2 and 8 clusters.")
    rfm = build_rfm(transactions, reference_date=reference_date)
    if len(rfm) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} unique customers, found {len(rfm)}.")

    # Log transform the heavily skewed frequency and monetary distributions.
    feature_cols = ["recency_days", "frequency", "monetary"]
    X = rfm[feature_cols].copy()
    X["frequency"] = _safe_log1p(X["frequency"])
    X["monetary"] = _safe_log1p(X["monetary"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    rfm["cluster_id"] = labels
    segment_names = _label_segments(rfm.groupby("cluster_id").agg(
        customers=("customer_id", "count"),
        recency_days=("recency_days", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
    ).reset_index())
    rfm["segment"] = rfm["cluster_id"].map(segment_names)
    # Use medians to make profiles robust to outliers.
    profile = rfm.groupby(["cluster_id", "segment"], as_index=False).agg(
        customers=("customer_id", "count"),
        median_recency_days=("recency_days", "median"),
        median_orders=("frequency", "median"),
        median_revenue=("monetary", "median"),
        total_revenue=("monetary", "sum"),
        avg_customer_age_days=("customer_age_days", "mean"),
        median_units=("total_units", "median"),
    ).sort_values("total_revenue", ascending=False, ignore_index=True)

    score = None
    if len(rfm) > n_clusters and len(np.unique(labels)) > 1:
        try:
            score = float(silhouette_score(X_scaled, labels)) if len(rfm) <= 5000 else None
        except Exception:
            score = None

    importance = pd.DataFrame({
        "feature": feature_cols,
        "cluster_center_spread": np.std(model.cluster_centers_, axis=0),
    }).sort_values("cluster_center_spread", ascending=False, ignore_index=True)
    # Restore human-readable, original-scale customer measures in the output.
    return SegmentationResult(
        customers=rfm,
        profile=profile,
        feature_importance=importance,
        n_clusters=n_clusters,
        silhouette=score,
        reference_date=pd.Timestamp(reference_date if reference_date is not None else transactions["order_date"].max()).normalize(),
    )


def _label_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Assign transparent heuristic names using cluster medians and relative ranks."""
    if profile.empty:
        return {}
    recency_rank = profile["recency_days"].rank(pct=True, ascending=False)
    frequency_rank = profile["frequency"].rank(pct=True, ascending=True)
    monetary_rank = profile["monetary"].rank(pct=True, ascending=True)
    names: dict[int, str] = {}
    for i, row in profile.iterrows():
        # Better recency means fewer days since the last purchase.
        recent = recency_rank.loc[i] >= 0.67
        frequent = frequency_rank.loc[i] >= 0.67
        high_value = monetary_rank.loc[i] >= 0.67
        stale = row["recency_days"] >= profile["recency_days"].median()
        if recent and (frequent or high_value):
            name = "Champions"
        elif recent:
            name = "Recent Customers"
        elif stale and (frequent or high_value):
            name = "At-Risk High Value"
        elif frequent:
            name = "Loyal Customers"
        elif high_value:
            name = "High-Spend Customers"
        else:
            name = "Needs Nurturing"
        names[int(row["cluster_id"])] = name
    return names
