"""preprocessing.py — data cleaning, feature engineering, churn labeling"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Fix common typos
    df.rename(columns={
        "product_name_lenght":        "product_name_length",
        "product_description_lenght": "product_description_length",
    }, inplace=True)

    # Parse dates
    date_cols = ["order_purchase_timestamp","order_approved_at",
                 "order_delivered_customer_date","order_estimated_delivery_date",
                 "shipping_limit_date"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Drop rows missing price
    df.dropna(subset=["price"], inplace=True)
    df.drop_duplicates(subset=["order_id","order_item_id"], inplace=True) if "order_item_id" in df.columns else None

    # Fill categoricals
    if "product_category_name" in df.columns:
        df["product_category_name"].fillna("unknown", inplace=True)
    if "payment_type" in df.columns:
        df = df[df["payment_type"] != "not_defined"]

    # Numeric medians
    num_cols = ["product_name_length","product_description_length","product_photos_qty",
                "product_weight_g","product_length_cm","product_height_cm","product_width_cm","review_score"]
    for c in num_cols:
        if c in df.columns:
            df[c].fillna(df[c].median(), inplace=True)

    # Focus on delivered orders
    if "order_status" in df.columns:
        df_d = df[df["order_status"] == "delivered"].copy()
    else:
        df_d = df.copy()

    # Derived columns
    if "order_purchase_timestamp" in df_d.columns:
        if "order_delivered_customer_date" in df_d.columns:
            df_d["delivery_days"] = (df_d["order_delivered_customer_date"] -
                                     df_d["order_purchase_timestamp"]).dt.days
        if "order_estimated_delivery_date" in df_d.columns:
            df_d["estimated_days"] = (df_d["order_estimated_delivery_date"] -
                                      df_d["order_purchase_timestamp"]).dt.days
            df_d["delivery_delay"] = df_d.get("delivery_days", 0) - df_d["estimated_days"]
        df_d["order_month"] = df_d["order_purchase_timestamp"].dt.to_period("M")
        df_d["order_year"]  = df_d["order_purchase_timestamp"].dt.year
        df_d["order_hour"]  = df_d["order_purchase_timestamp"].dt.hour

    if "price" in df_d.columns and "freight_value" in df_d.columns:
        df_d["total_order_value"] = df_d["price"] + df_d["freight_value"]

    return df_d


def build_customer_features(df: pd.DataFrame, churn_days: int = 90) -> pd.DataFrame:
    """Aggregate transaction-level data to one row per customer."""
    ref_date = df["order_purchase_timestamp"].max() if "order_purchase_timestamp" in df.columns else pd.Timestamp.now()

    agg = df.groupby("customer_id").agg(
        last_purchase   =("order_purchase_timestamp", "max"),
        order_count     =("order_id", "nunique"),
        total_revenue   =("total_order_value", "sum"),
        avg_order_value =("total_order_value", "mean"),
        avg_freight     =("freight_value", "mean"),
        avg_installments=("payment_installments", "mean"),
        avg_review      =("review_score", "mean"),
        unique_cats     =("product_category_name", "nunique"),
        avg_delay       =("delivery_delay", "mean"),
    ).reset_index()

    agg["recency_days"]  = (ref_date - agg["last_purchase"]).dt.days
    agg["recency_score"] = 1 / (1 + agg["recency_days"])  # higher = more recent
    agg["payment_complexity"] = agg["avg_installments"] / agg["avg_order_value"].clip(lower=1)

    # Churn label: no purchase in last `churn_days`
    agg["churn"] = (agg["recency_days"] > churn_days).astype(int)

    agg.drop(columns=["last_purchase"], inplace=True)
    return agg


def prepare_ml_features(cust_df: pd.DataFrame):
    """Return scaled feature matrix X, labels y, feature names, scaler."""
    feature_cols = ["recency_score","order_count","avg_order_value","avg_freight",
                    "avg_installments","avg_review","unique_cats","avg_delay",
                    "payment_complexity","recency_days"]
    feature_cols = [c for c in feature_cols if c in cust_df.columns]

    X = cust_df[feature_cols].copy()
    X.fillna(X.median(), inplace=True)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    y = cust_df["churn"].values if "churn" in cust_df.columns else None
    return X_scaled, y, feature_cols, scaler
