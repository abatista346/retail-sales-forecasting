"""Clean and merge the raw Olist tables into a single order-item-level dataset.

Joins orders -> order_items -> products -> category translation -> customers,
drops canceled/unavailable orders (they don't reflect real demand), and
writes the result to data/processed/orders_merged.csv.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DROPPED_STATUSES = {"canceled", "unavailable"}


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "orders": pd.read_csv(RAW_DIR / "olist_orders_dataset.csv", parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]),
        "items": pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"]),
        "products": pd.read_csv(RAW_DIR / "olist_products_dataset.csv"),
        "customers": pd.read_csv(RAW_DIR / "olist_customers_dataset.csv"),
        "category_translation": pd.read_csv(RAW_DIR / "product_category_name_translation.csv"),
        "payments": pd.read_csv(RAW_DIR / "olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(RAW_DIR / "olist_order_reviews_dataset.csv", parse_dates=[
            "review_creation_date", "review_answer_timestamp",
        ]),
    }


def build_merged_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = tables["orders"]
    orders = orders[~orders["order_status"].isin(DROPPED_STATUSES)].copy()

    df = tables["items"].merge(orders, on="order_id", how="inner")
    df = df.merge(tables["products"][["product_id", "product_category_name"]], on="product_id", how="left")
    df = df.merge(tables["category_translation"], on="product_category_name", how="left")
    df = df.merge(
        tables["customers"][["customer_id", "customer_state", "customer_city"]],
        on="customer_id",
        how="left",
    )

    payments_agg = (
        tables["payments"]
        .groupby("order_id")
        .agg(payment_value=("payment_value", "sum"), payment_type=("payment_type", "first"))
        .reset_index()
    )
    df = df.merge(payments_agg, on="order_id", how="left")

    reviews = tables["reviews"][["order_id", "review_score"]].drop_duplicates("order_id")
    df = df.merge(reviews, on="order_id", how="left")

    df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days

    keep_cols = [
        "order_id", "order_item_id", "product_id", "seller_id",
        "order_purchase_timestamp", "order_status",
        "product_category_name_english", "price", "freight_value",
        "customer_state", "customer_city", "payment_type", "payment_value",
        "review_score", "delivery_days",
    ]
    return df[keep_cols]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_raw()
    merged = build_merged_table(tables)
    out_path = PROCESSED_DIR / "orders_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged):,} rows to {out_path}")
    print(merged.dtypes)


if __name__ == "__main__":
    main()
