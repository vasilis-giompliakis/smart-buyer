"""
Transforms the raw product catalog into a clean,
tabular dataset of product prices by retailer.
"""

import json
import pandas as pd


# --------------------------------------------------
# Load the raw product snapshot from JSON
# --------------------------------------------------
def load_raw_products(filename):

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        products = json.load(f)

    return products


# --------------------------------------------------
# Flatten the nested API data
#
# Raw structure:
# 1 product -> multiple retailer prices
#
# Output structure:
# 1 row = 1 product × 1 supermarket
# --------------------------------------------------
def flatten_products(products):

    rows = []

    for product in products:

        for price_info in product["retailer_prices"]:

            # Keep only Greek supermarket prices
            if price_info["country"] == "GR":

                rows.append({
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "brand": product["brand"],
                    "category": product["category"],
                    "subcategory": product["subcategory"],
                    "unit_quantity": product["unit_quantity"],
                    "unit": product["unit"],
                    "private_label": product["private_label"],
                    "image_url": product["image_url"],

                    "retailer_id": price_info["retailer"],
                    "retailer": price_info["retailer_display_name"],
                    "price": price_info["price"],
                    "price_normalized": price_info["price_normalized"],
                    "is_discount": price_info["is_discount"],
                    "last_updated": price_info["last_updated"]
                })

    df = pd.DataFrame(rows)

    return df


# --------------------------------------------------
# Run basic data quality checks
# --------------------------------------------------
def check_data_quality(df):

    print("\nDataset shape:")
    print(df.shape)

    print("\nUnique products:")
    print(df["product_id"].nunique())

    print("\nNumber of retailers:")
    print(df["retailer"].nunique())

    print("\nProducts per retailer:")
    print(df["retailer"].value_counts())

    # Check duplicate product-retailer combinations
    duplicates = df.duplicated(
        subset=["product_id", "retailer_id"]
    ).sum()

    print("\nDuplicate product-retailer rows:")
    print(duplicates)

    # Check missing or invalid prices
    print("\nMissing prices:")
    print(df["price"].isna().sum())

    print("\nNon-positive prices:")
    print((df["price"] <= 0).sum())

    # Check missing values in all columns
    print("\nMissing values per column:")
    print(df.isna().sum())


# --------------------------------------------------
# Save the processed analytical dataset to CSV
# --------------------------------------------------
def save_processed_data(df, filename):

    df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved:", filename)


# --------------------------------------------------
# Main program
# --------------------------------------------------
def main():

    # Load raw API snapshot
    products = load_raw_products(
        "all_products_2026-09-15.json"
    )

    # Convert nested JSON into analytical table
    df = flatten_products(products)

    # Inspect dataset quality before saving
    check_data_quality(df)

    # Save processed dataset
    save_processed_data(
        df,
        "all_supermarket_prices_2026-09-15.csv"
    )


# Run main() only when this file
# is executed directly
if __name__ == "__main__":
    main()