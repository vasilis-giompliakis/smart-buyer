"""
Finds the cheapest supermarket for a shopping basket
and recommends substitute products when exact items are unavailable.
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    df = pd.read_csv("all_supermarket_prices.csv")

    return df


# --------------------------------------------------
# Create one row per unique product
# We keep only the product-level information
# --------------------------------------------------
def prepare_products(df):

    products = (
        df[
            [
                "product_id",
                "product_name",
                "brand",
                "subcategory",
                "unit_quantity",
                "unit"
            ]
        ]
        .drop_duplicates(subset="product_id")
        .reset_index(drop=True)
    )

    return products


# --------------------------------------------------
# Find the cheapest supermarket that contains ALL products of the basket
# --------------------------------------------------
def find_cheapest_basket(df, basket):

    # Keep only the rows that belong to products included in the basket
    basket_df = df[df["product_id"].isin(basket)    ]

    # Count how many different basket products each retailer has
    retailer_counts = (
        basket_df
        .groupby("retailer")["product_id"]
        .nunique()
    )

    # Keep only retailers that contain every product of the basket
    complete_retailers = retailer_counts[retailer_counts == len(basket)].index

    # If no retailer has the complete basket, return empty results
    if len(complete_retailers) == 0:
        return None, None, None

    # Keep only rows from retailers that have the complete basket
    complete_basket_df = basket_df[
        basket_df["retailer"].isin(complete_retailers)
    ]

    # Calculate the total basket cost for each retailer
    basket_totals = (
        complete_basket_df
        .groupby("retailer")["price"]
        .sum()
        .sort_values()
    )

    # Retailer with the minimum basket cost
    cheapest_retailer = basket_totals.idxmin()

    # Minimum basket cost
    cheapest_total = basket_totals.min()

    return basket_totals, cheapest_retailer, cheapest_total


# --------------------------------------------------
# Find substitute products for a missing product inside a specific retailer
# --------------------------------------------------
def find_substitute(
    df,
    products,
    missing_product_id,
    retailer_name,
    top_n=5
):

    # Find the product information for the missing product
    missing_product = products[
        products["product_id"] == missing_product_id
    ].iloc[0]

    # Keep only products that are available in the selected retailer
    retailer_products = (
        df[
            df["retailer"] == retailer_name
        ][
            [
                "product_id",
                "product_name",
                "brand",
                "subcategory",
                "unit_quantity",
                "unit"
            ]
        ]
        .drop_duplicates(subset="product_id")
        .reset_index(drop=True)
    )

    # For now, candidate substitutes must belong to the same subcategory as the missing product
    candidate_products = retailer_products[
        retailer_products["subcategory"]
        == missing_product["subcategory"]
    ].reset_index(drop=True)

    # If the retailer has no products in the same subcategory, no substitute exists
    if len(candidate_products) == 0:
        return None

    # Create the TF-IDF model
    vectorizer = TfidfVectorizer()

    # Convert candidate product names into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(
        candidate_products["product_name"]
    )

    # Convert the missing product name using the SAME vocabulary
    missing_vector = vectorizer.transform([
        missing_product["product_name"]
    ])

    # Compare the missing product with every candidate product
    similarities = cosine_similarity(
        missing_vector,
        tfidf_matrix
    )[0]

    # Add the similarity score to each candidate product
    candidate_products["similarity"] = similarities

    # Sort from most similar to least similar and keep only the top N recommendations
    recommendations = (
        candidate_products
        .sort_values(
            "similarity",
            ascending=False
        )
        .head(top_n)
    )

    return recommendations


# --------------------------------------------------
# Main program
# --------------------------------------------------
def main():

    # Load dataset
    df = load_data()

    # Create the unique-products table
    products = prepare_products(df)

    # Example shopping basket
    basket = [
        "31d63a58f22543ebb99b84b8957bbe85",
        "517a59d4692e4cb79c4937b22ac6a5a8",
        "1b6e69836b514072a080cc542c7584dc"
    ]

    # Find cheapest retailer for the complete basket
    basket_totals, winner, total = find_cheapest_basket(df, basket)

    print("\nBasket totals:")
    print(basket_totals)

    print("\nCheapest retailer:")
    print(winner)

    print("\nTotal:")
    print(total)

    # Example:  GOURMET product is missing from Synka
    missing_product_id = "517a59d4692e4cb79c4937b22ac6a5a8"
    retailer_name = "Synka"

    # Find possible substitute products
    recommendations = find_substitute(
        df,
        products,
        missing_product_id,
        retailer_name
    )

    print("\nSubstitute recommendations:")
    print(
        recommendations[
            ["product_name", "similarity"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()