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


def evaluate_basket_for_retailer(df, products, basket, retailer_name):

    selected_items = []

    for product_id in basket:

        # Check if the exact product is available at this retailer
        exact_match = df[
            (df["product_id"] == product_id) &
            (df["retailer"] == retailer_name)
        ]

        if not exact_match.empty:
            price = exact_match.iloc[0]["price"]

            # construct the dictionary
            selected_item = {
                "requested_product_id": product_id,
                "selected_product_id": product_id,
                "match_type": "exact",
                "price": price
            }

            selected_items.append(selected_item)

        else:
            substitutes = find_substitute(
                df,
                products,
                product_id,
                retailer_name
            )

            best_substitute = substitutes.iloc[0]
            substitute_id = best_substitute["product_id"]

            substitute_match = df[
                (df["product_id"] == substitute_id) &
                (df["retailer"] == retailer_name)
                ]

            substitute_price = substitute_match.iloc[0]["price"]

            selected_item = {
                "requested_product_id": product_id,
                "selected_product_id": substitute_id,
                "match_type": "substitute",
                "price": substitute_price
            }

            selected_items.append(selected_item)

    total_price = 0

    for item in selected_items:
        total_price += item["price"]


    exact_matches = 0
    substitutions = 0

    for item in selected_items:
        if item["match_type"] == "exact":
            exact_matches += 1
        else:
            substitutions += 1


    result = {
        "retailer": retailer_name,
        "total_price": total_price,
        "exact_matches": exact_matches,
        "substitutions": substitutions,
        "items": selected_items
    }

    return result


def compare_retailers(df, products, basket):

    retailers = df["retailer"].unique()
    all_results = []

    for retailer in retailers:

        result = evaluate_basket_for_retailer(
            df,
            products,
            basket,
            retailer
        )

        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("total_price")

    return results_df


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

    # Evaluate basket for all retailers using substitutes when needed
    results_df = compare_retailers(
        df,
        products,
        basket
    )

    print("\nRetailer comparison:")
    print(
        results_df[
            ["retailer", "total_price", "exact_matches", "substitutions"]
        ].to_string(index=False)
    )

    # Select the best basket result
    best_basket_result = results_df.iloc[0]
    print("\nBest result is:")
    print(
        best_basket_result[
            ["retailer", "total_price", "exact_matches", "substitutions"]
        ]
    )


if __name__ == "__main__":
    main()
