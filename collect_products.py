"""
Collects supermarket product data from the API
and stores the raw product catalog as a JSON file.
"""

import subprocess
import json


# --------------------------------------------------
# Root product categories used to collect
# the complete supermarket product catalog
# --------------------------------------------------
ROOT_CATEGORIES = [
    "b2a17c2ad4235ea8574d602763a39395",
    "b2a17c2ad4235ea8574d602763988be6",
    "cFsywHNrftQ6yeittltcSFQ4qbM14Q3F",
    "fb7311d1172f411dba075194a4120689",
    "7f677200338b447cb4d469622588b749",
    "648a987abf254feb8cf62a10ea1eb117"
]


# --------------------------------------------------
# Download one page from the API using curl
# and return the parsed JSON response
# --------------------------------------------------
def fetch_page(category_id, page):

    url = (
        "https://api.posokanei.gov.gr/products"
        f"?category={category_id}"
        f"&page={page}"
        "&page_size=50"
    )

    result = subprocess.run(
        ["curl", url],
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    data = json.loads(result.stdout)

    return data


# --------------------------------------------------
# Download all products from one root category
# --------------------------------------------------
def fetch_category(category_id):

    # Download the first page to get the total number of pages
    first_page = fetch_page(category_id, 1)
    total_pages = first_page["total_pages"]

    print("\nCategory:", category_id)
    print("Total pages:", total_pages)

    # Keep the products already downloaded from page 1
    category_products = first_page["products"]

    # Download the remaining pages
    for page in range(2, total_pages + 1):

        print("Downloading page:", page)

        page_data = fetch_page(category_id, page)
        category_products.extend(page_data["products"])

    return category_products


# --------------------------------------------------
# Download products from all root categories
# --------------------------------------------------
def fetch_all_products():

    all_products = []

    for category_id in ROOT_CATEGORIES:

        category_products = fetch_category(
            category_id
        )

        all_products.extend(
            category_products
        )

    return all_products


# --------------------------------------------------
# Save the raw API data to a JSON file
# --------------------------------------------------
def save_products(products, filename):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            products,
            f,
            ensure_ascii=False
        )


# --------------------------------------------------
# Check whether duplicate product IDs exist
# --------------------------------------------------
def check_duplicates(products):

    product_ids = [
        product["id"]
        for product in products
    ]

    print("\nTotal products:", len(product_ids))
    print("Unique products:", len(set(product_ids)))


# --------------------------------------------------
# Main program
# --------------------------------------------------
def main():

    # Download the complete product catalog
    all_products = fetch_all_products()

    print(
        "\nTotal products collected:",
        len(all_products)
    )

    # Save raw API snapshot
    save_products(
        all_products,
        "all_products_2026-09-15.json"
    )

    # Basic data quality check
    check_duplicates(all_products)


# Run main() only when this file
# is executed directly
if __name__ == "__main__":
    main()