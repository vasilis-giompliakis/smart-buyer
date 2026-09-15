# SmartBuyer

SmartBuyer is a Python project for collecting and analyzing supermarket
product prices, with the goal of optimizing shopping baskets and recommending
alternative products when exact items are unavailable.

## Data Pipeline
```text
API
 ↓
collect_products.py
 ↓
Raw JSON
 ↓
process_products.py
 ↓
Clean CSV
 ↓
basket_optimizer.py
 ↓
Basket optimization & product recommendations
```


## Current Features

- Collect supermarket product and price data
- Process raw API data into a clean tabular dataset
- Compare basket prices across supermarkets
- Find the cheapest supermarket for a basket
- Recommend substitute products using TF-IDF and cosine similarity

## Planned Improvements

- Store historical price data in a local database
- Automate daily data collection
- Build price-history visualizations and supermarket comparison dashboards
- Analyze price trends, discounts, and retailer pricing patterns
- Improve substitute recommendations using semantic product embeddings
- Develop ML-based product similarity and recommendation methods