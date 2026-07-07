# ==========================================
# DASHBOARD DATA PREPARATION
# Run this on your FULL dataset (after merge + feature engineering)
# It aggregates product-level stats for the dashboard
# ==========================================

import pandas as pd
import json
from pathlib import Path

DATA_PATH = Path("data/eda_ready_data.csv")
data = pd.read_csv(DATA_PATH)

# ==========================================
# STEP 1: AGGREGATE BY PRODUCT
# ==========================================

# Make sure 'data' is your merged dataframe with all features
# If you added is_recommended back, use it. Otherwise we estimate from rating.

product_stats = data.groupby(["product_id", "product_name", "brand_name"]).agg(
    avg_rating=("rating", "mean"),
    total_reviews=("rating", "count"),
    price=("price_usd", "first"),
    category=("primary_category", "first"),
    has_fragrance=("has_fragrance", "first"),
    has_alcohol=("has_alcohol", "first"),
    has_paraben=("has_paraben", "first"),
    has_sulfate=("has_sulfate", "first"),
    irritation_rate=("irritation_flag", "mean"),
    ingredients=("ingredients", "first")
).reset_index()

# Recommendation rate: % of reviews with rating >= 4
recommend_data = data.copy()
recommend_data["is_positive"] = (recommend_data["rating"] >= 4).astype(int)

recommend_rate = recommend_data.groupby("product_id")["is_positive"].mean().reset_index()
recommend_rate.columns = ["product_id", "recommend_pct"]

product_stats = product_stats.merge(recommend_rate, on="product_id", how="left")

# ==========================================
# STEP 2: SKIN TYPE SPECIFIC STATS
# ==========================================

skin_types = ["combination", "dry", "normal", "oily"]

for skin in skin_types:
    col = f"skin_type_{skin}"
    if col in data.columns:
        skin_data = data[data[col] == True]
        
        # Irritation rate for this skin type per product
        skin_irr = skin_data.groupby("product_id")["irritation_flag"].mean().reset_index()
        skin_irr.columns = ["product_id", f"irritation_{skin}"]
        product_stats = product_stats.merge(skin_irr, on="product_id", how="left")
        
        # Average rating for this skin type per product
        skin_rat = skin_data.groupby("product_id")["rating"].mean().reset_index()
        skin_rat.columns = ["product_id", f"rating_{skin}"]
        product_stats = product_stats.merge(skin_rat, on="product_id", how="left")
        
        # Review count for this skin type per product
        skin_cnt = skin_data.groupby("product_id")["rating"].count().reset_index()
        skin_cnt.columns = ["product_id", f"reviews_{skin}"]
        product_stats = product_stats.merge(skin_cnt, on="product_id", how="left")

# ==========================================
# STEP 3: COMPUTE SUITABILITY SCORE
# ==========================================

# Suitability = weighted combination of:
# - Low irritation rate (40%)
# - High rating (30%)
# - High recommendation rate (30%)

product_stats["base_suitability"] = (
    (1 - product_stats["irritation_rate"]) * 40 +
    (product_stats["avg_rating"] / 5) * 30 +
    product_stats["recommend_pct"] * 30
)

# Normalize to 0-100
max_suit = product_stats["base_suitability"].max()
min_suit = product_stats["base_suitability"].min()
product_stats["suitability_score"] = (
    (product_stats["base_suitability"] - min_suit) / (max_suit - min_suit) * 100
).round(1)

# ==========================================
# STEP 4: FILTER TO PRODUCTS WITH ENOUGH REVIEWS
# ==========================================

# Keep only products with at least 10 reviews for reliability
product_stats = product_stats[product_stats["total_reviews"] >= 10]

print(f"Total products for dashboard: {len(product_stats)}")
print(f"\nSample output:")
print(product_stats[["product_name", "brand_name", "avg_rating", "total_reviews",
                       "recommend_pct", "irritation_rate", "suitability_score"]].head(10))

# ==========================================
# STEP 5: EXPORT TO JSON FOR DASHBOARD
# ==========================================

# Round numeric columns
for col in product_stats.select_dtypes(include=["float64"]).columns:
    product_stats[col] = product_stats[col].round(4)

# Convert to list of dicts
dashboard_data = product_stats.to_dict(orient="records")

# Save as JSON
output_path = "dashboard_data.json"
with open(output_path, "w") as f:
    json.dump(dashboard_data, f, indent=2)

print(f"\nDashboard data saved to: {output_path}")
print(f"Total products: {len(dashboard_data)}")

# ==========================================
# STEP 6: ALSO EXPORT AS CSV FOR REFERENCE
# ==========================================

product_stats.to_csv("dashboard_data.csv", index=False)
print("Also saved as dashboard_data.csv")

# ==========================================
# QUICK STATS FOR VERIFICATION
# ==========================================

print("\n" + "=" * 50)
print("DASHBOARD DATA SUMMARY")
print("=" * 50)
print(f"Products: {len(product_stats)}")
print(f"Avg rating range: {product_stats['avg_rating'].min():.2f} - {product_stats['avg_rating'].max():.2f}")
print(f"Price range: ${product_stats['price'].min():.0f} - ${product_stats['price'].max():.0f}")
print(f"Irritation rate range: {product_stats['irritation_rate'].min():.2%} - {product_stats['irritation_rate'].max():.2%}")
print(f"\nAllergen breakdown:")
print(f"  Products with fragrance: {product_stats['has_fragrance'].sum()}")
print(f"  Products with alcohol: {product_stats['has_alcohol'].sum()}")
print(f"  Products with paraben: {product_stats['has_paraben'].sum()}")
print(f"  Products with sulfate: {product_stats['has_sulfate'].sum()}")
print(f"\nCategories: {product_stats['category'].value_counts().to_dict()}")
