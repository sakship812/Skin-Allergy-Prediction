import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Skincare Safety Recommendation Engine",
    page_icon="🧴",
    layout="wide",
)

DATA_PATH = Path("data/dashboard_data.csv")

ALLERGEN_COLUMNS = {
    "Fragrance": "has_fragrance",
    "Alcohol": "has_alcohol",
    "Paraben": "has_paraben",
    "Sulfate": "has_sulfate",
    "Essential Oils": "has_essential_oil",
    "Retinol": "has_retinol",
    "Formaldehyde Releasers": "has_formaldehyde",
    "Phthalates": "has_phthalate",
    "Mineral Oil": "has_mineral_oil",
    "Silicones": "has_silicone",
}

INGREDIENT_SEARCH_MAP = {
    "Niacinamide": "niacinamide",
    "Vitamin C": "vitamin c|ascorbic acid|ascorbyl",
    "Hyaluronic Acid": "hyaluronic acid|sodium hyaluronate",
    "Retinol": "retinol|retinal|retinoic",
    "Salicylic Acid": "salicylic acid",
    "Ceramides": "ceramide",
    "Peptides": "peptide",
    "Squalane": "squalane|squalene",
    "Glycolic Acid": "glycolic acid",
    "Centella / Cica": "centella|madecassoside|asiaticoside|cica",
    "Aloe Vera": "aloe",
    "Green Tea": "green tea|camellia sinensis",
    "Azelaic Acid": "azelaic acid",
    "Kojic Acid": "kojic acid",
    "Zinc": "zinc",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            "Missing data/dashboard_data.csv. Run dashboard_data_prep.py locally first, "
            "then move the generated dashboard_data.csv into a folder named data."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    if "ingredients" in df.columns:
        df["ingredients"] = df["ingredients"].fillna("").str.lower()

    for col in ALLERGEN_COLUMNS.values():
        if col not in df.columns:
            df[col] = 0

    if "price" in df.columns and "price_usd" not in df.columns:
        df["price_usd"] = df["price"]

    if "category" not in df.columns and "primary_category" in df.columns:
        df["category"] = df["primary_category"]

    return df


def compute_suitability(row: pd.Series, skin_type: str) -> float:
    irr_col = f"irritation_{skin_type}"
    rating_col = f"rating_{skin_type}"

    irritation = row.get(irr_col, row.get("irritation_rate", 0))
    rating = row.get(rating_col, row.get("avg_rating", row.get("rating", 3)))
    recommend = row.get("recommend_pct", 0.5)

    if pd.isna(irritation):
        irritation = row.get("irritation_rate", 0)
    if pd.isna(rating):
        rating = row.get("avg_rating", row.get("rating", 3))
    if pd.isna(recommend):
        recommend = 0.5

    return round((1 - irritation) * 40 + (rating / 5) * 30 + recommend * 30, 1)


def score_label(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Fair"
    return "Low"


def main() -> None:
    df = load_data()

    st.title("🧴 Skincare Safety Recommendation Engine")
    st.write(
        "A machine-learning powered skincare recommendation dashboard that combines "
        "ingredient safety signals, Sephora review data, skin type, pricing, ratings, "
        "and allergen filters."
    )

    with st.sidebar:
        st.header("Your Preferences")

        skin_type = st.selectbox(
            "Skin type",
            ["combination", "dry", "normal", "oily"],
            format_func=str.capitalize,
        )

        selected_allergens = st.multiselect(
            "Ingredients/allergens to avoid",
            list(ALLERGEN_COLUMNS.keys()),
            default=[],
        )

        desired_ingredients = st.multiselect(
            "Ingredients you want",
            list(INGREDIENT_SEARCH_MAP.keys()),
            default=[],
        )

        categories = ["All"]
        if "category" in df.columns:
            categories += sorted(df["category"].dropna().astype(str).unique().tolist())

        category = st.selectbox("Category", categories)

        price_range = st.selectbox(
            "Price range",
            ["Any", "Under $25", "$25-$50", "$50-$100", "$100+"],
        )

        sort_by = st.selectbox(
            "Sort by",
            ["Best Suitability", "Highest Rating", "Lowest Irritation", "Most Reviewed", "Lowest Price"],
        )

    filtered = df.copy()

    for allergen in selected_allergens:
        col = ALLERGEN_COLUMNS[allergen]
        if col in filtered.columns:
            filtered = filtered[filtered[col].fillna(0).astype(int) == 0]

    if desired_ingredients and "ingredients" in filtered.columns:
        for ing in desired_ingredients:
            pattern = INGREDIENT_SEARCH_MAP[ing]
            filtered = filtered[
                filtered["ingredients"].str.contains(pattern, case=False, na=False, regex=True)
            ]

    if category != "All" and "category" in filtered.columns:
        filtered = filtered[filtered["category"].astype(str) == category]

    if "price_usd" in filtered.columns:
        if price_range == "Under $25":
            filtered = filtered[filtered["price_usd"] < 25]
        elif price_range == "$25-$50":
            filtered = filtered[(filtered["price_usd"] >= 25) & (filtered["price_usd"] <= 50)]
        elif price_range == "$50-$100":
            filtered = filtered[(filtered["price_usd"] >= 50) & (filtered["price_usd"] <= 100)]
        elif price_range == "$100+":
            filtered = filtered[filtered["price_usd"] >= 100]

    filtered["suitability"] = filtered.apply(lambda row: compute_suitability(row, skin_type), axis=1)

    irr_col = f"irritation_{skin_type}"
    rating_col = f"rating_{skin_type}"
    if irr_col not in filtered.columns:
        irr_col = "irritation_rate"
    if rating_col not in filtered.columns:
        rating_col = "avg_rating" if "avg_rating" in filtered.columns else "rating"

    if sort_by == "Best Suitability":
        filtered = filtered.sort_values("suitability", ascending=False)
    elif sort_by == "Highest Rating" and rating_col in filtered.columns:
        filtered = filtered.sort_values(rating_col, ascending=False)
    elif sort_by == "Lowest Irritation" and irr_col in filtered.columns:
        filtered = filtered.sort_values(irr_col, ascending=True)
    elif sort_by == "Most Reviewed" and "total_reviews" in filtered.columns:
        filtered = filtered.sort_values("total_reviews", ascending=False)
    elif sort_by == "Lowest Price" and "price_usd" in filtered.columns:
        filtered = filtered.sort_values("price_usd", ascending=True)

    st.subheader(f"{len(filtered):,} matching products")

    if filtered.empty:
        st.warning("No products match these filters. Try removing one filter.")
        return

    top = filtered.head(50).copy()

    for _, row in top.iterrows():
        name = row.get("product_name", "Unknown product")
        brand = row.get("brand_name", "Unknown brand")
        price = row.get("price_usd", np.nan)
        rating = row.get(rating_col, row.get("avg_rating", np.nan))
        reviews = row.get("total_reviews", np.nan)
        irritation = row.get(irr_col, row.get("irritation_rate", np.nan))
        score = row["suitability"]

        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 1.2, 1.4])

            with c1:
                st.markdown(f"### {name}")
                st.caption(str(brand))
                warnings = [
                    label for label, col in ALLERGEN_COLUMNS.items()
                    if col in row.index and not pd.isna(row[col]) and int(row[col]) == 1
                ]
                if warnings:
                    st.warning("Contains: " + ", ".join(warnings))
                else:
                    st.success("No flagged allergens detected")

            with c2:
                st.metric("Safety Score", f"{score:.0f}", score_label(score))
                if not pd.isna(irritation):
                    st.metric("Irritation Rate", f"{irritation * 100:.1f}%")

            with c3:
                if not pd.isna(price):
                    st.metric("Price", f"${price:.0f}")
                if not pd.isna(rating):
                    st.metric("Rating", f"{rating:.1f}/5")
                if not pd.isna(reviews):
                    st.caption(f"{int(reviews):,} reviews")

    st.divider()
    st.caption(
        "Academic project by Sakshi Patil, Harshmeet Kaur, Sai Athale, and Shreya Pandey. "
        "This tool is not medical advice."
    )


if __name__ == "__main__":
    main()
