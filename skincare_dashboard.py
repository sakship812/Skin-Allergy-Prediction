# ==========================================
# SKINCARE RECOMMENDATION DASHBOARD v3
# Run with: streamlit run skincare_dashboard.py
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import math

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Skincare Recommendation Engine",
    page_icon="🧴",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

    .main { background-color: #FDFBF7; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #1A1A1A; }
    p, span, div, label { font-family: 'DM Sans', sans-serif !important; }

    .stSelectbox label, .stMultiSelect label {
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #A0937D !important;
        font-weight: 600 !important;
    }

    .product-card {
        background: #FDFBF7;
        border: 1px solid #E8E2D6;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .product-card:hover {
        border-color: #D4A853;
        box-shadow: 0 8px 30px rgba(212,168,83,0.12);
    }

    .brand-name {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #A0937D;
        font-weight: 600;
    }
    .product-name {
        font-family: 'Playfair Display', serif !important;
        font-size: 18px;
        font-weight: 600;
        color: #1A1A1A;
        margin: 2px 0 8px;
    }
    .price-tag {
        font-size: 20px;
        font-weight: 700;
        color: #1A1A1A;
    }

    .score-circle {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
    }
    .score-excellent { border: 3px solid #4A7C59; color: #4A7C59; background: rgba(74,124,89,0.07); }
    .score-good { border: 3px solid #D4A853; color: #D4A853; background: rgba(212,168,83,0.07); }
    .score-fair { border: 3px solid #C47D3B; color: #C47D3B; background: rgba(196,125,59,0.07); }
    .score-low { border: 3px solid #8B3A3A; color: #8B3A3A; background: rgba(139,58,58,0.07); }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-safe { background: rgba(74,124,89,0.1); color: #4A7C59; border: 1px solid rgba(74,124,89,0.2); }
    .badge-warn { background: rgba(139,58,58,0.1); color: #8B3A3A; border: 1px solid rgba(139,58,58,0.2); }
    .badge-stat { background: #F5F1EA; color: #6B5F4F; border: 1px solid #E8E2D6; }
    .badge-ingredient { background: rgba(74,124,89,0.15); color: #3A6B49; border: 1px solid rgba(74,124,89,0.3); font-weight: 600; }

    .top-pick {
        display: inline-block;
        background: #4A7C59;
        color: white;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 4px 10px;
        border-radius: 20px;
    }

    .star { color: #D4A853; font-size: 14px; }
    .star-empty { color: #E8E2D6; font-size: 14px; }

    div[data-testid="stMetric"] {
        background: #F5F1EA;
        border: 1px solid #E8E2D6;
        border-radius: 12px;
        padding: 12px 16px;
    }

    .methodology-box {
        background: #F5F1EA;
        border: 1px solid #E8E2D6;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 24px;
    }

    /* Pagination styles */
    .pagination {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        margin: 24px 0;
        flex-wrap: wrap;
    }
    .page-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 40px;
        height: 40px;
        padding: 0 12px;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 500;
        font-family: 'DM Sans', sans-serif;
        cursor: pointer;
        border: 1px solid #E8E2D6;
        background: #FDFBF7;
        color: #6B5F4F;
        text-decoration: none;
    }
    .page-btn-active {
        background: #1A1A1A;
        color: white;
        border-color: #1A1A1A;
    }
    .page-info {
        font-size: 13px;
        color: #A0937D;
        text-align: center;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SKIN CONCERN DEFINITIONS
# ==========================================

SKIN_CONCERNS = {
    "Acne & Breakouts": {
        "ingredients": ["salicylic acid", "benzoyl peroxide", "tea tree", "niacinamide", "zinc", "azelaic acid", "sulfur"],
        "icon": "🔴"
    },
    "Anti-Aging & Wrinkles": {
        "ingredients": ["retinol", "retinal", "peptide", "collagen", "hyaluronic acid", "vitamin c", "ascorbic acid", "bakuchiol"],
        "icon": "⏳"
    },
    "Dryness & Dehydration": {
        "ingredients": ["hyaluronic acid", "ceramide", "glycerin", "squalane", "shea butter", "jojoba", "aloe"],
        "icon": "💧"
    },
    "Dark Spots & Hyperpigmentation": {
        "ingredients": ["vitamin c", "ascorbic acid", "niacinamide", "kojic acid", "arbutin", "tranexamic", "licorice"],
        "icon": "🌗"
    },
    "Sensitivity & Redness": {
        "ingredients": ["centella", "cica", "madecassoside", "aloe", "chamomile", "bisabolol", "allantoin", "oat"],
        "icon": "🌿"
    },
    "Oiliness & Pores": {
        "ingredients": ["niacinamide", "salicylic acid", "clay", "charcoal", "witch hazel", "zinc", "bha"],
        "icon": "✨"
    },
}

DESIRED_INGREDIENTS = [
    "Niacinamide",
    "Vitamin C",
    "Hyaluronic Acid",
    "Retinol",
    "Salicylic Acid",
    "Ceramides",
    "Peptides",
    "Squalane",
    "Glycolic Acid",
    "Centella / Cica",
    "Aloe Vera",
    "Green Tea",
    "Snail Mucin",
    "Azelaic Acid",
    "Bakuchiol",
    "Collagen",
    "Kojic Acid",
    "Zinc",
    "Turmeric",
    "Charcoal",
]

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
    "Centella / Cica": "centella|madecassoside|asiaticoside",
    "Aloe Vera": "aloe",
    "Green Tea": "green tea|camellia sinensis",
    "Snail Mucin": "snail|secretion filtrate",
    "Azelaic Acid": "azelaic acid",
    "Bakuchiol": "bakuchiol",
    "Collagen": "collagen",
    "Kojic Acid": "kojic acid",
    "Zinc": "zinc",
    "Turmeric": "turmeric|curcuma",
    "Charcoal": "charcoal",
}

PRODUCTS_PER_PAGE = 20


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_and_process_data():
    DATA_PATH = "eda_ready_data.csv"

    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Could not find data file at: {DATA_PATH}")
        st.info("""
        **How to fix this:**
        1. Run your ML_proj.py in Spyder first
        2. Make sure eda_ready_data.csv is in the same folder as this dashboard
        3. Restart: `streamlit run skincare_dashboard.py`
        """)
        st.stop()

    # Detect skin type columns
    skin_cols = [c for c in data.columns if c.startswith("skin_type_")]
    if len(skin_cols) == 0 and "skin_type" in data.columns:
        data["skin_type"] = data["skin_type"].fillna("Unknown")
        skin_dummies = pd.get_dummies(data["skin_type"], prefix="skin_type")
        data = pd.concat([data, skin_dummies], axis=1)

    # Ensure ingredients column
    if "ingredients" in data.columns:
        data["ingredients"] = data["ingredients"].fillna("").str.lower()

        allergen_checks = {
            "has_fragrance": "fragrance",
            "has_alcohol": "alcohol",
            "has_paraben": "paraben",
            "has_sulfate": "sulfate",
            "has_essential_oil": "essential oil|eucalyptus|tea tree|lavender oil|peppermint",
            "has_retinol": "retinol|retinal|retinoic",
            "has_formaldehyde": "formaldehyde|dmdm hydantoin|quaternium-15|imidazolidinyl urea",
            "has_phthalate": "phthalate",
            "has_mineral_oil": "mineral oil|paraffinum liquidum|petrolatum",
            "has_silicone": "dimethicone|cyclomethicone|siloxane|silicone",
        }
        for col, pattern in allergen_checks.items():
            if col not in data.columns:
                data[col] = data["ingredients"].str.contains(pattern, na=False).astype(int)

    # Create irritation flag if missing
    if "irritation_flag" not in data.columns:
        keywords = ["rash", "burn", "itch", "irritation", "allergy", "breakout", "redness"]
        if "review_text" in data.columns:
            data["irritation_flag"] = data["review_text"].str.contains(
                "|".join(keywords), case=False, na=False
            ).astype(int)
        else:
            data["irritation_flag"] = 0

    # Category column
    cat_col = "secondary_category" if "secondary_category" in data.columns else "primary_category"

    # Aggregate by product
    agg_cols = {
        "rating": "mean",
        "price_usd": "first",
        "ingredients": "first",
        "has_fragrance": "first",
        "has_alcohol": "first",
        "has_paraben": "first",
        "has_sulfate": "first",
        "has_essential_oil": "first",
        "has_retinol": "first",
        "has_formaldehyde": "first",
        "has_phthalate": "first",
        "has_mineral_oil": "first",
        "has_silicone": "first",
        "irritation_flag": "mean",
    }
    agg_cols = {k: v for k, v in agg_cols.items() if k in data.columns}

    if cat_col in data.columns:
        agg_cols[cat_col] = "first"

    product_stats = data.groupby(
        ["product_id", "product_name", "brand_name"]
    ).agg(**{
        col: pd.NamedAgg(column=col, aggfunc=func)
        for col, func in agg_cols.items()
    }).reset_index()

    # Review count
    review_counts = data.groupby("product_id").size().reset_index(name="total_reviews")
    product_stats = product_stats.merge(review_counts, on="product_id", how="left")

    # Recommendation rate
    if "is_recommended" in data.columns:
        rec_rate = data.groupby("product_id")["is_recommended"].mean().reset_index()
        rec_rate.columns = ["product_id", "recommend_pct"]
    else:
        data["is_positive"] = (data["rating"] >= 4).astype(int)
        rec_rate = data.groupby("product_id")["is_positive"].mean().reset_index()
        rec_rate.columns = ["product_id", "recommend_pct"]

    product_stats = product_stats.merge(rec_rate, on="product_id", how="left")

    # Skin type specific stats
    skin_types = ["combination", "dry", "normal", "oily"]
    for skin in skin_types:
        col = f"skin_type_{skin}"
        if col in data.columns:
            skin_data = data[data[col] == True]
            if len(skin_data) > 0:
                skin_irr = skin_data.groupby("product_id")["irritation_flag"].mean().reset_index()
                skin_irr.columns = ["product_id", f"irritation_{skin}"]
                product_stats = product_stats.merge(skin_irr, on="product_id", how="left")

                skin_rat = skin_data.groupby("product_id")["rating"].mean().reset_index()
                skin_rat.columns = ["product_id", f"rating_{skin}"]
                product_stats = product_stats.merge(skin_rat, on="product_id", how="left")

                skin_cnt = skin_data.groupby("product_id")["rating"].count().reset_index()
                skin_cnt.columns = ["product_id", f"reviews_{skin}"]
                product_stats = product_stats.merge(skin_cnt, on="product_id", how="left")

    # Filter to products with enough reviews
    product_stats = product_stats[product_stats["total_reviews"] >= 10]

    return product_stats, cat_col


def compute_suitability(row, skin_type):
    irr_col = f"irritation_{skin_type}"
    rat_col = f"rating_{skin_type}"

    irritation = row.get(irr_col, row.get("irritation_flag", 0))
    rating = row.get(rat_col, row.get("rating", 3))
    recommend = row.get("recommend_pct", 0.5)

    if pd.isna(irritation): irritation = row.get("irritation_flag", 0)
    if pd.isna(rating): rating = row.get("rating", 3)
    if pd.isna(recommend): recommend = 0.5

    score = (1 - irritation) * 40 + (rating / 5) * 30 + recommend * 30
    return round(score, 1)


def render_stars(rating):
    full = int(rating)
    return f'<span class="star">{"★" * full}</span><span class="star-empty">{"☆" * (5 - full)}</span> <span style="color:#A0937D;font-size:13px;">{rating:.1f}</span>'


def render_product_card(row, skin_type, cat_col, is_top_pick=False, desired_ingredients=None):
    suitability = compute_suitability(row, skin_type)

    irr_col = f"irritation_{skin_type}"
    rat_col = f"rating_{skin_type}"
    rev_col = f"reviews_{skin_type}"

    irritation = row.get(irr_col, row.get("irritation_flag", 0))
    skin_rating = row.get(rat_col, row.get("rating", 0))
    skin_reviews = row.get(rev_col, 0)

    if pd.isna(irritation): irritation = row.get("irritation_flag", 0)
    if pd.isna(skin_rating): skin_rating = row.get("rating", 0)
    if pd.isna(skin_reviews): skin_reviews = 0

    irr_pct = irritation * 100

    if suitability >= 85:
        score_class, score_label = "score-excellent", "Excellent"
    elif suitability >= 70:
        score_class, score_label = "score-good", "Good"
    elif suitability >= 55:
        score_class, score_label = "score-fair", "Fair"
    else:
        score_class, score_label = "score-low", "Low"

    if irr_pct <= 5:
        irr_icon, irr_color = "✓", "#4A7C59"
    elif irr_pct <= 10:
        irr_icon, irr_color = "⚡", "#C47D3B"
    else:
        irr_icon, irr_color = "⚠️", "#8B3A3A"

    # Build simple allergen text
    allergen_names = [
        ("has_fragrance", "Fragrance"),
        ("has_alcohol", "Alcohol"),
        ("has_paraben", "Paraben"),
        ("has_sulfate", "Sulfate"),
        ("has_essential_oil", "Essential Oils"),
        ("has_retinol", "Retinol"),
        ("has_formaldehyde", "Formaldehyde"),
        ("has_phthalate", "Phthalates"),
        ("has_mineral_oil", "Mineral Oil"),
        ("has_silicone", "Silicones"),
    ]
    present = []
    free_of = []
    for key, label in allergen_names:
        val = row.get(key, 0)
        if pd.isna(val): val = 0
        if val == 1:
            present.append(label)
        else:
            free_of.append(label)

    if present:
        allergen_html = "Contains: " + ", ".join(present)
    else:
        allergen_html = ""

    # Only show TOP PICK if explicitly marked
    top_pick = '<span class="top-pick">Top Pick</span>' if is_top_pick else ""

    price = row.get("price_usd", 0)
    if pd.isna(price): price = 0

    total_reviews = int(row.get("total_reviews", 0))
    skin_reviews = int(skin_reviews)
    reviews_text = f"{total_reviews:,} reviews"
    if skin_reviews > 0:
        reviews_text += f" · {skin_reviews:,} from {skin_type} skin"

    recommend_pct = row.get("recommend_pct", 0)
    if pd.isna(recommend_pct): recommend_pct = 0

    category = row.get(cat_col, row.get("primary_category", ""))
    if pd.isna(category): category = ""

    # Show matched desired ingredients
    ingredient_html = ""
    if desired_ingredients and len(desired_ingredients) > 0:
        ingredients_text = str(row.get("ingredients", "")).lower()
        for ing in desired_ingredients:
            pattern = INGREDIENT_SEARCH_MAP.get(ing, ing.lower())
            patterns = pattern.split("|")
            if any(p in ingredients_text for p in patterns):
                ingredient_html += f'<span class="badge badge-ingredient">✓ Contains {ing}</span>'

    score_color = '#4A7C59' if suitability >= 85 else '#D4A853' if suitability >= 70 else '#C47D3B' if suitability >= 55 else '#8B3A3A'

    card_html = f"""
    <div class="product-card">
        <div style="display:flex;gap:20px;align-items:flex-start;">
            <div style="text-align:center;min-width:70px;">
                <div class="score-circle {score_class}">{suitability:.0f}</div>
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;margin-top:4px;font-weight:600;color:{score_color}">{score_label}</div>
            </div>
            <div style="flex:1;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div class="brand-name">{row.get('brand_name', '')} {top_pick}</div>
                        <div class="product-name">{row.get('product_name', '')}</div>
                    </div>
                    <div class="price-tag">${price:.0f}</div>
                </div>
                <div style="margin-bottom:8px;">
                    {render_stars(skin_rating)}
                    <span style="font-size:12px;color:#A0937D;margin-left:8px;">{reviews_text}</span>
                </div>
                <div style="margin-bottom:10px;">
                    <span class="badge badge-stat">👍 {recommend_pct*100:.0f}% recommend</span>
                    <span class="badge badge-stat" style="color:{irr_color}">{irr_icon} {irr_pct:.1f}% irritation for {skin_type} skin</span>
                    <span class="badge badge-stat">{category}</span>
                </div>
                {"<div style='margin-bottom:10px;'>" + ingredient_html + "</div>" if ingredient_html else ""}
    </div>
    """
    return card_html


# ==========================================
# MAIN APP
# ==========================================

st.markdown("""
<div style="text-align:center;padding:20px 0 10px;">
    <span style="font-size:11px;text-transform:uppercase;letter-spacing:4px;color:#A0937D;font-weight:600;">
        Powered by ML · Sephora Reviews Analysis · Team 4 · MISM 6212
    </span>
    <h1 style="font-size:36px;margin:8px 0;">Skincare Recommendation Engine</h1>
    <p style="font-size:15px;color:#6B5F4F;max-width:550px;margin:0 auto;line-height:1.6;">
        Personalized product recommendations based on analysis of 1M+ reviews,
        ingredient safety data, and skin-type-specific irritation modeling.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

data, cat_col = load_and_process_data()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("## 🧴 Your Preferences")

# Skin type dropdown
skin_type = st.sidebar.selectbox(
    "Your Skin Type",
    options=["combination", "dry", "normal", "oily"],
    index=0,
    format_func=lambda x: x.capitalize()
)

# Skin concerns
st.sidebar.markdown("### Skin Concerns")
selected_concerns = st.sidebar.multiselect(
    "What are your skin concerns?",
    options=list(SKIN_CONCERNS.keys()),
    default=[],
    format_func=lambda x: f"{SKIN_CONCERNS[x]['icon']} {x}"
)

# Desired ingredients
st.sidebar.markdown("### Desired Ingredients")
desired_ingredients = st.sidebar.multiselect(
    "Ingredients you want in your products",
    options=DESIRED_INGREDIENTS,
    default=[]
)

# Allergens to avoid
st.sidebar.markdown("### Avoid These Ingredients")
avoid_fragrance = st.sidebar.checkbox("🌸 Fragrance", value=False)
avoid_alcohol = st.sidebar.checkbox("🧪 Alcohol", value=False)
avoid_paraben = st.sidebar.checkbox("⚠ Paraben", value=False)
avoid_sulfate = st.sidebar.checkbox("🫧 Sulfate", value=False)
avoid_essential_oil = st.sidebar.checkbox("🌿 Essential Oils", value=False)
avoid_retinol = st.sidebar.checkbox("💊 Retinol", value=False)
avoid_formaldehyde = st.sidebar.checkbox("☣ Formaldehyde Releasers", value=False)
avoid_phthalate = st.sidebar.checkbox("🚫 Phthalates", value=False)
avoid_mineral_oil = st.sidebar.checkbox("🛢 Mineral Oil", value=False)
avoid_silicone = st.sidebar.checkbox("💧 Silicones", value=False)

# Category
categories = ["All"] + sorted(data[cat_col].dropna().unique().tolist())
category = st.sidebar.selectbox("Category", options=categories)

# Price range
price_option = st.sidebar.selectbox(
    "Price Range",
    options=["Any Price", "Under $25", "$25 – $50", "$50 – $100", "$100+"]
)

# Sort by
sort_option = st.sidebar.selectbox(
    "Sort By",
    options=["Best Suitability", "Highest Rating", "Lowest Irritation", "Most Reviewed", "Price: Low → High", "Price: High → Low"]
)

# ==========================================
# FILTER PRODUCTS
# ==========================================

filtered = data.copy()

# Allergen filters
allergen_filters = {
    "has_fragrance": avoid_fragrance,
    "has_alcohol": avoid_alcohol,
    "has_paraben": avoid_paraben,
    "has_sulfate": avoid_sulfate,
    "has_essential_oil": avoid_essential_oil,
    "has_retinol": avoid_retinol,
    "has_formaldehyde": avoid_formaldehyde,
    "has_phthalate": avoid_phthalate,
    "has_mineral_oil": avoid_mineral_oil,
    "has_silicone": avoid_silicone,
}

for col, should_avoid in allergen_filters.items():
    if should_avoid and col in filtered.columns:
        filtered = filtered[filtered[col] == 0]

# Category filter
if category != "All":
    filtered = filtered[filtered[cat_col] == category]

# Price filter
if "price_usd" in filtered.columns:
    if price_option == "Under $25":
        filtered = filtered[filtered["price_usd"] < 25]
    elif price_option == "$25 – $50":
        filtered = filtered[(filtered["price_usd"] >= 25) & (filtered["price_usd"] <= 50)]
    elif price_option == "$50 – $100":
        filtered = filtered[(filtered["price_usd"] >= 50) & (filtered["price_usd"] <= 100)]
    elif price_option == "$100+":
        filtered = filtered[filtered["price_usd"] >= 100]

# Skin concern filter - products must contain at least one relevant ingredient
if selected_concerns:
    concern_patterns = []
    for concern in selected_concerns:
        concern_patterns.extend(SKIN_CONCERNS[concern]["ingredients"])

    if "ingredients" in filtered.columns:
        pattern = "|".join(concern_patterns)
        filtered = filtered[filtered["ingredients"].str.contains(pattern, case=False, na=False)]

# Desired ingredient filter - products must contain ALL selected ingredients
if desired_ingredients:
    if "ingredients" in filtered.columns:
        for ing in desired_ingredients:
            pattern = INGREDIENT_SEARCH_MAP.get(ing, ing.lower())
            filtered = filtered[filtered["ingredients"].str.contains(pattern, case=False, na=False)]

# Compute suitability
filtered["suitability"] = filtered.apply(lambda row: compute_suitability(row, skin_type), axis=1)

# Sort
irr_col = f"irritation_{skin_type}"
rat_col = f"rating_{skin_type}"

if sort_option == "Best Suitability":
    filtered = filtered.sort_values("suitability", ascending=False)
elif sort_option == "Highest Rating":
    sort_col = rat_col if rat_col in filtered.columns else "rating"
    filtered = filtered.sort_values(sort_col, ascending=False)
elif sort_option == "Lowest Irritation":
    sort_col = irr_col if irr_col in filtered.columns else "irritation_flag"
    filtered = filtered.sort_values(sort_col, ascending=True)
elif sort_option == "Most Reviewed":
    filtered = filtered.sort_values("total_reviews", ascending=False)
elif sort_option == "Price: Low → High":
    filtered = filtered.sort_values("price_usd", ascending=True)
elif sort_option == "Price: High → Low":
    filtered = filtered.sort_values("price_usd", ascending=False)

# Reset index for clean pagination
filtered = filtered.reset_index(drop=True)

# Mark only top 3 as top picks
top_pick_ids = set()
if len(filtered) > 0:
    top_3 = filtered.head(3)
    top_pick_ids = set(top_3["product_id"].tolist())

# ==========================================
# METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("Products Found", f"{len(filtered)}")
col2.metric("Avg Suitability", f"{filtered['suitability'].mean():.0f}/100" if len(filtered) > 0 else "N/A")

avg_irr = filtered[irr_col].mean() if irr_col in filtered.columns and len(filtered) > 0 else filtered["irritation_flag"].mean() if len(filtered) > 0 else 0
if pd.isna(avg_irr): avg_irr = 0
col3.metric("Avg Irritation Rate", f"{avg_irr*100:.1f}%")
col4.metric("Skin Type", skin_type.capitalize())

st.markdown("")

# ==========================================
# DISPLAY PRODUCTS WITH PAGINATION
# ==========================================

if len(filtered) == 0:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#A0937D;font-size:15px;">
        <span style="font-size:40px;display:block;margin-bottom:12px;">🔍</span>
        No products match your current filters.<br>
        Try removing some allergen restrictions or changing the category.
    </div>
    """, unsafe_allow_html=True)
else:
    total_products = len(filtered)
    total_pages = math.ceil(total_products / PRODUCTS_PER_PAGE)

    # Initialize page state
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    # Reset page to 1 when filters change
    filter_key = f"{skin_type}_{category}_{price_option}_{sort_option}_{str(selected_concerns)}_{str(desired_ingredients)}_{str(allergen_filters)}"
    if "last_filter_key" not in st.session_state or st.session_state.last_filter_key != filter_key:
        st.session_state.current_page = 1
        st.session_state.last_filter_key = filter_key

    current_page = st.session_state.current_page

    # Clamp page
    if current_page > total_pages:
        current_page = total_pages
        st.session_state.current_page = current_page

    start_idx = (current_page - 1) * PRODUCTS_PER_PAGE
    end_idx = min(start_idx + PRODUCTS_PER_PAGE, total_products)

    # Show info
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;padding:0 4px;">
        <span style="font-size:14px;color:#6B5F4F;">
            Showing <strong style="color:#1A1A1A;">{start_idx + 1}–{end_idx}</strong> of <strong style="color:#1A1A1A;">{total_products}</strong> products
        </span>
        <span style="font-size:12px;color:#A0937D;">
            Scores personalized for <strong style="text-transform:capitalize;">{skin_type}</strong> skin · Page {current_page} of {total_pages}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Render product cards for current page
    page_data = filtered.iloc[start_idx:end_idx]
    for _, row in page_data.iterrows():
        is_top_pick = row["product_id"] in top_pick_ids
        st.markdown(
            render_product_card(row, skin_type, cat_col, is_top_pick, desired_ingredients),
            unsafe_allow_html=True
        )

    # Pagination controls
    st.markdown("")
    page_cols = st.columns([1, 1, 2, 1, 1])

    with page_cols[0]:
        if st.button("◀ First", disabled=(current_page == 1), use_container_width=True):
            st.session_state.current_page = 1
            st.rerun()

    with page_cols[1]:
        if st.button("← Previous", disabled=(current_page == 1), use_container_width=True):
            st.session_state.current_page = current_page - 1
            st.rerun()

    with page_cols[2]:
        # Page number selector
        page_num = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            index=current_page - 1,
            label_visibility="collapsed"
        )
        if page_num != current_page:
            st.session_state.current_page = page_num
            st.rerun()

    with page_cols[3]:
        if st.button("Next →", disabled=(current_page == total_pages), use_container_width=True):
            st.session_state.current_page = current_page + 1
            st.rerun()

    with page_cols[4]:
        if st.button("Last ▶", disabled=(current_page == total_pages), use_container_width=True):
            st.session_state.current_page = total_pages
            st.rerun()

    st.markdown(f"""
    <div class="page-info">Page {current_page} of {total_pages} · {PRODUCTS_PER_PAGE} products per page</div>
    """, unsafe_allow_html=True)

# ==========================================
# METHODOLOGY
# ==========================================

st.markdown("""
<div class="methodology-box">
    <h4 style="margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:1.5px;color:#A0937D;">
        How Suitability Scores Work
    </h4>
    <p style="margin:0;font-size:13px;color:#6B5F4F;line-height:1.7;">
        Each score (0–100) combines three factors weighted by importance:
        <strong>Irritation Rate</strong> for your skin type (40%),
        <strong>Average Rating</strong> from reviewers with your skin type (30%), and
        <strong>Recommendation Rate</strong> across all reviewers (30%).
        Scores are personalized — the same product scores differently for different skin types.
        Analysis based on 1M+ Sephora reviews using TF-IDF NLP and Random Forest classification.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("")
st.markdown("""
<div style="text-align:center;padding:20px;color:#A0937D;font-size:12px;">
    Team 4 · Harshmeet Kaur, Sai Athale, Shreya Pandey, Sakshi Patel · MISM 6212 · Northeastern University
</div>
""", unsafe_allow_html=True)
