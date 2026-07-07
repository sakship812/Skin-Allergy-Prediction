#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 22:08:48 2026

@author: sakshipatil
"""

import pandas as pd

products = pd.read_csv('/Users/sakshipatil/Desktop/MISM-6212_Data_Mining/PROJECT/product_info.csv')

reviews1 = pd.read_csv("/Users/sakshipatil/Desktop/MISM-6212_Data_Mining/PROJECT/reviews_0-250.csv")
reviews2 = pd.read_csv("/Users/sakshipatil/Desktop/MISM-6212_Data_Mining/PROJECT/reviews_250-500.csv")
reviews3 = pd.read_csv("/Users/sakshipatil/Desktop/MISM-6212_Data_Mining/PROJECT/reviews_500-750.csv")
reviews4 = pd.read_csv("/Users/sakshipatil/Desktop/MISM-6212_Data_Mining/PROJECT/reviews_750-1250.csv")
reviews5 = pd.read_csv("/Users/sakshipatil/Desktop/MISM-6212_Data_Mining/PROJECT/reviews_1250-end.csv")

ingredients_ref = pd.read_csv("ingredientsList.csv")

#cleaning
products.info()
# make column names neat
products.columns = products.columns.str.strip().str.lower()

# keep only useful columns
products = products[[
    "product_id",
    "product_name",
    "brand_name",
    "ingredients",
    "rating",
    "reviews",
    "price_usd",
    "primary_category"
]]

# remove duplicates
products = products.drop_duplicates()

# remove rows where product_id is missing
products = products.dropna(subset=["product_id"])

reviews1.columns = reviews1.columns.str.strip().str.lower()
reviews1 = reviews1.drop_duplicates()
reviews1 = reviews1.dropna(subset=["product_id"])

reviews2.columns = reviews2.columns.str.strip().str.lower()
reviews2 = reviews2.drop_duplicates()
reviews2 = reviews2.dropna(subset=["product_id"])

reviews3.columns = reviews3.columns.str.strip().str.lower()
reviews3 = reviews3.drop_duplicates()
reviews3 = reviews3.dropna(subset=["product_id"])

reviews4.columns = reviews4.columns.str.strip().str.lower()
reviews4 = reviews4.drop_duplicates()
reviews4 = reviews4.dropna(subset=["product_id"])

reviews5.columns = reviews5.columns.str.strip().str.lower()
reviews5 = reviews5.drop_duplicates()
reviews5 = reviews5.dropna(subset=["product_id"])

#Combine
reviews = pd.concat([reviews1, reviews2, reviews3, reviews4, reviews5], ignore_index=True)
print(reviews.columns)

reviews = reviews[[
    "product_id",
    "rating",
    "total_pos_feedback_count",
    "total_neg_feedback_count",
    "review_text",
    "skin_type",
    "product_name",
    "brand_name", 
    "is_recommended"
]]

reviews.info()
# remove duplicate rows
reviews = reviews.drop_duplicates()

# remove rows with missing review text
reviews = reviews.dropna(subset=["review_text"])

# fill missing skin_type with "Unknown"
reviews["skin_type"] = reviews["skin_type"].fillna("Unknown")

# convert is_recommended to integer
reviews["is_recommended"] = reviews["is_recommended"].fillna(0).astype(int)

#Cleaning ingredients
ingredients_ref.columns = ingredients_ref.columns.str.strip().str.lower()
ingredients_ref = ingredients_ref.drop_duplicates()

print(products.shape)
print(reviews.shape)
print(ingredients_ref.shape)

data = pd.merge(reviews, products, on="product_id", how="left")
data.info()

data["product_name"] = data["product_name_y"]
data["brand_name"] = data["brand_name_y"]
data["rating"] = data["rating_x"]

data = data.drop(columns=[
    "product_name_x",
    "product_name_y",
    "brand_name_x",
    "brand_name_y",
    "rating_x",
    "rating_y"
])


data = data[[
    "product_id",
    "product_name",
    "brand_name",
    "ingredients",
    "rating",
    "review_text",
    "skin_type",
    "price_usd",
    "primary_category"
]]

data["ingredients"] = data["ingredients"].str.lower()


#Detecting allergens
data["has_fragrance"] = data["ingredients"].str.contains("fragrance", na=False).astype(int)
data["has_alcohol"] = data["ingredients"].str.contains("alcohol", na=False).astype(int)
data["has_paraben"] = data["ingredients"].str.contains("paraben", na=False).astype(int)
data["has_sulfate"] = data["ingredients"].str.contains("sulfate", na=False).astype(int)

keywords = ["rash","burn","itch","irritation","allergy","breakout","redness"]

data["irritation_flag"] = data["review_text"].str.contains(
    "|".join(keywords),
    case=False,
    na=False
).astype(int)
data["irritation_flag"].value_counts()

data = pd.get_dummies(data, columns=["skin_type"])

# ==========================================
# SAVE TOP 100 ROWS OF MERGED DATASET
# ==========================================

# take first 100 rows
top_100_data = data.head(100)

# save to csv
top_100_data.to_csv("merged_top_100_rows.csv", index=False)

print("Top 100 rows saved successfully!")


#EDA
# ==========================================
# EDA for Skincare Recommendation Project
# Run this after cleaning + merging + creating:
# has_fragrance, has_alcohol, has_paraben,
# has_sulfate, irritation_flag
# ==========================================

# ==========================================
# EDA for Skincare Recommendation Project
# Updated for your current column names
# ==========================================

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. BASIC OVERVIEW OF THE DATA
# ==========================================

# show first few rows
print(data.head())

# show column names
print(data.columns)

# show dataset info
print(data.info())

# show summary statistics
print(data.describe())

# ==========================================
# 2. DISTRIBUTION OF RATINGS
# ==========================================

# histogram of ratings
plt.figure(figsize=(8,5))
plt.hist(data["rating"], bins=5, edgecolor="black")
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# ==========================================
# 3. DISTRIBUTION OF IRRITATION FLAG
# ==========================================

# count irritation mentions
print(data["irritation_flag"].value_counts())

# bar chart for irritation flag
plt.figure(figsize=(6,4))
data["irritation_flag"].value_counts().plot(kind="bar")
plt.title("Distribution of Irritation Mentions")
plt.xlabel("Irritation Flag (0 = No, 1 = Yes)")
plt.ylabel("Number of Reviews")
plt.show()

# ==========================================
# 4. PRESENCE OF RISK INGREDIENTS
# ==========================================

# fragrance
print(data["has_fragrance"].value_counts())

plt.figure(figsize=(6,4))
data["has_fragrance"].value_counts().plot(kind="bar")
plt.title("Presence of Fragrance in Products")
plt.xlabel("Has Fragrance (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# alcohol
print(data["has_alcohol"].value_counts())

plt.figure(figsize=(6,4))
data["has_alcohol"].value_counts().plot(kind="bar")
plt.title("Presence of Alcohol in Products")
plt.xlabel("Has Alcohol (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# paraben
print(data["has_paraben"].value_counts())

plt.figure(figsize=(6,4))
data["has_paraben"].value_counts().plot(kind="bar")
plt.title("Presence of Paraben in Products")
plt.xlabel("Has Paraben (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# sulfate
print(data["has_sulfate"].value_counts())

plt.figure(figsize=(6,4))
data["has_sulfate"].value_counts().plot(kind="bar")
plt.title("Presence of Sulfate in Products")
plt.xlabel("Has Sulfate (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# ==========================================
# 5. INGREDIENTS VS IRRITATION
# ==========================================

# average irritation for products with and without fragrance
print(data.groupby("has_fragrance")["irritation_flag"].mean())

plt.figure(figsize=(6,4))
sns.barplot(x="has_fragrance", y="irritation_flag", data=data)
plt.title("Fragrance vs Average Irritation")
plt.xlabel("Has Fragrance")
plt.ylabel("Average Irritation Flag")
plt.show()

# alcohol vs irritation
print(data.groupby("has_alcohol")["irritation_flag"].mean())

plt.figure(figsize=(6,4))
sns.barplot(x="has_alcohol", y="irritation_flag", data=data)
plt.title("Alcohol vs Average Irritation")
plt.xlabel("Has Alcohol")
plt.ylabel("Average Irritation Flag")
plt.show()

# paraben vs irritation
print(data.groupby("has_paraben")["irritation_flag"].mean())

plt.figure(figsize=(6,4))
sns.barplot(x="has_paraben", y="irritation_flag", data=data)
plt.title("Paraben vs Average Irritation")
plt.xlabel("Has Paraben")
plt.ylabel("Average Irritation Flag")
plt.show()

# sulfate vs irritation
print(data.groupby("has_sulfate")["irritation_flag"].mean())

plt.figure(figsize=(6,4))
sns.barplot(x="has_sulfate", y="irritation_flag", data=data)
plt.title("Sulfate vs Average Irritation")
plt.xlabel("Has Sulfate")
plt.ylabel("Average Irritation Flag")
plt.show()

# ==========================================
# 6. TOP BRANDS BY AVERAGE RATING
# ==========================================

# calculate average rating by brand
top_brands = data.groupby("brand_name")["rating"].mean().sort_values(ascending=False).head(10)

# print top brands
print(top_brands)

# plot top brands
plt.figure(figsize=(10,5))
top_brands.plot(kind="bar")
plt.title("Top 10 Brands by Average Rating")
plt.xlabel("Brand")
plt.ylabel("Average Rating")
plt.show()

# ==========================================
# 7. CORRELATION HEATMAP
# ==========================================

# numeric columns for correlation
corr_cols = [
    "rating",
    "has_fragrance",
    "has_alcohol",
    "has_paraben",
    "has_sulfate",
    "irritation_flag",
    "is_recommended",
    "price_usd",
    "reviews"
]

# correlation matrix
corr_matrix = data[corr_cols].corr()

# heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print(data.columns.tolist())

# ==========================================
# 8. SKIN TYPE ANALYSIS
# Since your skin_type column is now dummy variables,
# we will use the dummy columns directly
# ==========================================

# average irritation rate by skin type
skin_irritation = {
    "Unknown": data.loc[data["skin_type_Unknown"] == True, "irritation_flag"].mean(),
    "Combination": data.loc[data["skin_type_combination"] == True, "irritation_flag"].mean(),
    "Dry": data.loc[data["skin_type_dry"] == True, "irritation_flag"].mean(),
    "Normal": data.loc[data["skin_type_normal"] == True, "irritation_flag"].mean(),
    "Oily": data.loc[data["skin_type_oily"] == True, "irritation_flag"].mean()
}

skin_irritation = pd.Series(skin_irritation)
print(skin_irritation)

plt.figure(figsize=(8,5))
skin_irritation.plot(kind="bar")
plt.title("Average Irritation Rate by Skin Type")
plt.xlabel("Skin Type")
plt.ylabel("Average Irritation Rate")
plt.show()

# ==========================================
# 9. SKIN TYPE COUNTS
# ==========================================

skin_counts = {
    "Unknown": data["skin_type_Unknown"].sum(),
    "Combination": data["skin_type_combination"].sum(),
    "Dry": data["skin_type_dry"].sum(),
    "Normal": data["skin_type_normal"].sum(),
    "Oily": data["skin_type_oily"].sum()
}

skin_counts = pd.Series(skin_counts)
print(skin_counts)

plt.figure(figsize=(8,5))
skin_counts.plot(kind="bar")
plt.title("Distribution of Skin Types")
plt.xlabel("Skin Type")
plt.ylabel("Number of Reviews")
plt.show()

# ==========================================
# 10. AVERAGE RATING BY SKIN TYPE
# ==========================================

skin_rating = {
    "Unknown": data.loc[data["skin_type_Unknown"] == True, "rating"].mean(),
    "Combination": data.loc[data["skin_type_combination"] == True, "rating"].mean(),
    "Dry": data.loc[data["skin_type_dry"] == True, "rating"].mean(),
    "Normal": data.loc[data["skin_type_normal"] == True, "rating"].mean(),
    "Oily": data.loc[data["skin_type_oily"] == True, "rating"].mean()
}

skin_rating = pd.Series(skin_rating)
print(skin_rating)

plt.figure(figsize=(8,5))
skin_rating.plot(kind="bar")
plt.title("Average Rating by Skin Type")
plt.xlabel("Skin Type")
plt.ylabel("Average Rating")
plt.show()

# ==========================================
# 11. CATEGORY DISTRIBUTION
# ==========================================

print(data["primary_category"].value_counts())

plt.figure(figsize=(8,5))
data["primary_category"].value_counts().plot(kind="bar")
plt.title("Distribution of Primary Category")
plt.xlabel("Primary Category")
plt.ylabel("Count")
plt.show()

# ==========================================
# 12. OPTIONAL: SAVE DATA AFTER EDA
# ==========================================

data.to_csv("eda_ready_data.csv", index=False)

# ==========================================
# END OF EDA
# ==========================================

#Define X and Y for the Model

y = data["irritation_flag"]
X = data[[
    "rating",
    "has_fragrance",
    "has_alcohol",
    "has_paraben",
    "has_sulfate"
] + [col for col in data.columns if col.startswith("skin_type_")]]

#train-test split 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)

accuracy_score(y_test, predictions)
