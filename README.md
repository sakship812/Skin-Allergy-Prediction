# 🧴 Skincare Product Safety Prediction & Recommendation Engine

A machine learning-powered recommendation system that helps users identify skincare products that are safer for their skin by combining **ingredient analysis**, **consumer review mining**, and **predictive modeling**.

> Developed as a graduate group project for **MISM 6212 – Data Mining & Machine Learning** at Northeastern University.

## Team

- **Sakshi Patil**
- **Harshmeet Kaur**
- **Sai Athale**
- **Shreya Pandey**

---

## Project Overview

Choosing skincare products can be difficult because ingredient labels are complex and reviews are subjective. Our goal was to build a recommendation engine that predicts irritation risk and recommends products based on a user's allergies and skin preferences.

The system analyzes ingredients, identifies common allergens, learns patterns from more than **1 million Sephora reviews**, and assigns a safety score to products.

Users can:
- Select ingredients they are allergic to or want to avoid.
- Filter products based on skin type and concerns.
- View an overall safety/suitability score.
- Compare products using ratings, prices, reviews, and predicted irritation risk.

---

## Problem Statement

Consumers typically depend on marketing claims or anecdotal reviews when selecting skincare products. There is no scalable way to evaluate whether a product is likely to cause irritation for a specific user.

Our project answers three questions:

1. Can ingredient composition predict skin irritation?
2. Can review text identify early signs of adverse reactions?
3. Can we recommend safer skincare products based on allergies and ingredient preferences?

---

## Datasets

The project combines **three Kaggle datasets** into a unified pipeline.

### 1. Sephora Product Dataset
Contains approximately **8,000 skincare products** including:
- Product name
- Brand
- Ingredients
- Category
- Price
- Ratings

### 2. Sephora Reviews Dataset
Contains over **1.09 million customer reviews** including:
- Review text
- Ratings
- Skin type
- Recommendation status
- Product IDs

The reviews were distributed across multiple CSV files:
- reviews_0-250.csv
- reviews_250-500.csv
- reviews_500-750.csv
- reviews_750-1250.csv
- reviews_1250-end.csv

### 3. INCI Ingredients Dataset (ingredientsList.csv)

A standardized ingredient reference containing cosmetic ingredient names used to:
- Normalize ingredient names
- Identify known allergenic ingredients
- Flag potentially harmful ingredients

---

## Data Pipeline

Our workflow consisted of:

1. Collecting all three datasets from Kaggle.
2. Cleaning duplicate and missing records.
3. Standardizing column names.
4. Merging product information with customer reviews.
5. Mapping ingredients against the INCI reference dataset.
6. Engineering allergen and toxicity features.
7. Creating irritation labels using review keywords.
8. Training multiple machine learning models.
9. Building an interactive Streamlit dashboard.

---

## Feature Engineering

We engineered several features including:

- Ingredient allergen flags
- Toxicity score
- Allergen count
- Ingredient interaction features
- TF-IDF review vectors
- VADER sentiment scores
- Skin-type encoding
- Irritation labels generated from review text

Common allergen families included:
- Fragrance
- Alcohol
- Sulfates
- Parabens
- Essential oils
- Retinol
- Formaldehyde releasers
- Phthalates
- Mineral oil
- Silicones

---

## Machine Learning Models

We evaluated multiple supervised learning algorithms including:

- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

TF-IDF features extracted from review text significantly improved model performance over ingredient-only features.

---

## Recommendation Engine

The final recommendation engine cross-references a user's selected allergies with product ingredients and combines this with predicted irritation risk to generate personalized recommendations.

For every product, the dashboard displays:

- Product name
- Brand
- Price
- Average rating
- Number of reviews
- Irritation probability
- Suitability score
- Ingredient warnings
- Skin-type-specific recommendations

Suitability scores are calculated using a weighted combination of:
- Safety (predicted irritation risk)
- Product quality (ratings)
- Consumer recommendation rate

---

## Dashboard

The Streamlit dashboard allows users to:

- Select skin type
- Choose skin concerns
- Filter products by category
- Filter by price range
- Avoid selected allergens
- Search for desired ingredients
- Browse personalized product recommendations

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- TF-IDF
- VADER Sentiment Analysis
- Matplotlib
- Seaborn
- Streamlit

---

## Repository Structure

```text
ML_proj.py                  # Data cleaning and preprocessing
Step_2_and_Step_3.py        # Feature engineering and ML models
dashboard_data_prep.py      # Dashboard data generation
skincare_dashboard.py       # Streamlit application
README.md
```

---

## Key Contributions

This project demonstrates how structured ingredient data and unstructured customer reviews can be combined to build an explainable AI recommendation system for skincare safety.

The solution helps users make more informed purchasing decisions by identifying potentially irritating products while recommending safer alternatives tailored to their preferences.

---

## Disclaimer

This project was developed for academic purposes. Predictions are based on historical product reviews and ingredient analysis and should not replace professional medical advice or dermatological consultation.
