
# 🧴 Skin Allergy Prediction & Recommendation Engine

An end-to-end **Machine Learning & NLP** application that predicts skincare irritation risk and recommends safer skincare products based on ingredient composition, customer reviews, skin type, and allergen preferences.

> Built using **1M+ Sephora reviews**, ingredient analysis, TF-IDF, and multiple machine learning models, then deployed as an interactive Streamlit web application.

## 🚀 Live Demo

**🌐 Website:** https://skin-allergy-prediction.streamlit.app/

## 📂 GitHub Repository

https://github.com/sakship812/Skin-Allergy-Prediction

---

## Project Highlights

- 🔬 Analyzed **1.09+ million Sephora reviews**
- 🧴 Processed **8,000+ skincare products**
- 🧠 Compared **6 Machine Learning models**
- 📈 Achieved **ROC-AUC of 0.9917**
- 💬 Used **TF-IDF + NLP** for review analysis
- ⚠️ Detects ingredient allergens and irritation risk
- 🎯 Personalized recommendations based on skin type & allergies
- 🌐 Fully deployed as a **Streamlit web application**

---

## Table of Contents

- Overview
- Features
- Project Architecture
- Project Structure
- Datasets
- Methodology
- Machine Learning Models
- Results
- Technologies Used
- Installation
- Running the Application
- Team
- Future Improvements
- Disclaimer

---

# Overview

Consumers often rely on marketing claims and scattered online reviews when selecting skincare products. Understanding ingredient labels and determining whether a product is suitable for a specific skin type can be difficult.

This project addresses that problem by combining structured product information with over one million customer reviews to build a machine learning recommendation engine capable of:

- Predicting irritation risk
- Identifying potentially harmful ingredients
- Filtering products based on allergies
- Providing personalized skincare recommendations
- Ranking products using a suitability score

The application combines ingredient analysis, natural language processing, and predictive machine learning into an interactive web dashboard.

---

# Features

- Personalized skincare recommendations
- Ingredient-based allergen filtering
- Skin type specific recommendations
- Product category filtering
- Price range filtering
- Ingredient search
- Safety/Suitability Score
- Product ratings and review counts
- Interactive Streamlit dashboard
- Fast search and filtering

---

# Project Architecture

```text
                Kaggle Datasets
                       │
        ┌──────────────┴──────────────┐
        │                             │
 Product Information           Sephora Reviews
        │                             │
        └──────────────┬──────────────┘
                       │
             Data Cleaning & Merging
                       │
             Feature Engineering
                       │
        Ingredient Analysis + NLP (TF-IDF)
                       │
        Machine Learning Model Training
                       │
          Suitability Score Generation
                       │
        Interactive Streamlit Dashboard
```

---

# Project Structure

```text
Skin-Allergy-Prediction/
│
├── app.py                         # Main Streamlit web application
├── README.md
├── requirements.txt
├── Skin_allergy.pptx
│
├── data/
│   └── dashboard_data.csv         # Processed deployment dataset
│
├── scripts/
│   ├── ML_proj.py                 # Data preprocessing & EDA
│   ├── Step_2_and_Step_3.py       # Feature engineering & ML models
│   └── dashboard_data_prep.py     # Dashboard dataset generation
│
└── frontend/
    └── skincare_dashboard.py      # Original dashboard implementation
```

---

# Datasets

This project combines three Kaggle datasets.

| Dataset | Description |
|----------|-------------|
| **Sephora Product Dataset** | Product metadata including ingredients, brand, category, price and ratings |
| **Sephora Reviews Dataset** | More than 1.09 million customer reviews with review text, ratings, recommendations and skin type |
| **INCI Ingredient Dataset** | Standardized cosmetic ingredient reference used to identify allergenic ingredients |

The review dataset consists of multiple CSV files that were merged into a single machine learning dataset during preprocessing.

---

# Methodology

## 1. Data Cleaning

- Removed duplicate records
- Standardized column names
- Handled missing values
- Combined review datasets
- Merged product and review information

## 2. Feature Engineering

Engineered features include:

- Ingredient allergen flags
- Toxicity score
- Allergen count
- Ingredient interaction features
- TF-IDF review vectors
- VADER sentiment analysis
- Skin type encoding
- Irritation labels extracted from review text

---

# Machine Learning Models

The following supervised learning algorithms were evaluated:

- Logistic Regression
- Linear SVM
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The final recommendation engine combines ingredient-based safety indicators with NLP features extracted from customer reviews.

---

# Results

| Metric | Value |
|--------|-------:|
| Reviews Analyzed | **1.09M+** |
| Products | **8,000+** |
| Models Compared | **6** |
| Best ROC-AUC | **0.9917** |
| Best F1 Score | **97.7%** |
| Dashboard Products | **2,122** |

---

# Technologies Used

| Category | Technologies |
|-----------|--------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| NLP | TF-IDF, VADER Sentiment Analysis |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn |

---

# Installation

Clone the repository

```bash
git clone https://github.com/sakship812/Skin-Allergy-Prediction.git
cd Skin-Allergy-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Or visit the deployed application:

**https://skin-allergy-prediction.streamlit.app/**

---

# Team

Developed by:

- **Sakshi Patil**
- **Harshmeet Kaur**
- **Sai Athale**
- **Shreya Pandey**

Graduate Project — MISM 6212: Data Mining & Machine Learning  
D'Amore-McKim School of Business, Northeastern University

---

# Future Improvements

- Deep Learning models
- Personalized user profiles
- Barcode scanner for skincare products
- Product image search
- Mobile application
- Explainable AI recommendations
- Cloud-hosted prediction API

---

# Disclaimer

This project was developed for educational and research purposes. Predictions are generated using historical customer reviews and ingredient analysis and are intended to assist users in making informed skincare decisions. They should not be considered medical advice or a substitute for consultation with a qualified dermatologist.

---

## ⭐ If you found this project interesting, feel free to star the repository!
