# ==========================================
# STEP 2 & 3: TF-IDF FEATURES + MULTI-MODEL COMPARISON
# Run this AFTER your train-test split
# Uses undersampling as the balancing approach
# ==========================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.utils import resample

# ==========================================
# 2A. PREPARE REVIEW TEXT FOR TF-IDF
# ==========================================
 
# Get train/test indices from your original split
# We need the review_text aligned with X_train and X_test
 
train_text = data.loc[X_train.index, "review_text"].fillna("")
test_text = data.loc[X_test.index, "review_text"].fillna("")
 
print("Train text shape:", train_text.shape)
print("Test text shape:", test_text.shape)

# ==========================================
# 2B. FIT TF-IDF VECTORIZER
# ==========================================
 
# max_features=5000 keeps it manageable for local machine
# ngram_range=(1,2) captures phrases like "broke out", "dried out"
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95
)
 
# Fit on training text ONLY, transform both
train_tfidf = tfidf.fit_transform(train_text)
test_tfidf = tfidf.transform(test_text)
 
print(f"\nTF-IDF matrix shape (train): {train_tfidf.shape}")
print(f"TF-IDF matrix shape (test):  {test_tfidf.shape}")

# ==========================================
# 2C. COMBINE TF-IDF WITH EXISTING FEATURES
# ==========================================
 
from scipy.sparse import csr_matrix

# Convert to float to fix the dtype issue
X_train_sparse = csr_matrix(X_train.values.astype(float))
X_test_sparse = csr_matrix(X_test.values.astype(float))

# Stack: [existing features | TF-IDF features]
X_train_combined = hstack([X_train_sparse, train_tfidf])
X_test_combined = hstack([X_test_sparse, test_tfidf])

print(f"\nCombined feature matrix (train): {X_train_combined.shape}")
print(f"Combined feature matrix (test):  {X_test_combined.shape}")

# ==========================================
# 2D. APPLY UNDERSAMPLING ON COMBINED DATA
# ==========================================
 
# Convert to arrays for undersampling
# Since data is sparse, we work with indices
 
np.random.seed(42)
 
majority_idx = np.where(y_train == 0)[0]
minority_idx = np.where(y_train == 1)[0]
 
# Randomly sample majority to match minority count
majority_downsampled_idx = np.random.choice(
    majority_idx,
    size=len(minority_idx),
    replace=False
)
 
# Combine indices
balanced_idx = np.concatenate([majority_downsampled_idx, minority_idx])
np.random.shuffle(balanced_idx)
 
X_train_bal = X_train_combined[balanced_idx]
y_train_bal = y_train.iloc[balanced_idx].values
 
print(f"\nBalanced training set: {X_train_bal.shape}")
print(f"Class distribution: 0={sum(y_train_bal==0)}, 1={sum(y_train_bal==1)}")

# ==========================================
# 3A. MODEL 1: LOGISTIC REGRESSION
# ==========================================
 
print("\n" + "=" * 50)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 50)
 
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver="saga",    # works well with sparse + large data
    n_jobs=-1
)
lr_model.fit(X_train_bal, y_train_bal)
pred_lr = lr_model.predict(X_test_combined)
 
print("\nClassification Report:")
print(classification_report(y_test, pred_lr, target_names=["No Irritation", "Irritation"]))
print(f"ROC-AUC: {roc_auc_score(y_test, lr_model.predict_proba(X_test_combined)[:, 1]):.4f}")
 
# ==========================================
# 3B. MODEL 2: RANDOM FOREST (with TF-IDF)
# ==========================================
 
print("\n" + "=" * 50)
print("MODEL 2: RANDOM FOREST (with TF-IDF)")
print("=" * 50)
 
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_bal, y_train_bal)
pred_rf = rf_model.predict(X_test_combined)
 
print("\nClassification Report:")
print(classification_report(y_test, pred_rf, target_names=["No Irritation", "Irritation"]))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_model.predict_proba(X_test_combined)[:, 1]):.4f}")

# ==========================================
# 3C. MODEL 3: GRADIENT BOOSTING
# ==========================================
# NOTE: GradientBoosting is slower than RF.
# If it takes too long, reduce n_estimators to 50
 
print("\n" + "=" * 50)
print("MODEL 3: GRADIENT BOOSTING")
print("=" * 50)
 
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    subsample=0.8
)
gb_model.fit(X_train_bal, y_train_bal.ravel())
pred_gb = gb_model.predict(X_test_combined)
 
print("\nClassification Report:")
print(classification_report(y_test, pred_gb, target_names=["No Irritation", "Irritation"]))
print(f"ROC-AUC: {roc_auc_score(y_test, gb_model.predict_proba(X_test_combined)[:, 1]):.4f}")

# ==========================================
# 4. COMPARISON VISUALIZATIONS
# ==========================================
 
# ---- 4A. ROC CURVES ----
 
fig, ax = plt.subplots(figsize=(8, 6))
 
for model, name, color in [
    (lr_model, "Logistic Regression", "#3498db"),
    (rf_model, "Random Forest", "#e67e22"),
    (gb_model, "Gradient Boosting", "#2ecc71")
]:
    proba = model.predict_proba(X_test_combined)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2)
 
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison - Models with TF-IDF Features")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_tfidf_models.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- 4B. CONFUSION MATRICES ----
 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
 
for ax, preds, name in [
    (axes[0], pred_lr, "Logistic Regression"),
    (axes[1], pred_rf, "Random Forest"),
    (axes[2], pred_gb, "Gradient Boosting")
]:
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds,
        display_labels=["No Irritation", "Irritation"],
        ax=ax, cmap="Blues"
    )
    ax.set_title(name)
 
plt.tight_layout()
plt.savefig("confusion_matrix_tfidf_models.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- 4C. SUMMARY TABLE ----
 
results = []
for name, preds, model in [
    ("Logistic Regression", pred_lr, lr_model),
    ("Random Forest", pred_rf, rf_model),
    ("Gradient Boosting", pred_gb, gb_model)
]:
    proba = model.predict_proba(X_test_combined)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": round((preds == y_test).mean(), 4),
        "Precision (Irritation)": round(precision_score(y_test, preds, zero_division=0), 4),
        "Recall (Irritation)": round(recall_score(y_test, preds, zero_division=0), 4),
        "F1 (Irritation)": round(f1_score(y_test, preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, proba), 4)
    })
 
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY (with TF-IDF features)")
print("=" * 60)
print(results_df.to_string(index=False))

# ---- 4D. BEFORE vs AFTER TF-IDF COMPARISON ----
 
print("\n" + "=" * 60)
print("IMPACT OF TF-IDF: BEFORE vs AFTER")
print("=" * 60)
 
# Best model without TF-IDF (your undersampling RF from earlier)
print("\nBest WITHOUT TF-IDF (RF + Undersampling):")
print(f"  ROC-AUC: 0.5695")
print(f"  F1 (Irritation): 0.2314")
print(f"  Recall (Irritation): 0.6312")
 
# Best model with TF-IDF
best_idx = results_df["ROC-AUC"].idxmax()
best = results_df.iloc[best_idx]
print(f"\nBest WITH TF-IDF ({best['Model']}):")
print(f"  ROC-AUC: {best['ROC-AUC']}")
print(f"  F1 (Irritation): {best['F1 (Irritation)']}")
print(f"  Recall (Irritation): {best['Recall (Irritation)']}")

# ==========================================
# 5. TOP TF-IDF FEATURES (FEATURE IMPORTANCE)
# ==========================================
# This answers Business Question 3:
# "Which words in reviews are most associated with irritation?"
 
# Get feature names
existing_features = list(X_train.columns)
tfidf_features = tfidf.get_feature_names_out().tolist()
all_features = existing_features + tfidf_features
 
# For Logistic Regression: coefficients show feature importance
lr_coefs = lr_model.coef_[0]
 
# Top 20 words that INCREASE irritation prediction
top_positive = pd.Series(lr_coefs, index=all_features).sort_values(ascending=False).head(20)
 
# Top 20 words that DECREASE irritation prediction
top_negative = pd.Series(lr_coefs, index=all_features).sort_values(ascending=True).head(20)
 
print("\n" + "=" * 60)
print("TOP 20 WORDS ASSOCIATED WITH IRRITATION")
print("=" * 60)
print(top_positive)
 
print("\n" + "=" * 60)
print("TOP 20 WORDS ASSOCIATED WITH NO IRRITATION")
print("=" * 60)
print(top_negative)

# Visualize top features
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
 
top_positive.sort_values().plot(kind="barh", ax=axes[0], color="#e74c3c")
axes[0].set_title("Top 20 Words → Irritation")
axes[0].set_xlabel("Coefficient")
 
top_negative.sort_values(ascending=False).plot(kind="barh", ax=axes[1], color="#2ecc71")
axes[1].set_title("Top 20 Words → No Irritation")
axes[1].set_xlabel("Coefficient")
 
plt.tight_layout()
plt.savefig("top_tfidf_features.png", dpi=150, bbox_inches="tight")
plt.show()
 





























