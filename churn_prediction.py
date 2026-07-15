# Customer Churn Prediction
# Author: [Pratyusha Swain] | Tech: Python, Pandas, Scikit-Learn, Matplotlib, Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── 1. LOAD ──────────────────────────────────────────────────────────────────
# Read the CSV file into a DataFrame (rows = customers, columns = features)
df = pd.read_csv("Sample_customers.csv")

# ── 2. SPLIT ─────────────────────────────────────────────────────────────────
# X = input features the model learns from  |  y = the label we want to predict
X = df[["tenure_months", "monthly_logins", "support_tickets", "monthly_spend"]]
y = df["churned"]

# Split data: 80% for training, 20% for testing (random_state makes it reproducible)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. TRAIN ─────────────────────────────────────────────────────────────────
# Random Forest = an ensemble of many Decision Trees voting on the final answer
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)          # The model "learns" patterns from training data

# ── 4. EVALUATE ──────────────────────────────────────────────────────────────
# Ask the trained model to predict churn on data it has NEVER seen before
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ── 5. FEATURE IMPORTANCE ────────────────────────────────────────────────────
# Which features had the most influence on the model's decisions?
importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importances (higher = more influential):")
print(importance.sort_values(ascending=False))

# ── 6. VISUALIZE ─────────────────────────────────────────────────────────────
# Generate charts to explain the results to business stakeholders

# Chart 1: Confusion Matrix (Proves technical accuracy)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix: Actual vs. Predicted Churn")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.savefig("confusion_matrix.png", bbox_inches='tight') # <-- SAVES THE IMAGE
plt.close() # Closes the figure to keep memory clean

# Chart 2: Feature Importance (Shows business value)
sorted_importance = importance.sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=sorted_importance.values, y=sorted_importance.index,
            palette="viridis", hue=sorted_importance.index, legend=False)
plt.title("What Drives Customer Churn?")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.savefig("feature_importance.png", bbox_inches='tight') # <-- SAVES THE IMAGE
