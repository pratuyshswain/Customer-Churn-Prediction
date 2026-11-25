# 📉 Customer Churn Prediction.

A machine learning project that predicts which customers are likely to cancel their subscription, helping a business take proactive retention action before it's too late.

---

## 🎯 The Business Problem

Customer churn — when a subscriber cancels their plan — is one of the most costly problems for any subscription-based business. Acquiring a new customer can cost **5–7× more** than retaining an existing one.

By predicting churn **before** it happens, a business can:
- Offer targeted discounts or re-engagement campaigns
- Identify customers who need better support
- Reduce monthly revenue loss (MRR)

This project builds a classification model to answer: **"Will this customer churn next month?"**

---

## 🗂️ Project Structure

```
├── churn_prediction.py          # Main ML pipeline script
├── Sample_customers.csv         # Synthetic customer dataset
├── confusion_matrix.png         # Generated accuracy visual
├── feature_importance.png       # Generated business insight visual
└── README.md                    # You are here
```

---

## 🔬 Dataset Features

| Feature           | Description                              |
|-------------------|------------------------------------------|
| `tenure_months`   | How long the customer has been subscribed |
| `monthly_logins`  | Number of app logins in the last month   |
| `support_tickets` | Support complaints raised last month     |
| `monthly_spend`   | Amount paid per month (USD)              |
| `churned`         | **Target** — 1 = churned, 0 = retained   |

> **Key Insight:** Customers with low tenure and high support tickets churn most frequently — exactly what the model learns to detect.

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python 3.x` | Core language |
| `Pandas` | Data loading and manipulation |
| `Scikit-Learn` | ML model (Random Forest), train/test split, evaluation metrics |
| `Matplotlib` & `Seaborn` | Data Visualization |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/pratuyshswain/Customer-Churn-Prediction.git

cd Customer-Churn-Prediction
```

**2. Install dependencies**
```bash
python -m pip install pandas scikit-learn matplotlib seaborn
```

**3. Run the prediction pipeline**
```bash
python churn_prediction.py
```

**Illustrative output (not a performance guarantee):**
```
Accuracy: 100.00%

Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00         2
           1       1.00      1.00      1.00         1

Feature Importances (higher = more influential):
tenure_months      0.45
support_tickets    0.30
monthly_logins     0.15
monthly_spend      0.10
```

---

## 🧠 Model: Random Forest Classifier

A **Random Forest** trains hundreds of Decision Trees on random subsets of the data and combines their votes for a final prediction. It is:
- ✅ Resistant to overfitting
- ✅ Works well on small datasets
- ✅ Provides feature importance scores out of the box

---

## 📌 Key Takeaways

- **Feature Importance** revealed that `tenure_months` and `support_tickets` are the strongest predictors of churn — aligning with real-world business intuition.
- The 15-row synthetic dataset illustrates the workflow only; the small test set cannot establish real-world generalization.

---

## 👤 Author

**[Pratyusha Swain]**  
Aspiring Data Scientist | [LinkedIn](https://www.linkedin.com/in/pratyusha-swain-221407315/) | [GitHub](https://github.com/pratuyshswain)

---

*This project was built as part of a data science portfolio to demonstrate end-to-end ML pipeline skills.*

## License

See [LICENSE](LICENSE) (MIT).
