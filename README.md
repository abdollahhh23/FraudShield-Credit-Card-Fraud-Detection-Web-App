#  Real Time Fraud Prevention — Credit Card Fraud Detection Web App

A clean, locally-hosted Flask web app that lets you simulate real-time credit card transaction screening using your trained Random Forest model.
<img width="1918" height="896" alt="image" src="https://github.com/user-attachments/assets/cd88a7c3-b1e8-4de0-88b0-7a16156b5f84" />
<img width="1731" height="896" alt="image" src="https://github.com/user-attachments/assets/e84d131c-b622-4c60-949f-d4d5d7828712" />

---

##  Project Structure

```
fraud_app/
├── app.py                  ← Flask backend
├── model.pkl               ← Your trained model (copy here)
├── requirements.txt
└── templates/
    └── index.html          ← Frontend UI
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

>  Your `model.pkl` was trained with scikit-learn 1.2.2 — use **exactly** that version to avoid compatibility issues.

### 2. Place your model

Copy `model.pkl` into the `fraud_app/` folder (same directory as `app.py`).

### 3. Run the app

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

---

## 🎛️ How It Works

The model was trained on 30 features: `V1–V28` (PCA-reduced) + `Normalized Amount`. Since real users can't enter 28 PCA values, the app maps **4 human-readable inputs** to realistic feature patterns:

| Input | Description |
|---|---|
| **Amount ($)** | Transaction value, normalized to match training distribution |
| **Merchant Type** | Category (grocery, ATM, luxury, etc.) — maps to PCA patterns |
| **Location** | Local vs. International — foreign transactions use high-risk vectors |
| **Time of Day** | Daytime vs. Night-time — night transactions use riskier feature patterns |

These combinations represent real behavioral patterns observed in the original Kaggle dataset (e.g., ATM withdrawals at night are a strong fraud signal).

---

##  Model Info

- **Algorithm**: Random Forest Classifier (SMOTE-balanced)
- **Dataset**: [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,315 transactions
- **Accuracy**: 99.98% | **Precision**: 99.97% | **Recall**: 1.0
