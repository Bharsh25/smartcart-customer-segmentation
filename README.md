# 🛒 SmartCart Customer Segmentation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smartcart-customer-segmentation-4fz73cavxnty9rrxbjz9zf.streamlit.app/)

## 🚀 Live Demo
👉 **[Click here to try the app](https://smartcart-customer-segmentation-4fz73cavxnty9rrxbjz9zf.streamlit.app/)**

---

An intelligent **unsupervised machine learning** system that segments e-commerce customers into meaningful groups based on purchasing behaviour, engagement levels, and loyalty indicators — enabling personalised marketing and improved customer retention.

---

## 📌 Problem Statement

SmartCart is a growing e-commerce platform that was using **generic marketing strategies** for all 2240 customers, resulting in:
- Inefficient marketing spend
- Missed opportunities to retain high-value customers
- Delayed identification of churn-prone users

This project solves that by building a **customer segmentation system using KMeans clustering** to discover hidden patterns in customer behaviour.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| Records | 2240 customers |
| Features | 22 attributes |
| Source | SmartCart internal data |

**Feature Categories:**
- **Demographics** — Age, Education, Marital Status, Income, Children
- **Spending** — Wines, Fruits, Meat, Fish, Sweets, Gold Products
- **Purchase Frequency** — Web, Catalog, Store, Deal purchases
- **Engagement** — Website visits, Recency, Customer tenure
- **Feedback** — Complaints, Campaign response

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-purple)

- **Language:** Python 3.13
- **ML:** Scikit-learn (KMeans, PCA, StandardScaler)
- **Dashboard:** Streamlit
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Elbow Detection:** KneeLocator

---

## 🚀 Project Pipeline

```
Raw Data
   │
   ▼
Data Cleaning          ← Handle 24 missing Income values
   │
   ▼
Feature Engineering    ← Age, Tenure, Total Spending, Total Children
   │
   ▼
Encoding               ← Education (3 levels), Living_With (Alone/Partner)
   │
   ▼
Outlier Removal        ← Remove extreme Age & Income values
   │
   ▼
Standard Scaling       ← StandardScaler
   │
   ▼
PCA (3 components)     ← Dimensionality reduction
   │
   ▼
KMeans Clustering      ← k=4 (Elbow + Silhouette)
   │
   ▼
Cluster Profiling      ← Label & interpret segments
   │
   ▼
Streamlit Dashboard    ← Interactive insights
```

---

## 📈 Finding Optimal K

Used two methods to find the best number of clusters:

- **Elbow Method** with KneeLocator → detected k=4
- **Silhouette Score** → validated k=4 as meaningful

---

## 👥 Customer Segments

| Cluster | Segment | Income | Spending | Living | Key Trait |
|---------|---------|--------|----------|--------|-----------|
| 0 | 🟡 Mid-Income Families | ~$39k | ~$222 | Partner | Has kids, moderate spend |
| 1 | 💎 High-Value Couples | ~$73k | ~$1237 | Partner | High spend, loyal |
| 2 | 🔴 Budget Solo Shoppers | ~$37k | ~$166 | Alone | Low spend, deal-driven |
| 3 | 🏆 Premium Solo Spenders | ~$71k | ~$1190 | Alone | High spend, catalog buyer |

---

## 💡 Marketing Strategies

| Segment | Strategy |
|---------|----------|
| 🟡 Mid-Income Families | Family bundles, back-to-school campaigns, loyalty points |
| 💎 High-Value Couples | VIP memberships, premium launches, exclusive early access |
| 🔴 Budget Solo Shoppers | Flash sales, discount alerts, referral bonuses |
| 🏆 Premium Solo Spenders | Personalised recommendations, catalog campaigns, concierge service |

---

## 🖥️ Streamlit Dashboard Features

- 📋 Dataset overview with key metrics
- 📊 EDA charts — Income, Spending, Age distributions
- 🎯 PCA cluster scatter plot
- 🔥 Cluster profile heatmap
- 📋 Cluster summary table
- 💡 Marketing strategy per segment
- ⬇️ Download cluster assignments as CSV

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/smartcart-customer-segmentation.git
cd smartcart-customer-segmentation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate model files
Run the Jupyter notebook first to train and save the models:
```bash
jupyter notebook smartcart.ipynb
```
This will create `kmeans_model.pkl`, `scaler.pkl`, and `pca.pkl`.

### 4. Run the dashboard
```bash
streamlit run app.py
```

### 5. Upload CSV
Upload `smartcart_customers.csv` in the dashboard to see all insights.

---

## 📁 Project Structure

```
smartcart-customer-segmentation/
│
├── smartcart.ipynb          # Main notebook (EDA + ML pipeline)
├── app.py                   # Streamlit dashboard
├── smartcart_customers.csv  # Dataset
├── requirements.txt         # Dependencies
├── README.md                # This file
│
├── kmeans_model.pkl         # (generated after running notebook)
├── scaler.pkl               # (generated after running notebook)
└── pca.pkl                  # (generated after running notebook)
```

---

## 📦 Requirements

Create a `requirements.txt` with:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
kneed
```

---

## 📬 Contact

**Harsh Bhendarkar**
- 📧 bhendarkarharsh92@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
- 🐙 [GitHub](https://github.com/YourUsername)

---

## ⭐ If you found this project helpful, give it a star!
