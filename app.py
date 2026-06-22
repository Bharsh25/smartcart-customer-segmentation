import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="SmartCart Segmentation", layout="wide")
st.title("🛒 SmartCart Customer Segmentation Dashboard")

# ── Load saved models ──────────────────────────────────────────
model  = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl",       "rb"))
pca    = pickle.load(open("pca.pkl",          "rb"))

# ── Upload CSV ─────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload Customer CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {df.shape[0]} customers, {df.shape[1]} columns")

    # ── SECTION 1: Dataset Overview ───────────────────────────
    st.markdown("---")
    st.header("📋 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Total Features",  df.shape[1])
    col3.metric("Avg Income",      f"${df['Income'].mean():,.0f}")
    st.dataframe(df.head())

    # ── PREPROCESSING (exact same steps as notebook) ──────────

    # 1. Missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # 2. Feature Engineering
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    ref = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (ref - df["Dt_Customer"]).dt.days
    df["Total_Spending"] = (df["MntWines"] + df["MntFruits"] +
                            df["MntMeatProducts"] + df["MntFishProducts"] +
                            df["MntSweetProducts"] + df["MntGoldProds"])
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # 3. Encode Education
    df["Education"] = df["Education"].replace({
        "Basic":      "Undergraduate",
        "2n Cycle":   "Undergraduate",
        "Graduation": "Graduate",
        "Master":     "Postgraduate",
        "PhD":        "Postgraduate"
    })

    # 4. Encode Marital Status → Living_With
    df["Living_With"] = df["Marital_Status"].replace({
        "Married":  "Partner",
        "Together": "Partner",
        "Single":   "Alone",
        "Divorced": "Alone",
        "Widow":    "Alone",
        "Absurd":   "Alone",
        "YOLO":     "Alone"
    })

    # 5. Drop same columns as notebook
    cols_to_drop = [
        "ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome",
        "Dt_Customer", "MntWines", "MntFruits", "MntMeatProducts",
        "MntFishProducts", "MntSweetProducts", "MntGoldProds"
    ]
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")

    # 6. One-hot encode Education and Living_With
    df_clean = pd.get_dummies(df_clean, columns=["Education", "Living_With"])

    # 7. Align columns exactly to what scaler expects
    feature_cols = scaler.feature_names_in_
    for col in feature_cols:
        if col not in df_clean.columns:
            df_clean[col] = 0          # add any missing dummy columns as 0
    df_clean = df_clean[feature_cols]  # reorder to match scaler exactly

    # 8. Scale + Predict clusters
    X_scaled = scaler.transform(df_clean)
    X_pca = pca.transform(X_scaled)   
    df["Cluster"] = model.predict(X_pca)    
    
    cluster_labels = {
        0: "🟡 Mid-Income Families",
        1: "💎 High-Value Couples",
        2: "🔴 Budget Solo Shoppers",
        3: "🏆 Premium Solo Spenders"
    }
    df["Segment"] = df["Cluster"].map(cluster_labels)

    # ── SECTION 2: EDA Insights ───────────────────────────────
    st.markdown("---")
    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Income Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Income"], bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel("Income")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Total Spending Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Total_Spending"], bins=30, color="coral", edgecolor="white")
        ax.set_xlabel("Total Spending")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Age"], bins=20, color="mediumseagreen", edgecolor="white")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col4:
        st.subheader("Income vs Total Spending")
        fig, ax = plt.subplots()
        ax.scatter(df["Income"], df["Total_Spending"], alpha=0.4, color="mediumpurple")
        ax.set_xlabel("Income")
        ax.set_ylabel("Total Spending")
        st.pyplot(fig)

    # ── SECTION 3: Cluster Visualizations ─────────────────────
    st.markdown("---")
    st.header("🎯 Cluster Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Segments (PCA View)")
        X_pca = pca.transform(X_scaled)
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],X_pca[:, 2],
                             c=df["Cluster"], cmap="tab10", alpha=0.6)
        plt.colorbar(scatter, ax=ax)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        st.pyplot(fig)

    with col2:
        st.subheader("Cluster Size Distribution")
        fig, ax = plt.subplots()
        df["Segment"].value_counts().plot(kind="bar", ax=ax,
                                          color="steelblue", edgecolor="white")
        ax.set_ylabel("Count")
        plt.xticks(rotation=15, ha="right")
        st.pyplot(fig)

    # ── SECTION 4: Cluster Profile Heatmap ────────────────────
    st.markdown("---")
    st.header("🔥 Cluster Profile Heatmap")

    profile = df.groupby("Cluster")[
        ["Income", "Total_Spending", "Recency",
         "Age", "Total_Children", "Customer_Tenure_Days"]
    ].mean().round(2)
    profile.index = [cluster_labels[i] for i in profile.index]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(profile, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    plt.xticks(rotation=15, ha="right")
    st.pyplot(fig)

    # ── SECTION 5: Cluster Summary Table ──────────────────────
    st.markdown("---")
    st.header("📋 Cluster Summary Table")
    st.dataframe(profile)

    # ── SECTION 6: Marketing Strategies ───────────────────────
    st.markdown("---")
    st.header("💡 Marketing Strategy per Segment")

    strategies = {
        "🟡 Mid-Income Families":   "Family bundle offers, back-to-school campaigns, loyalty points",
        "💎 High-Value Couples":    "Premium product launches, VIP memberships, exclusive early access",
        "🔴 Budget Solo Shoppers":  "Discount alerts, flash sales, referral bonuses",
        "🏆 Premium Solo Spenders": "Personalized recommendations, catalog campaigns, concierge service",
    }
    for segment, strategy in strategies.items():
        st.info(f"**{segment}** → {strategy}")

    # ── SECTION 7: Download Results ───────────────────────────
    st.markdown("---")
    st.header("⬇️ Download Results")
    csv = df[["Cluster", "Segment"]].to_csv(index=False)
    st.download_button(
        label="📥 Download Cluster Assignments",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )