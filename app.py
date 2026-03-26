import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Uber Clustering Dashboard",
    page_icon="🚖",
    layout="wide"
)

# -----------------------------
# Load Files
# -----------------------------
kmeans = joblib.load("uber_kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("clustered_data.csv")
inertia = joblib.load("inertia.pkl")

# -----------------------------
# Sidebar (Input Section)
# -----------------------------
st.sidebar.title("🚗 Ride Input Panel")

distance = st.sidebar.slider("Ride Distance (km)", 1, 50, 10)
fare = st.sidebar.slider("Booking Value (₹)", 50, 3000, 500)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

# Input Summary
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Current Input")
st.sidebar.write(f"Distance: {distance} km")
st.sidebar.write(f"Fare: ₹{fare}")
st.sidebar.write(f"Hour: {hour}")

# -----------------------------
# Feature Engineering
# -----------------------------
fare_per_km = fare / distance

input_data = pd.DataFrame({
    'Booking Value': [fare],
    'Ride Distance': [distance],
    'fare_per_km': [fare_per_km],
    'hour': [hour]
})

scaled_input = scaler.transform(input_data)

cluster = kmeans.predict(scaled_input)[0]

# -----------------------------
# Main Title
# -----------------------------
st.title("🚖 Uber Ride Clustering Dashboard")
st.markdown("---")

# -----------------------------
# Prediction Section
# -----------------------------
st.header("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.success(f"Predicted Cluster: {cluster}")

with col2:
    st.subheader("🧠 Interpretation")

    if cluster == 0:
        st.info("🚗 Long-distance morning ride")
    elif cluster == 1:
        st.info("🌆 Long-distance evening ride")
    elif cluster == 2:
        st.warning("💎 Premium / surge pricing ride")
    elif cluster == 3:
        st.info("🏙️ Short-distance ride")

# -----------------------------
# 💰 Smart Price Analysis (NEW)
# -----------------------------
st.markdown("---")
st.header("💰 Smart Price Analysis")

# Cluster-based average pricing
cluster_avg = df.groupby('cluster')['fare_per_km'].mean()

# Estimated price based on cluster
estimated_price = distance * cluster_avg[cluster]

col_price1, col_price2 = st.columns(2)

with col_price1:
    st.subheader("Estimated Fair Price")
    st.success(f"₹ {estimated_price:.2f}")

with col_price2:
    st.subheader("📊 Pricing Evaluation")

    if fare > estimated_price * 1.3:
        st.error("❌ You were likely OVERCHARGED")
    elif fare < estimated_price * 0.7:
        st.warning("⚠️ This ride seems UNDERPRICED")
    else:
        st.success("✅ Pricing looks FAIR")

# -----------------------------
# Visualization Section
# -----------------------------
st.markdown("---")
st.header("📍 Data Visualizations")

col3, col4 = st.columns(2)

# Scatter Plot
with col3:
    st.subheader("Cluster Scatter Plot")

    fig, ax = plt.subplots()
    sns.scatterplot(
        x='Ride Distance',
        y='Booking Value',
        hue='cluster',
        data=df,
        ax=ax,
        alpha=0.6
    )

    # 🔥 Show user input point
    ax.scatter(distance, fare, color='red', s=100, label='Your Input')
    ax.legend()

    st.pyplot(fig)

# Heatmap
with col4:
    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# Elbow Method (Full Width)
# -----------------------------
st.markdown("---")
st.subheader("📉 Elbow Method")

fig3, ax3 = plt.subplots()
ax3.plot(range(1, 10), inertia)
ax3.set_xlabel("Number of Clusters")
ax3.set_ylabel("Inertia")

st.pyplot(fig3)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built using Streamlit | Uber Ride Clustering Project 🚀")