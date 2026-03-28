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
price_model = joblib.load("price_model.pkl")

df = pd.read_csv("clustered_data.csv")
inertia = joblib.load("inertia.pkl")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🚗 Ride Input Panel")

distance = st.sidebar.slider("Ride Distance (km)", 1, 50, 10)
fare = st.sidebar.slider("Booking Value (₹)", 50, 3000, 500)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Current Input")
st.sidebar.write(f"Distance: {distance} km")
st.sidebar.write(f"Fare: ₹{fare}")
st.sidebar.write(f"Hour: {hour}")

# -----------------------------
# Feature Engineering
# -----------------------------
fare_per_km = fare / distance

# Time category (same as training)
def get_time_category(h):
    if h < 6:
        return 0
    elif h < 12:
        return 1
    elif h < 18:
        return 2
    else:
        return 3

time_category = get_time_category(hour)

# -----------------------------
# Clustering
# -----------------------------
input_cluster = pd.DataFrame({
    'Booking Value': [fare],
    'Ride Distance': [distance],
    'fare_per_km': [fare_per_km],
    'hour': [hour]
})

scaled_input = scaler.transform(input_cluster)
cluster = kmeans.predict(scaled_input)[0]

# -----------------------------
# ML Price Prediction
# -----------------------------
predicted_price = price_model.predict(
    [[distance, time_category, fare_per_km]]
)[0]

# -------- FIX: Clamp values --------
min_price = distance * 8
max_price = distance * 25

predicted_price = max(predicted_price, min_price)
predicted_price = min(predicted_price, max_price)

# -----------------------------
# Main UI
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

    cluster_info = df.groupby('cluster').mean()
    avg_distance = cluster_info.loc[cluster, 'Ride Distance']

    # Time label
    if hour < 12:
        time_label = "Morning"
    elif hour < 18:
        time_label = "Afternoon"
    elif hour < 22:
        time_label = "Evening"
    else:
        time_label = "Night"

    # Distance label
    if avg_distance > 30:
        dist_label = "Long-distance"
    elif avg_distance > 15:
        dist_label = "Medium-distance"
    else:
        dist_label = "Short-distance"

    st.info(f"🚗 {dist_label} {time_label} ride")

# -----------------------------
# Smart Price Analysis
# -----------------------------
st.markdown("---")
st.header("💰 Smart Price Analysis")

col_price1, col_price2 = st.columns(2)

with col_price1:
    st.subheader("Estimated Fair Price (ML Model)")
    st.success(f"₹ {predicted_price:.2f}")

with col_price2:
    st.subheader("📊 Pricing Evaluation")

    ratio = fare / predicted_price

    if ratio > 1.5:
        st.error("❌ Significantly OVERCHARGED")
    elif ratio > 1.2:
        st.warning("⚠️ Slightly OVERPRICED")
    elif ratio < 0.6:
        st.warning("⚠️ Extremely cheap (possible anomaly)")
    elif ratio < 0.8:
        st.info("ℹ️ UNDERPRICED")
    else:
        st.success("✅ Pricing looks FAIR")

# -----------------------------
# Visualization
# -----------------------------
st.markdown("---")
st.header("📍 Data Visualizations")

col3, col4 = st.columns(2)

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

    ax.scatter(distance, fare, color='red', s=100, label='Your Input')
    ax.legend()

    st.pyplot(fig)

with col4:
    st.subheader("Correlation Heatmap")

    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax2)
    st.pyplot(fig2)

# -----------------------------
# Elbow Method
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
st.caption("Built using Streamlit | ML Powered Uber Pricing 🚀")