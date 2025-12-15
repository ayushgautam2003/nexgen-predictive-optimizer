# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="NexGen – Predictive Delivery Optimizer", layout="wide")

st.title("NexGen Logistics – Predictive Delivery Optimizer")

# Load data
orders = pd.read_csv("orders.csv")
delivery = pd.read_csv("delivery_performance.csv")
routes = pd.read_csv("routes_distance.csv")
costs = pd.read_csv("cost_breakdown.csv")

df = orders.merge(delivery, on="Order_ID", how="left") \
           .merge(routes, on="Order_ID", how="left") \
           .merge(costs, on="Order_ID", how="left")

# Target
df["delayed"] = (df["Actual_Delivery_Days"] > df["Promised_Delivery_Days"]).astype(int)
weather_mapping = {
    "Clear": 0,
    "Sunny": 0,
    "Cloudy": 1,
    "Rain": 2,
    "Fog": 3,
    "Storm": 4
}

df["Weather_Impact"] = (
    df["Weather_Impact"]
    .astype(str)
    .str.strip()
    .map(weather_mapping)
    .fillna(1)  # default medium impact
)

# Missing values
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df.fillna("Unknown", inplace=True)

features = [
    "Distance_KM",
    "Traffic_Delay_Minutes",
    "Weather_Impact",
    "Delivery_Cost_INR",
    "Fuel_Cost",
    "Labor_Cost"
]

X = df[features].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

y = df["delayed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# Sidebar
priority = st.sidebar.multiselect(
    "Select Priority",
    df["Priority"].unique(),
    df["Priority"].unique()
)

filtered = df[df["Priority"].isin(priority)]
filtered["delay_risk"] = model.predict_proba(filtered[features])[:, 1]
filtered["risk_level"] = pd.cut(
    filtered["delay_risk"],
    bins=[0, 0.6, 0.8, 1],
    labels=["Low", "Medium", "High"]
)

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Orders", len(filtered))
c2.metric("Delay Rate", f"{filtered['delayed'].mean()*100:.1f}%")
c3.metric("Avg Delivery Cost", f"₹{filtered['Delivery_Cost_INR'].mean():.0f}")
c4.metric("Model Accuracy", f"{accuracy*100:.1f}%")

# Charts
st.subheader("Delay Rate by Priority")
fig, ax = plt.subplots()
filtered.groupby("Priority")["delayed"].mean().plot(kind="bar", ax=ax)
st.pyplot(fig)
st.subheader("Top Delay Drivers (Model Insights)")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig2, ax2 = plt.subplots()
ax2.barh(importance_df["Feature"], importance_df["Importance"])
ax2.invert_yaxis()
ax2.set_xlabel("Importance Score")

st.pyplot(fig2)

st.subheader("Top At-Risk Orders")
st.dataframe(
    filtered.sort_values("delay_risk", ascending=False)
    [["Order_ID", "Priority", "Origin", "Destination", "delay_risk", "risk_level"]]
    .head(10)
)

st.download_button(
    "Download Risk Report",
    filtered.to_csv(index=False),
    "at_risk_deliveries.csv",
    "text/csv"
)
