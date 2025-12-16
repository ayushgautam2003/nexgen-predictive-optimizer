# NexGen Logistics – Predictive Delivery Optimizer 🚚📦

## Overview
This project is an interactive **machine learning–powered logistics analytics application** built for **NexGen Logistics Pvt. Ltd.**  
The goal is to transform logistics operations from **reactive** to **predictive** by identifying deliveries at risk of delay before they occur.

The application uses historical logistics data and a **Random Forest classification model**, deployed through an interactive **Streamlit dashboard**.

---

## Problem Statement
NexGen Logistics faces several operational challenges:
- Delivery performance issues
- Rising operational and transportation costs
- Reactive decision-making
- Limited predictive insights
- Increasing customer dissatisfaction

There is a strong need for a **data-driven, predictive solution** that can help logistics managers proactively manage delays and improve service quality.

---

## Solution Approach
The Predictive Delivery Optimizer:
- Integrates multiple logistics datasets (orders, delivery performance, routes, and costs)
- Engineers meaningful operational features
- Uses a **Random Forest machine learning model** to predict delivery delay risk
- Presents insights through an interactive Streamlit dashboard
- Enables proactive decision-making through risk prioritization

---

## Key Features
- 📊 **Predictive Delay Risk Scoring** using machine learning
- 🚦 **Priority-based filtering** (Express / Standard / Economy)
- 📈 **Operational KPIs** (Delay Rate, Cost, Volume)
- 🧠 **Model Explainability** via feature importance analysis
- 📥 **Downloadable At-Risk Orders Report**
- 🔄 **Dynamic dashboard updates** based on user input

---

## Machine Learning Details
- **Model Used:** Random Forest Classifier  
- **Problem Type:** Binary Classification (Delayed vs On-Time)
- **Target Variable:** `delayed`
- **Key Features:**
  - Distance (KM)
  - Traffic Delay (minutes)
  - Weather Impact
  - Delivery Cost
  - Fuel Cost
  - Labor Cost

The model outputs a **probability score (`delay_risk`)**, which is used to rank and prioritize high-risk deliveries.

---

## Datasets Used
The solution integrates the following datasets:
- `orders.csv`
- `delivery_performance.csv`
- `routes_distance.csv`
- `cost_breakdown.csv`
- `vehicle_fleet.csv`
- `warehouse_inventory.csv`
- `customer_feedback.csv`

> Note: Not all orders have complete data, reflecting real-world logistics scenarios.

---

## Tech Stack
- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**

---

## How to Run the Application Locally
streamlit run app.py

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
