# Food Delivery & Demand Analytics 🍔📊

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📖 Overview
In the last decade, instant food delivery apps (like Zomato, Swiggy, Uber Eats) have transformed urban dining into complex real-time logistic networks. This project tackles three critical challenges in the food delivery ecosystem: **Customer Satisfaction**, **Profitability**, and **Operational Efficiency**.

By leveraging Machine Learning techniques, this project provides solutions for estimating delivery times, implementing dynamic surge pricing, and segmenting delivery partners for performance evaluation.


 
## 🚀 Key Problems Solved

### 1. Predicting Delivery Time (Customer Expectations)
**Goal:** mitigate customer dissatisfaction caused by inaccurate delivery estimates[cite: 10].
* **Approach:** Implemented various regression models to predict the time taken for delivery based on factors like weather, traffic density, and vehicle condition.
* **Best Model:** Random Forest Regressor[cite: 79].

### 2. Dynamic Pricing Strategy (Profitability)
**Goal:** Determine when to apply surge fees vs. fixed fees to ensure maximum profitability.
* **Approach:** A four-phased pipeline including Spatial Segmentation (Clustering), Feature Engineering (Lag features), Demand Forecasting, and a Threshold-based Pricing Algorithm.

### 3. Driver Segmentation (Performance Insights)
**Goal:** Evaluate delivery partner performance to aid in recruitment and training.
**Approach:** Unsupervised learning (Clustering) to categorize drivers based on Speed vs. Ratings/Quality[cite: 249].

---

## 📂 Dataset & Preprocessing
The dataset includes order-level details such as `Delivery_person_Age`, `Ratings`, `Restaurant_latitude`, `Weather`, `Road_traffic_density`, and `Time_taken`

**Preprocessing Steps:**
**Missing Values:** Checked and confirmed zero missing values.
**Encoding:** Ordinal Encoding for categorical variables (e.g., Weather: Sunny=0, Stormy=2).
**Feature Engineering:** * Split `Order_Date` into Day, Month, Year, and Day_of_Week.
     Extracted `Order_hour` from order timestamps.
**Scaling:** Applied Standard Scaler to normalize the dataset.


## 🛠️ Methodology & Results

### Part 1: Delivery Time Regression
We compared multiple algorithms to minimize Root Mean Squared Error (RMSE).

| Model | RMSE | R2 Score |
| Linear Regression | 6.0934 | 0.5754 |
| Support Vector Regressor | 5.6311 | 0.6374 |
| Decision Trees | 6.0368 | 0.5833 |
| **Random Forests (Best)** | **4.3954** |**0.7791** |

> **Insight:** Regression error mapping using PCA showed errors were randomly distributed, indicating the model handles noise well without failing on specific data subsets.


### Part 2: Dynamic Pricing Pipeline
This solution utilizes a "Forecast-then-Optimize" approach:

1.  **Spatial Segmentation:** Used **K-Means Clustering** on Latitude/Longitude to create operational "Zones".
2.  **Lag Features:** Created time-series features like `Demand_Last_Hour` and `Demand_Yesterday_Same_Hour`.
3.  **Forecasting:** Trained a Random Forest Regressor to predict demand for the next hour (MAE: 2.00 orders).
4.  **Pricing Logic:**
    * If `Predicted > Capacity + 20%` → **1.5x Surge**.
    * If `Predicted > Capacity` → **1.2x Surge**.

**Business Impact:** The simulation resulted in an **extra revenue generation of 10.2%** (Rs. 7,016) compared to the base fixed-fee revenue.


### Part 3: Driver Segmentation
We applied **K-Means Clustering** on:
* Average Ratings
* Average Delivery Time
* Number of Deliveries.

This segmented drivers into clusters representing different tradeoffs between **Speed and Quality**.



## 💻 Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib/Seaborn
* **Techniques:** Random Forest, K-Means Clustering, PCA, Feature Engineering

## 🔗 Links
**Project Repository:** [GitHub Link](https://github.com/Gitika-26/ML-Final-Project) 


## 📝 Author
**Gitika** ID: 20221054 
