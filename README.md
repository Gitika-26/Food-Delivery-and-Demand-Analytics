# Food Delivery & Demand Analytics 🍔📊

![Project Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📖 Overview
[cite_start]In the last decade, instant food delivery apps (like Zomato, Swiggy, Uber Eats) have transformed urban dining into complex real-time logistic networks[cite: 6, 7]. This project tackles three critical challenges in the food delivery ecosystem: **Customer Satisfaction**, **Profitability**, and **Operational Efficiency**.

By leveraging Machine Learning techniques, this project provides solutions for estimating delivery times, implementing dynamic surge pricing, and segmenting delivery partners for performance evaluation.

---

## 🚀 Key Problems Solved

### 1. Predicting Delivery Time (Customer Expectations)
[cite_start]**Goal:** mitigate customer dissatisfaction caused by inaccurate delivery estimates[cite: 10].
* **Approach:** Implemented various regression models to predict the time taken for delivery based on factors like weather, traffic density, and vehicle condition.
* [cite_start]**Best Model:** Random Forest Regressor[cite: 79].

### 2. Dynamic Pricing Strategy (Profitability)
[cite_start]**Goal:** Determine when to apply surge fees vs. fixed fees to ensure maximum profitability[cite: 12, 13].
* [cite_start]**Approach:** A four-phased pipeline including Spatial Segmentation (Clustering), Feature Engineering (Lag features), Demand Forecasting, and a Threshold-based Pricing Algorithm[cite: 133].

### 3. Driver Segmentation (Performance Insights)
[cite_start]**Goal:** Evaluate delivery partner performance to aid in recruitment and training[cite: 15].
* [cite_start]**Approach:** Unsupervised learning (Clustering) to categorize drivers based on Speed vs. Ratings/Quality[cite: 249].

---

## 📂 Dataset & Preprocessing
[cite_start]The dataset includes order-level details such as `Delivery_person_Age`, `Ratings`, `Restaurant_latitude`, `Weather`, `Road_traffic_density`, and `Time_taken`[cite: 31].

**Preprocessing Steps:**
* [cite_start]**Missing Values:** Checked and confirmed zero missing values[cite: 34].
* [cite_start]**Encoding:** Ordinal Encoding for categorical variables (e.g., Weather: Sunny=0, Stormy=2) [cite: 36-38].
* [cite_start]**Feature Engineering:** * Split `Order_Date` into Day, Month, Year, and Day_of_Week[cite: 44].
    * [cite_start]Extracted `Order_hour` from order timestamps[cite: 45].
* [cite_start]**Scaling:** Applied Standard Scaler to normalize the dataset[cite: 47].

---

## 🛠️ Methodology & Results

### Part 1: Delivery Time Regression
We compared multiple algorithms to minimize Root Mean Squared Error (RMSE).

| Model | RMSE | R2 Score |
| :--- | :--- | :--- |
| Linear Regression | 6.0934 | [cite_start]0.5754 [cite: 68] |
| Support Vector Regressor | 5.6311 | [cite_start]0.6374 [cite: 71] |
| Decision Trees | 6.0368 | [cite_start]0.5833 [cite: 74] |
| **Random Forests (Best)** | **4.3954** | [cite_start]**0.7791** [cite: 76-78] |

> [cite_start]**Insight:** Regression error mapping using PCA showed errors were randomly distributed, indicating the model handles noise well without failing on specific data subsets[cite: 124].


### Part 2: Dynamic Pricing Pipeline
This solution utilizes a "Forecast-then-Optimize" approach:

1.  [cite_start]**Spatial Segmentation:** Used **K-Means Clustering** on Latitude/Longitude to create operational "Zones" [cite: 134-136].
2.  [cite_start]**Lag Features:** Created time-series features like `Demand_Last_Hour` and `Demand_Yesterday_Same_Hour`[cite: 140].
3.  [cite_start]**Forecasting:** Trained a Random Forest Regressor to predict demand for the next hour (MAE: 2.00 orders)[cite: 143, 235].
4.  **Pricing Logic:**
    * [cite_start]If `Predicted > Capacity + 20%` → **1.5x Surge**[cite: 148].
    * [cite_start]If `Predicted > Capacity` → **1.2x Surge**[cite: 149].

[cite_start]**Business Impact:** The simulation resulted in an **extra revenue generation of 10.2%** (Rs. 7,016) compared to the base fixed-fee revenue[cite: 241].


### Part 3: Driver Segmentation
We applied **K-Means Clustering** on:
* Average Ratings
* Average Delivery Time
* [cite_start]Number of Deliveries [cite: 272-276].

[cite_start]This segmented drivers into clusters representing different tradeoffs between **Speed and Quality**[cite: 263].

---

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib/Seaborn
* **Techniques:** Random Forest, K-Means Clustering, PCA, Feature Engineering

## 🔗 Links
* [cite_start]**Project Repository:** [GitHub Link](https://github.com/Gitika-26/ML-Final-Project) [cite: 28]
* [cite_start]**Dataset Source:** [Link provided in repo] [cite: 30]

---

## 📝 Author
[cite_start]**Gitika** ID: 20221054 [cite: 3, 4]
