# Food Delivery and Demand Analytics

## Live Demo
Access the interactive dashboard here:
[Food Delivery and Demand Analytics](https://food-delivery-and-demand-analytics-7xexl4jpudseankjpsqeyt.streamlit.app/)


This project analyzes food delivery data to predict Estimated Time of Arrival (ETA) using machine learning and provides actionable insights for demand management and driver performance.

## Overview
The application leverages a machine learning pipeline to help delivery platforms optimize logistics. Key features include:
* ETA Prediction: Regression models (Random Forest, SVR, Decision Tree, Linear Regression) to predict delivery duration.
* Demand Forecasting: Predicting order volume per zone to assist in surge fee classification.
* Driver Profiling: Clustering analysis to categorize drivers based on performance and efficiency.

## Tech Stack
* Language: Python
* Data Processing: pandas, numpy
* Machine Learning: scikit-learn
* Visualization: matplotlib, seaborn
* Deployment: streamlit
* Model Persistence: joblib

## Project Workflow
1. Preprocessing: Data cleaning, handling missing values, and engineering features like Haversine distance, prep time, and rush-hour flagging.
2. Geospatial Clustering: Using K-Means to divide delivery areas into distinct demand zones.
3. ETA Modeling: A comparison of regressor models where Random Forest achieved the best performance (R squared approximately 0.83).
4. Demand Analysis: Calculation of surge thresholds based on historical demand quantiles.
5. Driver Insights: Clustering drivers to identify performance segments.

## File Structure
* /data/: Contains the source food.csv dataset.
* /models/: Saved Scikit-Learn pipelines and cluster models (eta_model.joblib, zone_cluster_model.joblib, etc.).
* notebooks/: Exploratory Data Analysis (EDA) and model training scripts.
* app.py: The Streamlit dashboard code.

## Key Metrics (Random Forest)
| Model | R2 Score | RMSE |
| :--- | :--- | :--- |
| Random Forest | 0.8326 | 3.8259 |
| Decision Tree | 0.6863 | 5.2375 |
| SVR | 0.6423 | 5.5928 |
| Linear Regression | 0.5783 | 6.0725 |


