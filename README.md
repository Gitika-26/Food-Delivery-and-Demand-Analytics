# Food Delivery and Demand Analytics
An end-to-end machine learning project on a food delivery dataset, covering delivery time (ETA) prediction, demand forecasting for dynamic surge pricing, and driver performance segmentation.

## Live Demo
Access the interactive dashboard here:
[Food Delivery and Demand Analytics](https://food-delivery-and-demand-analytics-7xexl4jpudseankjpsqeyt.streamlit.app/)


## Overview

This project uses a food delivery dataset (restaurant/delivery coordinates, weather, traffic, order details, and driver ratings) to build three connected ML components:

1. **ETA Prediction** — regression models that predict delivery time in minutes
2. **Surge Fee / Demand Forecasting** — geospatial clustering of demand zones plus a model that forecasts next-hour order demand per zone, used to flag when a zone is approaching capacity
3. **Driver Segmentation** — clustering of drivers by speed and rating to surface "star driver" traits for hiring/coaching insights

## Data

The notebook expects a CSV at `data/food.csv` with columns including:

- `Restaurant_latitude`, `Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude`
- `Order_Date`, `Time_Ordered`, `Time_Order_picked`
- `Road_traffic_density`, `Weather`, `Type_of_order`, `Type_of_vehicle`, `City`
- `Delivery_person_ID`, `Delivery_person_Age`, `Delivery_person_Ratings`
- `Time_taken` (target for ETA prediction)

## Feature Engineering

- **Haversine distance** between restaurant and delivery location, computed from lat/lon coordinates
- **Time features**: order day of week, hour of day
- **Prep time**: minutes between order placed and order picked up
- **Rush hour flag**: order placed during common peak hours (8–9am, 12–1pm, 7–9pm)
- Whitespace cleanup on categorical text columns
- Irrelevant/ID columns dropped before modeling

## 1. ETA Prediction

Several regression models are trained on the engineered features and compared:

- Linear Regression
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor

Categorical features (traffic density, order type, vehicle type, city, weather) are ordinal-encoded with a defined ordering, and numeric features are standardized, all inside a single `scikit-learn` `Pipeline`/`ColumnTransformer` to avoid data leakage between train and test splits.

Models are compared using **R² score** and **RMSE**, visualized with a bar chart. The best-performing model (**Random Forest**) is retrained as the final pipeline and evaluated with an actual-vs-predicted scatter plot.

## 2. Surge Fee / Demand Forecasting

- Delivery locations are clustered into **20 geographic zones** using `KMeans` on standardized coordinates, visualized as a scatter plot of demand zones.
- Historical order data is aggregated by zone, day of week, and hour to build features such as `Demand_Last_Hour`, `Demand_Yesterday_Same_Hour`, and `Traffic_Last_Hour`.
- A Random Forest model is trained to forecast **next-hour order demand** per zone.
- Per-zone capacity thresholds (75th percentile of historical demand) are computed to flag when a zone should trigger a surge fee.

## 3. Driver Performance Segmentation

- Driver-level profiles are built from average rating, average delivery time, and total completed deliveries (filtered to drivers with more than 5 deliveries).
- `KMeans` (3 clusters) groups drivers by speed and quality.
- The cluster with the highest average rating is treated as the "star driver" segment, and its average age and delivery volume are summarized as hiring/coaching insights.
- Results are visualized as a speed-vs-rating scatter plot with average reference lines.

## Requirements

- Python 3
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `joblib`

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Usage

1. Place your dataset at `data/food.csv`.
2. Create a `models/` directory in the project root for saved model artifacts.
3. Run `FinalProject.ipynb` top to bottom in Jupyter.

The notebook will train and save the following artifacts to `models/`:

| File | Description |
|---|---|
| `eta_model.joblib` | Full preprocessing + Random Forest pipeline for ETA prediction |
| `zone_cluster_model.joblib` | KMeans zone clustering model + scaler |
| `demand_model.joblib` | Next-hour demand forecasting pipeline |
| `demand_features.joblib` | Feature list used by the demand model |
| `zone_thresholds.json` | Per-zone surge capacity thresholds |
| `driver_cluster_model.joblib` | Driver segmentation KMeans model + scaler |

 
