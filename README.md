# Food Delivery and Demand Analytics
An end-to-end machine learning pipeline for a food delivery platform, built from a single raw operations dataset. The pipeline produces three deployable artifacts: a delivery-time (ETA) predictor, a demand forecasting + surge pricing system, and a driver performance segmentation model.

## Live Demo
Access the interactive dashboard here:
[Food Delivery and Demand Analytics](https://food-delivery-and-demand-analytics-7xexl4jpudseankjpsqeyt.streamlit.app/)


 
## Overview
 
Food delivery platforms need to solve three interlocking operational problems at once: **how long will this delivery take**, **when should we charge a surge fee because a zone is overloaded**, and **who are our best drivers**. This notebook builds a lightweight but complete pipeline for all three, starting from a single raw CSV of historical deliveries and ending with saved, reusable model artifacts.
 
The workflow is organized as three sequential stages that share the same cleaned dataset:
 
```
Raw CSV (data/food.csv)
        │
        ▼
Cleaning & Feature Engineering
        │
   ┌────┴─────┬──────────────────┐
   ▼          ▼                  ▼
ETA Model  Zone Clustering   Driver Clustering
           + Demand Model    (Segmentation)
           (Surge Pricing)
```
 
## Dataset
 
The notebook expects a CSV at `data/food.csv`. Relevant columns used across the pipeline:
 
**Geospatial**
- `Restaurant_latitude`, `Restaurant_longitude`
- `Delivery_location_latitude`, `Delivery_location_longitude`
**Time**
- `Order_Date`, `Time_Ordered`, `Time_Order_picked`
**Categorical / operational context**
- `Road_traffic_density` (`Low`, `Medium`, `High`, `Jam`)
- `Weather` (`Sunny`, `Windy`, `Stormy`, `Sandstorms`, `Cloudy`, `Fog`)
- `Type_of_order` (`Drinks`, `Snack`, `Meal`, `Buffet`)
- `Type_of_vehicle` (`motorcycle`, `scooter`, `electric_scooter`)
- `City` (`Semi-Urban`, `Urban`, `Metropolitian`)
**People**
- `Delivery_person_ID`, `Delivery_person_Age`, `Delivery_person_Ratings`
**Target**
- `Time_taken` — actual delivery time in minutes, used as the label for ETA prediction and as the basis for demand/surge calculations
## Project Structure
 
```
.
├── MLPipeline.ipynb
├── data/
│   └── food.csv              # input dataset (not included — add your own)
└── models/                   # created during the run; stores saved artifacts
    ├── eta_model.joblib
    ├── zone_cluster_model.joblib
    ├── demand_model.joblib
    ├── demand_features.joblib
    ├── zone_thresholds.json
    └── driver_cluster_model.joblib
```
 
## Pipeline Walkthrough
 
### 1. Data Loading & Cleaning
 
- Loads `data/food.csv` with a guarded `try/except` so a missing file produces a clear error instead of a crash.
- Checks for missing values with `isnull().sum()`.
- Strips stray whitespace from categorical text columns (`Road_traffic_density`, `Type_of_order`, `Type_of_vehicle`, `City`, `Weather`), which is a common source of silent encoding bugs in scraped/exported operational data.
### 2. Feature Engineering
 
- **Haversine distance**: computes the great-circle distance in kilometers between the restaurant and the delivery address from raw latitude/longitude pairs — a far more informative feature than raw coordinates for predicting delivery time.
- **Time features**: extracts day-of-week and hour-of-day from the order timestamp, since delivery time is strongly influenced by *when* an order is placed, not just *where*.
- **Prep time**: the gap in minutes between `Time_Ordered` and `Time_Order_picked`, i.e. how long the restaurant took to have the order ready. Missing values are imputed with the median.
- **Rush hour flag**: a binary indicator for whether an order falls in common peak windows (8–9am, 12–1pm, 7–9pm).
- Drops identifier and now-redundant raw timestamp columns (`ID`, `Delivery_person_ID`, `Time_Order_picked`, `Time_Ordered`, `Order_Date`) before modeling, since these either leak identity or have already been converted into usable numeric features.
### 3. ETA Prediction
 
**Goal:** predict `Time_taken` (delivery time in minutes) from order/context features.
 
- Data is split 80/20 into train/test sets.
- A `ColumnTransformer` handles preprocessing:
  - `OrdinalEncoder` with an explicit, domain-informed category ordering for traffic density, order type, vehicle type, city tier, and weather (e.g. traffic is ordered `Low < Medium < High < Jam`, which preserves meaningful rank rather than treating categories as unordered).
  - `StandardScaler` for all remaining numeric features.
- Four regressors are trained and compared inside identical pipelines (so preprocessing is refit correctly within each, avoiding leakage):
  - Linear Regression
  - Support Vector Regression (SVR)
  - Decision Tree Regressor
  - Random Forest Regressor
- Models are scored on **R²** and **RMSE**, and results are visualized as a bar chart ranked by R².
- **Random Forest** is selected as the best-performing model, retrained as the final pipeline, and evaluated with an actual-vs-predicted scatter plot (with a red diagonal reference line showing a "perfect fit") to visually inspect prediction quality and bias.
- The final pipeline (preprocessing + model, bundled together) is saved to `models/eta_model.joblib` so it can be reused for inference without re-running preprocessing code separately.
### 4. Surge Fee / Demand Forecasting
 
**Goal:** identify overloaded delivery zones in near-real-time and support dynamic surge pricing decisions.
 
**Step 1 — Zone clustering.** Valid (non-zero) delivery coordinates are standardized and clustered into **20 geographic zones** with `KMeans`. This turns raw lat/lon into a discrete, interpretable "zone" feature usable in downstream models. Zones are visualized as a color-coded scatter plot over the delivery area, and the clustering model + scaler are saved together (`models/zone_cluster_model.joblib`) so new coordinates can be mapped to the same zones later.
 
**Step 2 — Historical demand aggregation.** Orders are grouped by `Zone_ID`, day of week, and hour to compute:
- `Actual_Demand` — order count in that zone/hour slot
- `Avg_Delivery_Time` — average delivery time in that slot
- `Avg_Rating` — average driver rating in that slot
From this, two lag features are engineered per zone:
- `Demand_Last_Hour` — demand one hour prior
- `Demand_Yesterday_Same_Hour` — demand 24 hours prior (captures daily seasonality, e.g. lunch/dinner rushes)
- `Traffic_Last_Hour` — average delivery time in the previous hour, used as a proxy for current congestion
**Step 3 — Demand forecasting model.** A `StandardScaler` + `RandomForestRegressor` pipeline is trained to predict `Actual_Demand` (next-hour order volume) from `Zone_ID`, `Hour`, `Order_day_of_week`, and the lag features above. The trained pipeline and its feature list are saved to `models/demand_model.joblib` and `models/demand_features.joblib`.
 
**Step 4 — Surge thresholds.** For each zone, the 75th percentile of historical demand is computed as a capacity threshold. When forecasted (or actual) demand exceeds this threshold, the zone is a candidate for a surge fee. Thresholds are exported to `models/zone_thresholds.json` for easy use by a downstream pricing service.
 
### 5. Driver Performance Segmentation
 
**Goal:** group drivers into performance tiers to surface hiring and coaching insights.
 
- Driver-level profiles are aggregated: average age, average rating, average delivery time, and total completed deliveries.
- Drivers with 5 or fewer deliveries are filtered out to avoid noisy profiles from very small sample sizes.
- Features (`Delivery_person_Ratings`, `Time_taken`, `Total_Deliveries`) are standardized and clustered into **3 groups** with `KMeans`.
- Clusters are visualized on a speed-vs-rating scatter plot, with dashed lines marking the overall average speed and rating for quick visual segmentation (e.g. fast+highly-rated vs. slow+low-rated).
- The cluster with the highest average rating is treated as the "star driver" segment, and its typical age and delivery volume are printed as a hiring/coaching insight.
- The clustering model + scaler are saved to `models/driver_cluster_model.joblib`.
## Model Artifacts
 
| File | Contents |
|---|---|---|
| `models/eta_model.joblib` | Full `Pipeline` (preprocessing + Random Forest) for ETA prediction |
| `models/zone_cluster_model.joblib` | Dict with `model` (KMeans) and `scaler` (StandardScaler) for mapping coordinates to zones |
| `models/demand_model.joblib` | Full `Pipeline` (scaler + Random Forest) for next-hour demand forecasting |
| `models/demand_features.joblib` | List of feature names expected by the demand model, for consistent inference |
| `models/zone_thresholds.json` | Dict mapping `Zone_ID` → 75th-percentile demand threshold |
| `models/driver_cluster_model.joblib` | Dict with `model` (KMeans) and `scaler` (StandardScaler) for driver segmentation |
 
## Requirements
 
- Python 3
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
Install with:
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```
 
## Setup & Usage
 
1. Clone the repository and add your dataset at `data/food.csv`.
2. Create a `models/` directory in the project root (if it doesn't already exist) — this is where all trained artifacts are saved.
3. Launch Jupyter and run `MLPipeline.ipynb` top to bottom:
```bash
jupyter notebook MLPipeline.ipynb
```
 
4. After running, load any saved artifact for inference, e.g.:
```python
import joblib
 
eta_model = joblib.load("models/eta_model.joblib")
prediction = eta_model.predict(new_orders_df)
```
 
## Design Notes
 
- **Pipelines over ad-hoc preprocessing**: every model (ETA, demand) is wrapped in a `scikit-learn` `Pipeline` bundling preprocessing and the estimator together. This prevents train/test leakage during model comparison and means the saved `.joblib` file is a single, self-contained object — no need to separately track and re-apply a scaler or encoder at inference time.
- **Ordinal encoding with explicit category order**: rather than one-hot encoding traffic density or weather, the notebook defines an explicit ordinal ranking (e.g. `Low < Medium < High < Jam`) where a natural order exists, preserving that signal for tree-based and linear models alike.
- **Lag features for demand forecasting**: `Demand_Last_Hour` and `Demand_Yesterday_Same_Hour` let a fairly simple Random Forest capture both short-term momentum and daily seasonality without needing a dedicated time-series model.
- **Zone-based rather than point-based surge logic**: clustering coordinates into discrete zones keeps the surge system interpretable and operationally simple (a small number of zone-level thresholds) rather than trying to price at the individual delivery level.
 
