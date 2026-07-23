"""
🍔 Food Delivery and Demand Analytics
--------------------------------------
A Streamlit app that serves the three models trained in MLPipeline.ipynb:

  1. ETA Prediction         -> models/eta_model.joblib
  2. Demand & Surge Forecast -> models/demand_model.joblib
                                 models/demand_features.joblib
                                 models/zone_thresholds.json
  3. Driver Segmentation    -> models/driver_cluster_model.joblib

Run with:
    streamlit run app.py
"""

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page config & style
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Food Delivery and Demand Analytics",
    page_icon="🍔",
    layout="wide",
)

MODELS_DIR = Path("models")

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.8rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 6px;}
        .stTabs [data-baseweb="tab"] {
            padding: 10px 18px;
            border-radius: 8px 8px 0 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍔 Food Delivery and Demand Analytics")
st.caption(
    "Predict delivery ETAs, forecast hourly demand & surge zones, "
    "and understand driver performance segments — all in one place."
)


# --------------------------------------------------------------------------
# Model loaders (cached so files are only read from disk once)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_eta_model():
    return joblib.load(MODELS_DIR / "eta_model.joblib")


@st.cache_resource(show_spinner=False)
def load_demand_assets():
    model = joblib.load(MODELS_DIR / "demand_model.joblib")
    features = joblib.load(MODELS_DIR / "demand_features.joblib")
    thresholds_path = MODELS_DIR / "zone_thresholds.json"
    thresholds = {}
    if thresholds_path.exists():
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
    return model, features, thresholds


@st.cache_resource(show_spinner=False)
def load_driver_bundle():
    return joblib.load(MODELS_DIR / "driver_cluster_model.joblib")


def missing_file_message(name: str, expected_path: str):
    st.warning(
        f"**{name} model not found.**\n\n"
        f"Expected the file at `{expected_path}`. Train and export it from "
        f"`MLPipeline.ipynb` (make sure `joblib.dump(...)` writes into a "
        f"relative `models/` folder rather than an absolute path like "
        f"`C:/Users/.../models/...`), then re-run this app."
    )


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

tab1, tab2, tab3 = st.tabs(
    ["🍔 ETA Prediction", "📈 Demand & Surge", "🛵 Driver Insights"]
)

# --------------------------------------------------------------------------
# TAB 1 — ETA Prediction
# --------------------------------------------------------------------------
with tab1:
    st.subheader("🍔 Predict Delivery Time")
    st.write("Adjust the order details below to estimate delivery time (minutes).")

    left, right = st.columns(2)

    with left:
        st.markdown("**Delivery Person**")
        age = st.slider("Delivery Person Age", 18, 60, 30)
        rating = st.slider("Delivery Person Rating", 1.0, 5.0, 4.6, 0.1)
        vehicle_condition = st.slider("Vehicle Condition (0 = poor, 3 = excellent)", 0, 3, 2)
        vehicle_type = st.selectbox(
            "Vehicle Type", ["motorcycle", "scooter", "electric_scooter"]
        )

        st.markdown("**Order Context**")
        order_type = st.selectbox("Order Type", ["Drinks", "Snack", "Meal", "Buffet"])
        multiple_deliveries = st.slider("Multiple Deliveries (bundled orders)", 0, 3, 0)
        festival = st.selectbox("Festival Day?", ["No", "Yes"]) == "Yes"

    with right:
        st.markdown("**Route & Environment**")
        weather = st.selectbox(
            "Weather", ["Sunny", "Windy", "Stormy", "Sandstorms", "Cloudy", "Fog"]
        )
        traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
        city_type = st.selectbox("City Type", ["Semi-Urban", "Urban", "Metropolitian"])
        distance_km = st.slider("Distance to Deliver (km)", 0.5, 25.0, 5.0, 0.1)

        st.markdown("**Timing**")
        order_hour = st.slider("Hour Order Placed (24h)", 0, 23, 13)
        order_day = st.selectbox("Day of Week", DAYS, index=0)
        prep_time = st.slider("Restaurant Prep Time (minutes)", 1, 60, 15)

    is_rush_hour = 1 if order_hour in [8, 9, 12, 13, 19, 20, 21] else 0
    day_of_week = DAYS.index(order_day)

    st.divider()
    predict_col, result_col = st.columns([1, 2])

    with predict_col:
        run_eta = st.button("Predict ETA", type="primary", use_container_width=True)

    with result_col:
        if run_eta:
            try:
                eta_pipeline = load_eta_model()
                # Coordinates aren't shown as sliders directly; we derive a
                # consistent lat/lon pair whose haversine distance equals the
                # slider value, since the model was trained on raw coordinates.
                restaurant_lat, restaurant_lon = 12.9716, 77.5946  # reference point
                delta_deg = distance_km / 111.0  # rough km-to-degree conversion
                delivery_lat = restaurant_lat + delta_deg
                delivery_lon = restaurant_lon + delta_deg

                row = pd.DataFrame([{
                    "Delivery_person_Age": age,
                    "Delivery_person_Ratings": rating,
                    "Restaurant_latitude": restaurant_lat,
                    "Restaurant_longitude": restaurant_lon,
                    "Delivery_location_latitude": delivery_lat,
                    "Delivery_location_longitude": delivery_lon,
                    "Weather": weather,
                    "Road_traffic_density": traffic,
                    "Vehicle_condition": vehicle_condition,
                    "Type_of_order": order_type,
                    "Type_of_vehicle": vehicle_type,
                    "multiple_deliveries": multiple_deliveries,
                    "Festival": int(festival),
                    "City": city_type,
                    "distance_km": distance_km,
                    "Order_day_of_week": day_of_week,
                    "Hour": order_hour,
                    "prep_time": prep_time,
                    "is_rush_hour": is_rush_hour,
                }])

                prediction = eta_pipeline.predict(row)[0]
                st.metric("Estimated Delivery Time", f"{prediction:.1f} minutes")
            except FileNotFoundError:
                missing_file_message("ETA", "models/eta_model.joblib")
            except Exception as e:
                st.error(f"Could not generate a prediction: {e}")
        else:
            st.info("Set your parameters and click **Predict ETA**.")

# --------------------------------------------------------------------------
# TAB 2 — Demand & Surge Forecast
# --------------------------------------------------------------------------
with tab2:
    st.subheader("📈 Forecast Next-Hour Demand & Surge Zones")
    st.write("Estimate how many orders a zone will see next hour and whether surge pricing should kick in.")

    left, right = st.columns(2)
    with left:
        zone_id = st.slider("Zone ID", 0, 19, 0)
        hour = st.slider("Hour of Day", 0, 23, 18)
        day_of_week_d = st.selectbox("Day of Week", DAYS, index=4, key="demand_day")
    with right:
        demand_last_hour = st.slider("Orders in the Last Hour", 0, 100, 20)
        demand_yesterday = st.slider("Orders at This Hour Yesterday", 0, 100, 18)
        traffic_last_hour = st.slider("Avg Delivery Time Last Hour (minutes)", 5, 90, 30)

    st.divider()
    predict_col2, result_col2 = st.columns([1, 2])

    with predict_col2:
        run_demand = st.button("Forecast Demand", type="primary", use_container_width=True)

    with result_col2:
        if run_demand:
            try:
                demand_model, feature_order, thresholds = load_demand_assets()

                row = pd.DataFrame([{
                    "Zone_ID": zone_id,
                    "Hour": hour,
                    "Demand_Last_Hour": demand_last_hour,
                    "Demand_Yesterday_Same_Hour": demand_yesterday,
                    "Traffic_Last_Hour": traffic_last_hour,
                    "Order_day_of_week": DAYS.index(day_of_week_d),
                }])[feature_order]

                predicted_demand = demand_model.predict(row)[0]

                threshold = thresholds.get(str(zone_id), thresholds.get(zone_id))
                m1, m2 = st.columns(2)
                m1.metric("Predicted Orders Next Hour", f"{predicted_demand:.0f}")

                if threshold is not None:
                    m2.metric("Zone Capacity Threshold (75th pct.)", f"{threshold:.0f}")
                    if predicted_demand > threshold:
                        st.error("🚨 Surge pricing recommended — predicted demand exceeds zone capacity.")
                    else:
                        st.success("✅ Demand within normal capacity — no surge needed.")
                else:
                    st.info("No capacity threshold found for this zone; showing raw forecast only.")
            except FileNotFoundError:
                missing_file_message("Demand", "models/demand_model.joblib")
            except Exception as e:
                st.error(f"Could not generate a forecast: {e}")
        else:
            st.info("Set your parameters and click **Forecast Demand**.")

# --------------------------------------------------------------------------
# TAB 3 — Driver Insights
# --------------------------------------------------------------------------
with tab3:
    st.subheader("🛵 Driver Performance Segmentation")
    st.write("See which performance cluster a driver profile falls into.")

    left, right = st.columns(2)
    with left:
        driver_rating = st.slider("Average Rating", 1.0, 5.0, 4.5, 0.1, key="driver_rating")
        avg_time_taken = st.slider("Average Delivery Time (minutes)", 10, 60, 28, key="driver_time")
    with right:
        total_deliveries = st.slider("Total Deliveries Completed", 5, 500, 50, key="driver_total")

    st.divider()
    predict_col3, result_col3 = st.columns([1, 2])

    with predict_col3:
        run_driver = st.button("Classify Driver", type="primary", use_container_width=True)

    with result_col3:
        if run_driver:
            try:
                bundle = load_driver_bundle()
                model, scaler = bundle["model"], bundle["scaler"]

                features = np.array([[driver_rating, avg_time_taken, total_deliveries]])
                scaled = scaler.transform(features)
                cluster = int(model.predict(scaled)[0])

                # Rank clusters by (unscaled) average rating to build a readable label
                centers = scaler.inverse_transform(model.cluster_centers_)
                order = np.argsort(-centers[:, 0])  # rating is column 0
                labels = ["⭐ Top Performer", "🚀 Reliable & Steady", "🐢 Needs Support"]
                label_map = {cluster_idx: labels[rank] if rank < len(labels) else f"Cluster {cluster_idx}"
                             for rank, cluster_idx in enumerate(order)}

                st.metric("Assigned Cluster", f"{label_map.get(cluster, cluster)}")
                st.caption(
                    f"Cluster center — avg rating {centers[cluster][0]:.2f}, "
                    f"avg delivery time {centers[cluster][1]:.1f} min, "
                    f"avg total deliveries {centers[cluster][2]:.0f}"
                )
            except FileNotFoundError:
                missing_file_message("Driver Clustering", "models/driver_cluster_model.joblib")
            except Exception as e:
                st.error(f"Could not classify this driver profile: {e}")
        else:
            st.info("Set your parameters and click **Classify Driver**.")

st.divider()
st.caption(
    "Models are loaded from the local `models/` folder. Retrain and export them "
    "from `MLPipeline.ipynb` if predictions look stale or files are missing."
)
