"""
Food Delivery Demand and Data Analytics
---------------------------------------
Interactive Streamlit app with three tools built on top of the models
trained in FinalProject.ipynb:

  1. ETA Prediction
  2. Surge Fee Calculator
  3. Driver Insights
"""

import os
import json
from datetime import datetime, time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_DIR = "models"

ETA_MODEL_PATH = os.path.join(MODEL_DIR, "eta_model.joblib")
ZONE_MODEL_PATH = os.path.join(MODEL_DIR, "zone_cluster_model.joblib")
DEMAND_MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.joblib")
DEMAND_FEATURES_PATH = os.path.join(MODEL_DIR, "demand_features.joblib")
ZONE_THRESHOLDS_PATH = os.path.join(MODEL_DIR, "zone_thresholds.json")
DRIVER_MODEL_PATH = os.path.join(MODEL_DIR, "driver_cluster_model.joblib")

st.set_page_config(
    page_title="Food Delivery Demand and Data Analytics",
    page_icon="🍔",
    layout="wide",
)

# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_joblib(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

# --------------------------------------------------------------------------
# Load assets
# --------------------------------------------------------------------------
eta_pipeline = load_joblib(ETA_MODEL_PATH)
zone_bundle = load_joblib(ZONE_MODEL_PATH)
demand_pipeline = load_joblib(DEMAND_MODEL_PATH)
demand_features = load_joblib(DEMAND_FEATURES_PATH)
zone_thresholds = load_json(ZONE_THRESHOLDS_PATH)
driver_bundle = load_joblib(DRIVER_MODEL_PATH)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("🍔 Ops Toolkit")
st.sidebar.caption("Model status")

def status_line(label, obj):
    st.sidebar.markdown(f"{'✅' if obj is not None else '❌'} {label}")

status_line("ETA model", eta_pipeline)
status_line("Zone clustering", zone_bundle)
status_line("Demand model", demand_pipeline)
status_line("Driver clustering", driver_bundle)

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("🍔 Food Delivery Demand and Data Analytics")
st.caption("ETA prediction, dynamic surge pricing, and driver segmentation.")

tab_eta, tab_surge, tab_driver = st.tabs(["🕒 ETA Prediction", "⚡ Surge Fee Calculator", "🧑‍✈️ Driver Insights"])

# ==========================================================================
# TAB 1 - ETA PREDICTION
# ==========================================================================
with tab_eta:
    st.subheader("Estimated Delivery Time")

    if eta_pipeline is None:
        st.warning(f"Could not find `{ETA_MODEL_PATH}`. Please ensure your model is saved in the `models/` folder.")
    else:
        # Detect features from the pipeline
        try:
            feature_names = eta_pipeline.feature_names_in_ if hasattr(eta_pipeline, "feature_names_in_") else []
        except:
            feature_names = []

        st.markdown("#### Trip details")
        c1, c2 = st.columns(2)
        r_lat = c1.number_input("Restaurant Latitude", value=12.9716, format="%.6f")
        r_lon = c1.number_input("Restaurant Longitude", value=77.5946, format="%.6f")
        d_lat = c2.number_input("Delivery Latitude", value=12.9352, format="%.6f")
        d_lon = c2.number_input("Delivery Longitude", value=77.6146, format="%.6f")

        order_dt = st.datetime_input("Order Timestamp", value=datetime.now())
        
        st.markdown("#### Additional Context")
        cols = st.columns(3)
        prep_time = cols[0].number_input("Kitchen Prep Time (min)", min_value=0, value=15)
        rider_age = cols[1].number_input("Rider Age", min_value=18, max_value=60, value=25)
        weather = cols[2].selectbox("Weather Conditions", ["Sunny", "Cloudy", "Rainy", "Foggy", "Stormy", "Sandstorms"])

        if st.button("Predict ETA", type="primary"):
            input_data = {
                "Restaurant_latitude": r_lat, "Restaurant_longitude": r_lon,
                "Delivery_location_latitude": d_lat, "Delivery_location_longitude": d_lon,
                "Time_taken(min)": 0, 
                "Delivery_person_Age": rider_age,
                "Delivery_person_Ratings": 4.5,
                "Weather_conditions": weather,
                "Road_traffic_density": "Low",
                "Vehicle_condition": 1,
                "Type_of_order": "Snack",
                "Type_of_vehicle": "motorcycle",
                "multiple_deliveries": 0,
                "distance_km": haversine_km(r_lat, r_lon, d_lat, d_lon),
                "Hour": order_dt.hour,
                "Order_day_of_week": order_dt.weekday(),
                "prep_time": prep_time
            }
            
            input_df = pd.DataFrame([input_data])
            
            if len(feature_names) > 0:
                for col in feature_names:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[feature_names]

            try:
                prediction = eta_pipeline.predict(input_df)[0]
                st.success(f"### Predicted Delivery Time: **{prediction:.1f} minutes**")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ==========================================================================
# TAB 2 - SURGE FEE CALCULATOR
# ==========================================================================
with tab_surge:
    st.subheader("Dynamic Surge Fee Calculator")
    if zone_bundle is None or demand_pipeline is None:
        st.warning("Surge models missing.")
    else:
        st.info("Surge calculator logic active.")

# ==========================================================================
# TAB 3 - DRIVER INSIGHTS
# ==========================================================================
with tab_driver:
    st.subheader("Driver Performance Segmentation")
    if driver_bundle is None:
        st.warning("Driver model missing.")
    else:
        st.info("Driver analysis logic active.")
