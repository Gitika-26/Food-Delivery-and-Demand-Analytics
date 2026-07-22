"""
Food Delivery Demand and Data Analytics
---------------------------------------
Interactive Streamlit app with three tools built on top of the models
from FinalProject.ipynb.
"""

import os
import json
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import sklearn.compose._column_transformer

# --- HOTFIX FOR VERSION MISMATCH ---
# Patches the missing attribute in Scikit-Learn to prevent load errors
class _RemainderColsList: pass
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    setattr(sklearn.compose._column_transformer, '_RemainderColsList', _RemainderColsList)
# -----------------------------------

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_DIR = "models"
ETA_MODEL_PATH = os.path.join(MODEL_DIR, "eta_model.joblib")
ZONE_MODEL_PATH = os.path.join(MODEL_DIR, "zone_cluster_model.joblib")
DEMAND_MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.joblib")
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
    if not os.path.exists(path): return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_json(path):
    if not os.path.exists(path): return None
    with open(path, "r") as f: return json.load(f)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

# --------------------------------------------------------------------------
# Load Assets
# --------------------------------------------------------------------------
eta_pipeline = load_joblib(ETA_MODEL_PATH)
zone_bundle = load_joblib(ZONE_MODEL_PATH)
demand_pipeline = load_joblib(DEMAND_MODEL_PATH)
zone_thresholds = load_json(ZONE_THRESHOLDS_PATH) #[cite: 6]
driver_bundle = load_joblib(DRIVER_MODEL_PATH) #[cite: 3, 5]

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("🍔 Food Delivery Demand and Data Analytics")
st.caption("ETA prediction, surge pricing, and driver insights.")

tab_eta, tab_surge, tab_driver = st.tabs(["🕒 ETA Prediction", "⚡ Surge Fee Calculator", "🧑‍✈️ Driver Insights"])

# --- TAB 1 ---
with tab_eta:
    st.subheader("Estimated Delivery Time")
    if eta_pipeline is None:
        st.warning("ETA model not found.")
    else:
        c1, c2 = st.columns(2)
        r_lat = c1.number_input("Restaurant Lat", value=12.9716, format="%.6f")
        r_lon = c1.number_input("Restaurant Lon", value=77.5946, format="%.6f")
        d_lat = c2.number_input("Delivery Lat", value=12.9352, format="%.6f")
        d_lon = c2.number_input("Delivery Lon", value=77.6146, format="%.6f")
        
        if st.button("Predict ETA"):
            dist = haversine_km(r_lat, r_lon, d_lat, d_lon)
            st.write(f"Distance: {dist:.2f} km")
            # Logic implementation continues here...

# --- TAB 2 ---
with tab_surge:
    st.subheader("Dynamic Surge Fee Calculator")
    if zone_bundle is None or zone_thresholds is None:
        st.warning("Surge models/thresholds missing.")
    else:
        # Features defined based on[cite: 2]
        lat = st.number_input("Latitude", value=12.9716, format="%.6f")
        lon = st.number_input("Longitude", value=77.5946, format="%.6f")
        
        if st.button("Calculate Surge"):
            # Zone prediction logic using zone_bundle[cite: 3, 5]
            scaler = zone_bundle["scaler"]
            kmeans = zone_bundle["model"]
            zone_id = int(kmeans.predict(scaler.transform([[lat, lon]]))[0])
            st.metric("Detected Zone", f"Zone {zone_id}")

# --- TAB 3 ---
with tab_driver:
    st.subheader("Driver Performance Segmentation")
    if driver_bundle is None:
        st.warning("Driver model not found.")
    else:
        st.write("Driver insights enabled.")
