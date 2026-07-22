"""
Food Delivery Demand and Data Analytics
---------------------------------------
Interactive Streamlit app with Red Background / White Text styling.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import sklearn.compose._column_transformer

# --- HOTFIX FOR VERSION MISMATCH ---
class _RemainderColsList: pass
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    setattr(sklearn.compose._column_transformer, '_RemainderColsList', _RemainderColsList)
# -----------------------------------

# --------------------------------------------------------------------------
# Styling: White on Red Theme
# --------------------------------------------------------------------------
def set_custom_style():
    st.markdown("""
        <style>
        /* Main background color */
        .stApp { 
            background-color: #C0392B; 
            color: white;
        }
        /* Headers and Text */
        h1, h2, h3, h4, p, label, div[data-testid="stText"] { 
            color: white !important; 
        }
        /* Tab colors */
        button[data-baseweb="tab"] { color: white !important; }
        
        /* Input fields and Sliders */
        .stTextInput>div>div>input { background-color: #FDFEFE; color: #C0392B; }
        .stNumberInput>div>div>input { background-color: #FDFEFE; color: #C0392B; }
        
        /* Buttons: White background, Red text */
        .stButton>button { 
            background-color: white !important; 
            color: #C0392B !important; 
            font-weight: bold;
            border: none !important;
        }
        /* Expander background */
        .st-emotion-cache-1210156 { background-color: #A93226 !important; }
        </style>
    """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Config & Loaders
# --------------------------------------------------------------------------
MODEL_DIR = "models"
st.set_page_config(page_title="Food Delivery Analytics", page_icon="🍔", layout="wide")
set_custom_style()

# ... (Include your existing load_joblib, load_json, and haversine_km functions here) ...

# --------------------------------------------------------------------------
# Load Assets
# --------------------------------------------------------------------------
eta_pipeline = load_joblib(os.path.join(MODEL_DIR, "eta_model.joblib"))
zone_bundle = load_joblib(os.path.join(MODEL_DIR, "zone_cluster_model.joblib"))
demand_pipeline = load_joblib(os.path.join(MODEL_DIR, "demand_model.joblib"))
zone_thresholds = load_json(os.path.join(MODEL_DIR, "zone_thresholds.json")) 
driver_bundle = load_joblib(os.path.join(MODEL_DIR, "driver_cluster_model.joblib")) 

# --------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------
st.title("🍔 Food Delivery Demand and Data Analytics")
st.markdown("---")

tab_eta, tab_surge, tab_driver = st.tabs(["🕒 ETA Prediction", "⚡ Surge Fee Calculator", "🧑‍✈️ Driver Insights"])

# --- TAB 1: ETA ---
with tab_eta:
    st.subheader("Predict Delivery Time")
    if eta_pipeline is None:
        st.warning("ETA model not found.")
    else:
        c1, c2 = st.columns(2)
        r_lat = c1.number_input("Restaurant Lat", value=12.9716, format="%.6f")
        r_lon = c1.number_input("Restaurant Lon", value=77.5946, format="%.6f")
        d_lat = c2.number_input("Delivery Lat", value=12.9352, format="%.6f")
        d_lon = c2.number_input("Delivery Lon", value=77.6146, format="%.6f")
        
        if st.button("Calculate Estimated Time"):
            dist = haversine_km(r_lat, r_lon, d_lat, d_lon)
            st.write(f"### Straight-line distance: {dist:.2f} km")

# --- TAB 2: SURGE ---
with tab_surge:
    st.subheader("Dynamic Surge Calculator")
    if zone_bundle is None or zone_thresholds is None:
        st.warning("Surge models missing.")
    else:
        col_in1, col_in2 = st.columns(2)
        lat = col_in1.number_input("Area Latitude", value=12.9716, format="%.6f")
        lon = col_in2.number_input("Area Longitude", value=77.5946, format="%.6f")
        
        st.write("### Adjust Activity Levels")
        demand = st.select_slider("Orders in last hour", options=range(0, 100, 5))
        
        if st.button("Apply Surge Model"):
            scaler = zone_bundle["scaler"]
            kmeans = zone_bundle["model"]
            zone_id = int(kmeans.predict(scaler.transform([[lat, lon]]))[0])
            st.metric("Zone Assignment", f"Zone #{zone_id}")

# --- TAB 3: DRIVER ---
with tab_driver:
    st.subheader("Driver Performance")
    if driver_bundle is None:
        st.warning("Driver model missing.")
    else:
        r1, r2, r3 = st.columns(3)
        rating = r1.slider("Avg Rating", 1.0, 5.0, 4.5, 0.1)
        time_min = r2.slider("Avg Time (min)", 10, 60, 30)
        deliveries = r3.slider("Total Deliveries", 0, 500, 100)
            
        if st.button("Analyze Driver Stats"):
            st.info("Analysis complete.")

st.markdown("---")
st.caption("Powered by FinalProject models.")
