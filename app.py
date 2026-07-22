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

# TAB 3 - DRIVER INSIGHTS



# ==========================================================================



with tab_driver:



    st.subheader("Driver Performance Segmentation")







    if driver_bundle is None:



        st.warning("`models/driver_cluster_model.joblib` not found.")



    else:



        driver_scaler = driver_bundle["scaler"]



        driver_kmeans = driver_bundle["model"]



        feature_order = ["Delivery_person_Ratings", "Time_taken", "Total_Deliveries"]







        c1, c2, c3 = st.columns(3)



        rating = c1.slider("Average rating", 1.0, 5.0, 4.5, 0.1)



        avg_time = c2.number_input("Average delivery time (min)", min_value=1.0, value=30.0)



        total_deliveries = c3.number_input("Total deliveries completed", min_value=1, value=100)







        if st.button("Analyze driver", type="primary"):



            try:



                scaled = driver_scaler.transform([[rating, avg_time, total_deliveries]])



                cluster_id = int(driver_kmeans.predict(scaled)[0])







                # Rank clusters using their (unscaled) centers to describe each one.



                centers_scaled = driver_kmeans.cluster_centers_



                centers = driver_scaler.inverse_transform(centers_scaled)



                centers_df = pd.DataFrame(centers, columns=feature_order)



                centers_df["Cluster"] = range(len(centers_df))







                best_rating_cluster = centers_df["Delivery_person_Ratings"].idxmax()



                fastest_cluster = centers_df["Time_taken"].idxmin()



                most_experienced_cluster = centers_df["Total_Deliveries"].idxmax()







                st.metric("Segment", f"Cluster {cluster_id}")







                labels = []



                if cluster_id == best_rating_cluster:



                    labels.append("⭐ Top-rated")



                if cluster_id == fastest_cluster:



                    labels.append("⚡ Fastest")



                if cluster_id == most_experienced_cluster:



                    labels.append("🏆 Most experienced")



                if not labels:



                    labels.append("📦 Steady / building experience")







                st.markdown("**Profile:** " + ", ".join(labels))







                row = centers_df.loc[cluster_id]



                st.write(



                    f"Drivers in this segment average a **{row['Delivery_person_Ratings']:.2f}⭐** rating, "



                    f"**{row['Time_taken']:.1f} min** delivery time, and **{row['Total_Deliveries']:.0f}** completed deliveries."



                )







                fig = go.Figure()



                fig.add_trace(go.Scatter(



                    x=centers_df["Time_taken"], y=centers_df["Delivery_person_Ratings"],



                    mode="markers+text", text=[f"Cluster {i}" for i in centers_df["Cluster"]],



                    textposition="top center", marker=dict(size=18, color="#4C78A8"), name="Other segments"



                ))



                fig.add_trace(go.Scatter(



                    x=[avg_time], y=[rating], mode="markers", marker=dict(size=20, color="#E45756", symbol="star"),



                    name="This driver"



                ))



                fig.update_layout(



                    title="Driver segments: speed vs. quality",



                    xaxis_title="Avg delivery time (min, lower is better)",



                    yaxis_title="Avg rating (higher is better)",



                    height=400,



                )



                st.plotly_chart(fig, use_container_width=True)



            except Exception as e:



                st.error(f"Couldn't analyze this driver: {e}")







st.markdown("---")



st.caption("Built from FinalProject.ipynb · ETA, surge pricing, and driver segmentation models.") 
