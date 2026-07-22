import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

# 1. Page Config
st.set_page_config(page_title="Logistics Analytics Suite", layout="wide")

# 2. Load Models (Cached for performance)
@st.cache_resource
def load_models():
    eta_model = joblib.load('models/eta_model.joblib')
    demand_model = joblib.load('models/demand_model.joblib')
    
    # Clustering bundles (model + scaler)
    zone_bundle = joblib.load('models/zone_cluster_model.joblib')
    driver_bundle = joblib.load('models/driver_cluster_model.joblib')
    
    # JSON thresholds
    with open('models/zone_thresholds.json', 'r') as f:
        thresholds = json.load(f)
        
    return eta_model, demand_model, zone_bundle, driver_bundle, thresholds

eta_model, demand_model, zone_bundle, driver_bundle, thresholds = load_models()

# 3. Main Dashboard Layout
st.title("📊 Food Delivery Logistics Engine")
tab1, tab2, tab3 = st.tabs(["ETA Predictor", "Demand & Surge", "Driver Insights"])

# --- TAB 1: ETA Predictor ---
with tab1:
    st.header("Delivery ETA Predictor")
    col1, col2 = st.columns(2)
    with col1:
        distance = st.number_input("Distance (km)", 0.0, 50.0, 5.0)
        weather = st.selectbox("Weather", ['Sunny', 'Windy', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog'])
        traffic = st.selectbox("Traffic Density", ['Low', 'Medium', 'High', 'Jam'])
    with col2:
        hour = st.slider("Hour of Day", 0, 23, 12)
        vehicle = st.selectbox("Vehicle", ['motorcycle', 'scooter', 'electric_scooter'])
        prep_time = st.number_input("Prep Time (min)", 0.0, 60.0, 15.0)

    if st.button("Predict ETA"):
        input_data = pd.DataFrame([[37, 4.9, 0, 0, 0, 0, weather, traffic, 2, 'Meal', vehicle, 0, 0, 'Urban', distance, 1, hour, prep_time, 0]], 
                                  columns=['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 
                                           'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude', 
                                           'Weather', 'Road_traffic_density', 'Vehicle_condition', 'Type_of_order', 
                                           'Type_of_vehicle', 'multiple_deliveries', 'Festival', 'City', 'distance_km', 
                                           'Order_day_of_week', 'Hour', 'prep_time', 'is_rush_hour'])
        
        prediction = eta_model.predict(input_data)[0]
        st.success(f"Predicted Delivery Time: {prediction:.2f} minutes")

# --- TAB 2: Surge Engine ---
with tab2:
    st.header("Surge Pricing & Demand Forecast")
    zone_id = st.number_input("Zone ID", 0, 19, 0)
    current_demand = st.number_input("Demand Last Hour", 0, 100, 10)
    
    if st.button("Check Surge"):
        # Predict demand using demand_model
        # (Assuming you pass the correct features derived from your notebook logic)
        pred_demand = demand_model.predict(pd.DataFrame([[zone_id, 12, current_demand, 10, 25.0, 5]], 
                                           columns=['Zone_ID', 'Hour', 'Demand_Last_Hour', 'Demand_Yesterday_Same_Hour', 'Traffic_Last_Hour','Order_day_of_week']))[0]
        
        st.write(f"Forecasted Demand: {pred_demand:.2f}")
        
        threshold = thresholds.get(str(zone_id), 50.0)
        if pred_demand > threshold:
            st.error(f"Surge Active: 1.5x Multiplier (Threshold: {threshold:.1f})")
        else:
            st.info("Standard Pricing: 1.0x")

# --- TAB 3: Driver Insights ---
with tab3:
    st.header("Driver Performance Clustering")
    rating = st.slider("Average Rating", 1.0, 5.0, 4.5)
    time_taken = st.slider("Avg Time Taken", 10.0, 60.0, 25.0)
    deliveries = st.number_input("Total Deliveries", 0, 500, 50)
    
    if st.button("Analyze Driver"):
        features = np.array([[rating, time_taken, deliveries]])
        scaled_features = driver_bundle['scaler'].transform(features)
        cluster = driver_bundle['model'].predict(scaled_features)[0]
        
        st.write(f"Driver assigned to Cluster: **{cluster}**")
        if cluster == 0:
            st.write("Profile: Star Driver (High efficiency, high quality)")
        else:
            st.write("Profile: Standard Fleet Driver")
