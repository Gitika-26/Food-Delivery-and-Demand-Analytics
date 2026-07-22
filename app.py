import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Page Configuration
st.set_page_config(page_title="Delivery ETA Engine", page_icon="🚚", layout="wide")

st.title("🚚 Delivery ETA Predictor")
st.markdown("Calculate the estimated time for delivery based on your operational logs.")

# 2. Helper Functions
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/food.csv")
        return df
    except FileNotFoundError:
        st.error("File 'data/food.csv' not found.")
        return None

# 3. Main Logic
df = load_data()

if df is not None:
    tab1, tab2 = st.tabs(["📊 Data Overview", "🔮 Prediction"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.write(f"Total Records: {len(df)}")
        
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_col = st.selectbox("Select feature to visualize", numeric_cols)
        st.bar_chart(df[selected_col].value_counts().head(20))

    with tab2:
        st.subheader("Predict Delivery Time")
        
        # User Input Form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Delivery Person Age", 18, 50, 30)
                rating = st.slider("Rating", 1.0, 5.0, 4.5)
                weather = st.selectbox("Weather", ['Sunny', 'Windy', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog'])
                traffic = st.selectbox("Traffic Density", ['Low', 'Medium', 'High', 'Jam'])
            with col2:
                vehicle = st.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'electric_scooter'])
                prep_time = st.number_input("Prep Time (min)", 0, 60, 15)
                distance = st.number_input("Distance (km)", 0.0, 50.0, 5.0)
                hour = st.slider("Hour of Day", 0, 23, 12)
            
            submit = st.form_submit_button("Get Estimate")

        if submit:
            # Create DataFrame for input
            input_data = pd.DataFrame({
                'Delivery_person_Age': [age],
                'Delivery_person_Ratings': [rating],
                'Weather': [weather],
                'Road_traffic_density': [traffic],
                'Type_of_vehicle': [vehicle],
                'prep_time': [prep_time],
                'distance_km': [distance],
                'Hour': [hour]
            })
            
            # Logic for prediction would go here. 
            # Since we are not loading the huge model, we simulate a response:
            st.success(f"Calculation Complete! Based on your inputs, the estimated time is approximately {int(distance * 3 + prep_time)} minutes.")
            st.info("Note: Connect your custom pipeline here to use your trained regressor.")

else:
    st.warning("Please place your 'food.csv' file in the 'data/' directory.")
