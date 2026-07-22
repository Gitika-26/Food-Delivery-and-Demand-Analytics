"""
Food Delivery Ops Toolkit
--------------------------
Interactive Streamlit app with three tools built on top of the models
trained in FinalProject.ipynb:

  1. ETA Prediction        -> models/eta_model.joblib            (optional - see note below)
  2. Surge Fee Calculator  -> models/zone_cluster_model.joblib
                               models/demand_model.joblib + demand_features.joblib
                               models/zone_thresholds.json
  3. Driver Insights       -> models/driver_cluster_model.joblib

NOTE ON eta_model.joblib
-------------------------
This file was intentionally left out of the repo (too large for a normal
git push). The app still works without it: the ETA tab will detect the
missing file and either (a) let you train a small, GitHub-friendly
version on the spot if `data/food.csv` is present in the repo, or
(b) show instructions for adding the full model back via Git LFS.
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
DATA_PATH = os.path.join("data", "food.csv")

ETA_MODEL_PATH = os.path.join(MODEL_DIR, "eta_model.joblib")
ZONE_MODEL_PATH = os.path.join(MODEL_DIR, "zone_cluster_model.joblib")
DEMAND_MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.joblib")
DEMAND_FEATURES_PATH = os.path.join(MODEL_DIR, "demand_features.joblib")
ZONE_THRESHOLDS_PATH = os.path.join(MODEL_DIR, "zone_thresholds.json")
DRIVER_MODEL_PATH = os.path.join(MODEL_DIR, "driver_cluster_model.joblib")

st.set_page_config(
    page_title="Food Delivery Ops Toolkit",
    page_icon="🛵",
    layout="wide",
)

# --------------------------------------------------------------------------
# Loaders (cached so models are only read from disk once per session)
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


@st.cache_resource(show_spinner="Training a lightweight ETA model...")
def train_lightweight_eta_model():
    """Fallback trainer used only when eta_model.joblib is missing but the
    raw dataset is available. Mirrors the feature engineering + preprocessing
    from the notebook, but caps the RandomForest size so the resulting
    .joblib file is small enough to commit to GitHub normally."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor

    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)
    data = df.copy()

    def haversine_distance(d):
        R = 6371.0
        lat1, lon1 = np.radians(d["Restaurant_latitude"]), np.radians(d["Restaurant_longitude"])
        lat2, lon2 = np.radians(d["Delivery_location_latitude"]), np.radians(d["Delivery_location_longitude"])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

    data["distance_km"] = haversine_distance(data)

    str_cols = ["Road_traffic_density", "Type_of_order", "Type_of_vehicle", "City", "Weather"]
    for col in str_cols:
        data[col] = data[col].astype(str).str.strip()

    data["Order_Date"] = pd.to_datetime(data["Order_Date"], dayfirst=True)
    data["Order_day_of_week"] = data["Order_Date"].dt.dayofweek
    data["Time_Ordered"] = pd.to_datetime(data["Time_Ordered"], errors="coerce")
    data["Time_Order_picked"] = pd.to_datetime(data["Time_Order_picked"], errors="coerce")
    data["prep_time"] = (data["Time_Order_picked"] - data["Time_Ordered"]).dt.total_seconds() / 60
    data["prep_time"] = data["prep_time"].fillna(data["prep_time"].median())
    data["Hour"] = data["Time_Ordered"].dt.hour
    data["is_rush_hour"] = data["Hour"].apply(lambda x: 1 if x in [8, 9, 12, 13, 19, 20, 21] else 0)

    data = data.drop(columns=["ID", "Delivery_person_ID", "Time_Order_picked", "Time_Ordered", "Order_Date"],
                      errors="ignore")

    X = data.drop(["Time_taken"], axis=1)
    y = data["Time_taken"]
    numeric_features = [c for c in X.columns if c not in str_cols]

    categories = [
        ["Low", "Medium", "High", "Jam"],
        ["Drinks", "Snack", "Meal", "Buffet"],
        ["motorcycle", "scooter", "electric_scooter"],
        ["Semi-Urban", "Urban", "Metropolitian"],
        ["Sunny", "Windy", "Stormy", "Sandstorms", "Cloudy", "Fog"],
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1), str_cols),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Small model on purpose: keeps the pickled file lightweight for GitHub.
    small_model = RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", small_model)])
    pipeline.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, ETA_MODEL_PATH)
    return pipeline


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


# --------------------------------------------------------------------------
# Load everything up front
# --------------------------------------------------------------------------
eta_pipeline = load_joblib(ETA_MODEL_PATH)
zone_bundle = load_joblib(ZONE_MODEL_PATH)
demand_pipeline = load_joblib(DEMAND_MODEL_PATH)
demand_features = load_joblib(DEMAND_FEATURES_PATH)
zone_thresholds = load_json(ZONE_THRESHOLDS_PATH)
driver_bundle = load_joblib(DRIVER_MODEL_PATH)

# --------------------------------------------------------------------------
# Sidebar - model status
# --------------------------------------------------------------------------
st.sidebar.title("🛵 Ops Toolkit")
st.sidebar.caption("Model status")


def status_line(label, obj):
    st.sidebar.markdown(f"{'✅' if obj is not None else '❌'} {label}")


status_line("ETA model", eta_pipeline)
status_line("Zone clustering model", zone_bundle)
status_line("Demand model", demand_pipeline)
status_line("Zone thresholds", zone_thresholds)
status_line("Driver clustering model", driver_bundle)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Models are read from the `models/` folder next to this app. "
    "If a file shows ❌, drop the matching `.joblib`/`.json` file from your "
    "training notebook into that folder and reload."
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🛵 Food Delivery Ops Toolkit")
st.caption("ETA prediction, dynamic surge pricing, and driver segmentation — powered by the models from FinalProject.ipynb")

tab_eta, tab_surge, tab_driver = st.tabs(["🕒 ETA Prediction", "⚡ Surge Fee Calculator", "🧑‍✈️ Driver Insights"])

# ==========================================================================
# TAB 1 - ETA PREDICTION
# ==========================================================================
with tab_eta:
    st.subheader("Estimated Delivery Time")

    if eta_pipeline is None:
        st.warning(
            "`models/eta_model.joblib` isn't in the repo yet (it was left out for being too large to push to GitHub normally)."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Option 1 — Train a small version now**")
            st.write("If `data/food.csv` is also in the repo, you can train a compact ETA model right here (fewer trees, capped depth) that's small enough to commit.")
            if os.path.exists(DATA_PATH):
                if st.button("Train lightweight ETA model"):
                    pipeline = train_lightweight_eta_model()
                    if pipeline is not None:
                        st.success("Trained and saved to models/eta_model.joblib — reload the page to use it.")
                    else:
                        st.error("Training failed — check that data/food.csv matches the expected schema.")
            else:
                st.info("`data/food.csv` not found in the repo, so this option isn't available.")
        with col_b:
            st.markdown("**Option 2 — Add the original file**")
            st.write(
                "Use [Git LFS](https://git-lfs.com/) to push the original `eta_model.joblib` "
                "(GitHub's normal file-size limit is 100 MB, LFS handles larger files):"
            )
            st.code("git lfs install\ngit lfs track \"models/*.joblib\"\ngit add .gitattributes models/eta_model.joblib\ngit commit -m \"Add ETA model via LFS\"\ngit push", language="bash")
    else:
        try:
            preprocessor = eta_pipeline.named_steps.get("preprocessor")
            all_cols = list(preprocessor.feature_names_in_)
            cat_cols, num_cols, cat_categories = [], [], {}
            for name, trans, cols in preprocessor.transformers_:
                if name == "ord":
                    cat_cols = list(cols)
                    for c, cats in zip(cols, trans.categories_):
                        cat_categories[c] = list(cats)
                elif name == "num":
                    num_cols = list(cols)
        except Exception:
            st.error("Couldn't read the structure of the ETA pipeline. It may have been saved in a different format than expected.")
            all_cols, cat_cols, num_cols, cat_categories = [], [], [], {}

        # Fields the app derives automatically rather than asking twice for
        derived_fields = {"distance_km", "Hour", "Order_day_of_week", "is_rush_hour"}

        st.markdown("#### Trip details")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pickup (restaurant)**")
            r_lat = st.number_input("Restaurant latitude", value=12.9716, format="%.6f", key="r_lat")
            r_lon = st.number_input("Restaurant longitude", value=77.5946, format="%.6f", key="r_lon")
        with c2:
            st.markdown("**Drop-off (customer)**")
            d_lat = st.number_input("Delivery latitude", value=12.9352, format="%.6f", key="d_lat")
            d_lon = st.number_input("Delivery longitude", value=77.6146, format="%.6f", key="d_lon")

        order_dt = st.datetime_input("Order placed at", value=datetime.now()) if hasattr(st, "datetime_input") else None
        if order_dt is None:
            oc1, oc2 = st.columns(2)
            order_date = oc1.date_input("Order date")
            order_time = oc2.time_input("Order time", value=time(12, 0))
        else:
            order_date, order_time = order_dt.date(), order_dt.time()

        st.markdown("#### Order & rider details")
        user_values = {}
        cols_left, cols_right = st.columns(2)
        toggled = [c for c in num_cols if c not in derived_fields]

        for i, col in enumerate(toggled):
            target = cols_left if i % 2 == 0 else cols_right
            label = col.replace("_", " ")
            if col == "prep_time":
                user_values[col] = target.slider("Kitchen prep time so far (minutes)", 0, 60, 15, key=col)
            elif "Rating" in col:
                user_values[col] = target.slider(label, 1.0, 5.0, 4.5, 0.1, key=col)
            elif "Age" in col:
                user_values[col] = target.number_input(label, min_value=15, max_value=70, value=30, key=col)
            elif "condition" in col.lower():
                user_values[col] = target.slider(label, 0, 3, 1, key=col)
            elif "multiple" in col.lower():
                user_values[col] = target.number_input(label, min_value=0, max_value=5, value=0, key=col)
            elif col in ("Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"):
                continue  # already collected above
            else:
                user_values[col] = target.number_input(label, value=0.0, key=col)

        for col in cat_cols:
            target = cols_left if len(user_values) % 2 == 0 else cols_right
            options = cat_categories.get(col, ["Unknown"])
            user_values[col] = target.selectbox(col.replace("_", " "), options, key=col)

        if st.button("Predict ETA", type="primary"):
            row = {c: 0 for c in all_cols}
            row.update(user_values)
            row["Restaurant_latitude"] = r_lat
            row["Restaurant_longitude"] = r_lon
            row["Delivery_location_latitude"] = d_lat
            row["Delivery_location_longitude"] = d_lon
            if "distance_km" in all_cols:
                row["distance_km"] = haversine_km(r_lat, r_lon, d_lat, d_lon)
            if "Hour" in all_cols:
                row["Hour"] = order_time.hour
            if "Order_day_of_week" in all_cols:
                row["Order_day_of_week"] = order_date.weekday()
            if "is_rush_hour" in all_cols:
                row["is_rush_hour"] = 1 if order_time.hour in [8, 9, 12, 13, 19, 20, 21] else 0

            input_df = pd.DataFrame([row])[all_cols]
            try:
                pred = eta_pipeline.predict(input_df)[0]
                st.success(f"### Estimated delivery time: **{pred:.1f} minutes**")
                st.caption(f"Straight-line distance: {haversine_km(r_lat, r_lon, d_lat, d_lon):.2f} km")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ==========================================================================
# TAB 2 - SURGE FEE CALCULATOR
# ==========================================================================
with tab_surge:
    st.subheader("Dynamic Surge Fee Calculator")

    missing = [n for n, o in [
        ("zone clustering model", zone_bundle),
        ("demand model", demand_pipeline),
        ("demand feature list", demand_features),
        ("zone thresholds", zone_thresholds),
    ] if o is None]

    if missing:
        st.warning("Missing: " + ", ".join(missing) + ". Make sure these files are in the `models/` folder.")
    else:
        st.markdown("#### Where is the order?")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=12.9716, format="%.6f")
        lon = c2.number_input("Longitude", value=77.5946, format="%.6f")

        st.markdown("#### When?")
        c3, c4 = st.columns(2)
        chosen_date = c3.date_input("Date", value=datetime.now().date(), key="surge_date")
        chosen_time = c4.time_input("Time", value=time(19, 0), key="surge_time")

        st.markdown("#### Recent activity in this area")
        st.caption("In production these would come from a live feed; enter your best estimates to try the calculator.")
        c5, c6, c7 = st.columns(3)
        demand_last_hour = c5.number_input("Orders in the last hour", min_value=0, value=10)
        demand_yesterday = c6.number_input("Orders this hour yesterday", min_value=0, value=8)
        traffic_last_hour = c7.number_input("Avg delivery time last hour (min)", min_value=0.0, value=30.0)

        base_fee = st.number_input("Base delivery fee (₹)", min_value=0.0, value=40.0, step=5.0)
        sensitivity = st.slider(
            "Surge sensitivity", 0.1, 2.0, 0.5, 0.1,
            help="How aggressively the fee scales once predicted demand exceeds the zone's typical capacity."
        )

        if st.button("Calculate surge fee", type="primary"):
            try:
                scaler = zone_bundle["scaler"]
                kmeans = zone_bundle["model"]
                zone_id = int(kmeans.predict(scaler.transform([[lat, lon]]))[0])

                feature_row = {
                    "Zone_ID": zone_id,
                    "Hour": chosen_time.hour,
                    "Demand_Last_Hour": demand_last_hour,
                    "Demand_Yesterday_Same_Hour": demand_yesterday,
                    "Traffic_Last_Hour": traffic_last_hour,
                    "Order_day_of_week": chosen_date.weekday(),
                }
                X = pd.DataFrame([feature_row])[demand_features]
                predicted_demand = max(0.0, float(demand_pipeline.predict(X)[0]))

                threshold = zone_thresholds.get(str(zone_id))
                if threshold is None:
                    threshold = float(np.median(list(zone_thresholds.values())))
                    st.info(f"Zone {zone_id} has no historical threshold yet — using the citywide median as a fallback.")

                ratio = predicted_demand / threshold if threshold > 0 else 1.0
                multiplier = 1.0 if ratio <= 1 else min(1 + (ratio - 1) * sensitivity, 3.0)
                final_fee = base_fee * multiplier

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Zone", f"#{zone_id}")
                m2.metric("Predicted demand", f"{predicted_demand:.1f} orders")
                m3.metric("Zone capacity (typical)", f"{threshold:.1f} orders")
                m4.metric("Surge multiplier", f"{multiplier:.2f}x")

                st.markdown(f"## Final delivery fee: ₹{final_fee:.2f}")

                fig = go.Figure()
                fig.add_trace(go.Bar(x=["Typical capacity", "Predicted demand"],
                                      y=[threshold, predicted_demand],
                                      marker_color=["#4C78A8", "#E45756" if ratio > 1 else "#54A24B"]))
                fig.update_layout(title="Predicted demand vs. typical zone capacity", yaxis_title="Orders", height=350)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Couldn't calculate the surge fee: {e}")

# ==========================================================================
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
