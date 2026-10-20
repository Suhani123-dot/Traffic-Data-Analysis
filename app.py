import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.sidebar.title("Smart City Traffic Analytics")
st.sidebar.write("Machine Learning Project")
st.sidebar.markdown("---")
st.sidebar.write("Developed Using")
st.sidebar.write("✓ Python")
st.sidebar.write("✓ Pandas")
st.sidebar.write("✓ Scikit-Learn")
st.sidebar.write("✓ Streamlit")

# Load Dataset - Auto path
csv_path = "data/traffic_data.csv" if os.path.exists("data/traffic_data.csv") else "traffic_data.csv"
data = pd.read_csv(csv_path)

# Encode data
le_location = LabelEncoder()
le_time = LabelEncoder()
le_congestion = LabelEncoder()

data["Location"] = le_location.fit_transform(data["Location"])
data["Time_of_Day"] = le_time.fit_transform(data["Time_of_Day"])
data["Congestion_Level"] = le_congestion.fit_transform(data["Congestion_Level"])

# Load or Train Model - No error anymore
try:
    model_path = "models/traffic_model.pkl" if os.path.exists("models/traffic_model.pkl") else "traffic_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError
except:
    X = data[["Location", "Vehicle_Count", "Average_Speed", "Time_of_Day"]]
    y = data["Congestion_Level"]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

st.title("🚦 Smart City Traffic Analytics System")

location = st.selectbox("Location", le_location.classes_)
vehicle_count = st.number_input("Vehicle Count", min_value=0)
average_speed = st.number_input("Average Speed", min_value=0)
time_of_day = st.selectbox("Time of Day", le_time.classes_)

if st.button("Predict"):
    loc_encoded = le_location.transform([location])[0]
    time_encoded = le_time.transform([time_of_day])[0]
    input_data = [[loc_encoded, vehicle_count, average_speed, time_encoded]]
    prediction = model.predict(input_data)
    congestion = le_congestion.inverse_transform(prediction)
    st.success(f"Predicted Congestion Level: {congestion[0]}")
