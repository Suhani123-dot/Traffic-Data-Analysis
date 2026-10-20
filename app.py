import streamlit as st
import pandas as pd
import os
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

# Load Dataset
csv_path = "data/traffic_data.csv" if os.path.exists("data/traffic_data.csv") else "traffic_data.csv"
data = pd.read_csv(csv_path)

# Create fresh encoders
le_location = LabelEncoder()
le_time = LabelEncoder()
le_congestion = LabelEncoder()

data["Location_enc"] = le_location.fit_transform(data["Location"])
data["Time_of_Day_enc"] = le_time.fit_transform(data["Time_of_Day"])
data["Congestion_enc"] = le_congestion.fit_transform(data["Congestion_Level"])

# Always train fresh model - No pkl error
X = data[["Location_enc", "Vehicle_Count", "Average_Speed", "Time_of_Day_enc"]]
y = data["Congestion_enc"]
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

st.title("🚦 Smart City Traffic Analytics System")

location = st.selectbox("Location", le_location.classes_)
vehicle_count = st.number_input("Vehicle Count", min_value=0, value=50)
average_speed = st.number_input("Average Speed", min_value=0, value=30)
time_of_day = st.selectbox("Time of Day", le_time.classes_)

if st.button("Predict"):
    loc_encoded = le_location.transform([location])[0]
    time_encoded = le_time.transform([time_of_day])[0]
    input_data = [[loc_encoded, vehicle_count, average_speed, time_encoded]]
    prediction = model.predict(input_data)
    congestion = le_congestion.inverse_transform(prediction)
    st.success(f"✅ Predicted Congestion Level: {congestion[0]}")
    st.balloons()
