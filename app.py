import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.sidebar.title("Smart City Traffic Analytics")
st.sidebar.write("Machine Learning Project")
st.sidebar.markdown("---")
st.sidebar.write("Developed Using")
st.sidebar.write("✔ Python")
st.sidebar.write("✔ Pandas")
st.sidebar.write("✔ Scikit-Learn")
st.sidebar.write("✔ Streamlit")
# Load Dataset
import os
csv_path = "data/traffic_data.csv" if os.path.exists("data/traffic_data.csv") else "traffic_data.csv"
data = pd.read_csv(csv_path)
# Encode data
le_location = LabelEncoder()
le_time = LabelEncoder()
le_congestion = LabelEncoder()

data["Location"] = le_location.fit_transform(data["Location"])
data["Time_of_Day"] = le_time.fit_transform(data["Time_of_Day"])
data["Congestion_Level"] = le_congestion.fit_transform(data["Congestion_Level"])

# Load Model
model = joblib.load("models/traffic_model.pkl")

st.title("🚦 Smart City Traffic Analytics System")

location = st.selectbox("Location", le_location.classes_)
vehicle_count = st.number_input("Vehicle Count", min_value=0)
average_speed = st.number_input("Average Speed", min_value=0)
time = st.selectbox("Time of Day", le_time.classes_)

if st.button("Predict"):
    location_value = le_location.transform([location])[0]
    time_value = le_time.transform([time])[0]

    prediction = model.predict([[location_value, vehicle_count, average_speed, time_value]])

    congestion = le_congestion.inverse_transform([round(prediction[0])])[0]

    st.success(f"Predicted Congestion Level: {congestion}")
    st.subheader("Traffic Dataset")
st.dataframe(pd.read_csv("data/traffic_data.csv"))
st.subheader("Vehicle Count Chart")
st.bar_chart(data["Vehicle_Count"])
st.subheader("Average Speed Chart")
st.line_chart(data["Average_Speed"])
st.subheader("Traffic Dataset")
st.dataframe(pd.read_csv("data/traffic_data.csv"))

st.subheader("Vehicle Count Chart")
st.bar_chart(data["Vehicle_Count"])

st.subheader("Average Speed Chart")
st.line_chart(data["Average_Speed"])

st.subheader("Congestion Level Distribution")
st.bar_chart(data["Congestion_Level"])
st.subheader("Dataset Information")
st.write(data.describe())
st.subheader("Vehicle Count")
st.bar_chart(data["Vehicle_Count"])
st.subheader("Average Speed")
st.line_chart(data["Average_Speed"])
st.subheader("Traffic Dataset")
st.dataframe(data)
st.subheader("Congestion Level")
st.bar_chart(data["Congestion_Level"])
st.subheader("Dataset Information")
st.write(data.describe())
st.subheader("Congestion Level Pie Chart")

fig, ax = plt.subplots()

data["Congestion_Level"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax
)

st.pyplot(fig)
st.subheader("Vehicle Count vs Average Speed")

chart_data = data[["Vehicle_Count", "Average_Speed"]]

st.scatter_chart(chart_data)
csv = data.to_csv(index=False)

st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="traffic_data.csv",
    mime="text/csv"
)
st.markdown("---")

st.header("Project Information")

st.write("""
**Project Name:** Smart City Traffic Analytics System

**Machine Learning Model:** Random Forest Regression

**Technologies Used:**
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
""")
st.markdown("---")
st.caption("Developed for Machine Learning Project")
st.markdown("---")
st.subheader("Project Statistics")

st.write("Total Records:", len(data))
st.write("Average Vehicle Count:", data["Vehicle_Count"].mean())
st.write("Average Speed:", data["Average_Speed"].mean())
st.subheader("Highest Traffic Location")

highest = data.loc[data["Vehicle_Count"].idxmax()]

st.write(highest)
st.subheader("Lowest Traffic Location")

lowest = data.loc[data["Vehicle_Count"].idxmin()]

st.write(lowest)
st.subheader("Complete Traffic Dataset")

st.dataframe(data)
st.markdown("""
This application predicts traffic congestion using a Machine Learning model.
Enter the traffic details below and click **Predict**.
""")
