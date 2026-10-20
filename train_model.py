import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("data/traffic_data.csv")

# Label Encoding
le_location = LabelEncoder()
le_time = LabelEncoder()
le_congestion = LabelEncoder()

data["Location"] = le_location.fit_transform(data["Location"])
data["Time_of_Day"] = le_time.fit_transform(data["Time_of_Day"])
data["Congestion_Level"] = le_congestion.fit_transform(data["Congestion_Level"])

# Features and Target
X = data[["Location", "Vehicle_Count", "Average_Speed", "Time_of_Day"]]
y = data["Congestion_Level"]

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save Model
joblib.dump(model, "models/traffic_model.pkl")

print("Model Trained Successfully")