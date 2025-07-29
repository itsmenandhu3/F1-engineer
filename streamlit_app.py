import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('model.pkl')
le_driver = joblib.load('le_driver.pkl')
le_compound = joblib.load('le_compound.pkl')

st.title("F1 Qualifying Lap Time Predictor (Pole Simulator)")

# Prediction UI Inputs
driver = st.selectbox("Driver", le_driver.classes_)
compound = st.selectbox("Tyre Compound", le_compound.classes_)
# Qualifying: usually lap number very low (often 1-3), tyre_age always 0
lap_number = st.number_input("Lap Number", min_value=1, max_value=5, value=1)
tyre_age = 0
air_temp = st.slider("Air Temperature (°C)", 10, 50, 25)
track_temp = st.slider("Track Temperature (°C)", 20, 70, 40)

driver_encoded = le_driver.transform([driver])[0]
compound_encoded = le_compound.transform([compound])[0]

features = np.array([[lap_number, tyre_age, driver_encoded, compound_encoded, air_temp, track_temp]])
pred_ms = model.predict(features)[0]

def format_time(ms):
    total_seconds = int(ms // 1000)
    milliseconds = int(ms % 1000)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}.{seconds:02d}.{milliseconds:03d}"

formatted_time = format_time(pred_ms)
st.write(f"### Predicted Qualifying Lap: {formatted_time} (min.sec.ms)")
