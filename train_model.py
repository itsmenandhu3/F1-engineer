import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# Enable FastF1 cache (ensure 'cache' directory exists)
fastf1.Cache.enable_cache('cache')

# ----- 1. Load Qualifying Session -----
session = fastf1.get_session(2024, 'Monaco Grand Prix', 'Q')
session.load()

# Only quicklaps
laps = session.laps.pick_quicklaps().reset_index()

# ----- 2. One Best Qualifying Lap per Driver -----
best_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()].reset_index(drop=True)
best_laps = best_laps[best_laps['Compound'].notna()]

# ----- 3. Feature Engineering -----
le_driver = LabelEncoder()
le_compound = LabelEncoder()
best_laps['driver_encoded'] = le_driver.fit_transform(best_laps['Driver'])
best_laps['compound_encoded'] = le_compound.fit_transform(best_laps['Compound'])
best_laps['tyre_age'] = 0  # Quali: always fresh tyres!

# ----- 4. Merge Weather Data -----
weather = session.weather_data.reset_index()
# Print columns for debugging
print("Weather columns:", weather.columns)

best_laps['Timestamp_unix'] = best_laps['Time'].astype(np.int64) // 10 ** 9
weather['Timestamp_unix'] = weather['Time'].astype(np.int64) // 10 ** 9

def merge_nearest_weather(laps_df, weather_df, time_col='Timestamp_unix'):
    weather_times = weather_df['Timestamp_unix'].values
    air_temps = weather_df['AirTemp'].values  # Use actual column names
    track_temps = weather_df['TrackTemp'].values
    air_temp_list = []
    track_temp_list = []
    for t in laps_df[time_col]:
        idx = (np.abs(weather_times - t)).argmin()
        air_temp_list.append(air_temps[idx])
        track_temp_list.append(track_temps[idx])
    laps_df['AirTemp'] = air_temp_list
    laps_df['TrackTemp'] = track_temp_list
    return laps_df

best_laps = merge_nearest_weather(best_laps, weather)

# ----- 5. Prepare Data for ML -----
features = ['LapNumber', 'tyre_age', 'driver_encoded', 'compound_encoded', 'AirTemp', 'TrackTemp']
best_laps = best_laps.dropna(subset=features + ['LapTime'])

X = best_laps[features]
y = best_laps['LapTime'].apply(lambda x: x.total_seconds() * 1000)  # ms

# Split (train split is small but works for demo, real projects use more years!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ----- 6. Train and Save Model -----
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"Mean Absolute Error: {mae:.2f} ms")

joblib.dump(model, 'model.pkl')
joblib.dump(le_driver, 'le_driver.pkl')
joblib.dump(le_compound, 'le_compound.pkl')

print("Qualifying model and encoders saved for Streamlit app.")
