"""
Generate synthetic IoT cargo logistics dataset for wear prediction.
Schema: Timestamp, DeviceID, Speed(kmph), Pressure(psi), Temperature(°C),
        Latitude, Longitude, Wear(mm), Status, Obs_Obj, Collision, Type
Dependent variable: Wear(mm)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)

np.random.seed(42)

N = 5000  # number of samples
devices = [f"D{i:03d}" for i in range(1, 21)]
types = ["Container", "Trailer", "Tanker", "Flatbed"]

start_ts = datetime(2024, 1, 1)
timestamps = [start_ts + timedelta(minutes=i * 15) for i in range(N)]

device_ids = np.random.choice(devices, N)
speed = np.clip(np.random.lognormal(2.5, 0.8, N), 5, 120)
pressure = np.clip(np.random.normal(45, 12, N), 10, 90)
temperature = np.clip(np.random.normal(35, 15, N), -10, 80)
lat = np.random.uniform(12.0, 32.0, N)
lon = np.random.uniform(77.0, 88.0, N)
obs_obj = np.random.choice([0, 1], N, p=[0.85, 0.15])
collision = np.random.choice([0, 1], N, p=[0.97, 0.03])
type_ = np.random.choice(types, N)

# Wear (mm): influenced by speed, pressure, temperature, collision, obs_obj
wear_base = 0.5
wear_speed = 0.02 * (speed - 40).clip(0, None)
wear_pressure = 0.015 * np.abs(pressure - 50)
wear_temp = 0.01 * np.abs(temperature - 40)
wear_collision = 2.0 * collision
wear_obs = 0.3 * obs_obj
noise = np.random.normal(0, 0.3, N)
wear = np.clip(
    wear_base + wear_speed + wear_pressure + wear_temp + wear_collision + wear_obs + noise,
    0.1, 8.0
)

# Status: Normal (< 2), Warning [2, 4), Critical >= 4
status = np.where(wear < 2, "Normal", np.where(wear < 4, "Warning", "Critical"))

df = pd.DataFrame({
    "Timestamp": timestamps,
    "DeviceID": device_ids,
    "Speed_kmph": speed,
    "Pressure_psi": pressure,
    "Temperature_C": temperature,
    "Latitude": lat,
    "Longitude": lon,
    "Wear_mm": wear,
    "Status": status,
    "Obs_Obj": obs_obj,
    "Collision": collision,
    "Type": type_,
})

df.to_csv("data/iot_cargo_dataset.csv", index=False)
print(f"Saved {len(df)} rows to data/iot_cargo_dataset.csv")
print(df.describe())
