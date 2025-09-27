# file: synthetic_gps_simulator.py

import numpy as np
import pandas as pd
import asyncio
import websockets
import json
from datetime import datetime, timedelta

# ---------------------------
# 1️⃣ Synthetic Data Generator
# ---------------------------
def generate_synthetic_data(n_tourists=10, n_points=50):
    """Generate synthetic GPS data with zone risk and itinerary deviation."""
    data = []
    for t_id in range(1, n_tourists + 1):
        lat, lon = 17.721561383541516, 83.32403712490817  # Starting point
        for i in range(n_points):
            # Random walk for latitude & longitude
            lat += np.random.normal(0, 0.001)
            lon += np.random.normal(0, 0.001)

            # Simulate speed in m/s
            speed = np.random.uniform(0, 2)

            # Random zone risk based on fake zones
            zone_risk = np.random.choice([0, 1], p=[0.95, 0.05])

            # Random itinerary deviation
            itinerary_deviation = np.random.choice([0, 1], p=[0.90, 0.10])

            timestamp = datetime.utcnow() + timedelta(seconds=i*30)

            data.append({
                "tourist_id": t_id,
                "timestamp": timestamp,
                "lat": lat,
                "lon": lon,
                "speed": speed,
                "zone_risk": zone_risk,
                "itinerary_deviation": itinerary_deviation
            })
    df = pd.DataFrame(data)
    return df

# ---------------------------
# 2️⃣ LSTM Autoencoder Preprocessing
# ---------------------------
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

def preprocess_features(df):
    features = ["lat", "lon", "speed", "zone_risk", "itinerary_deviation"]
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled, scaler

def create_sequences(df_scaled, seq_len=10):
    features = ["lat", "lon", "speed", "zone_risk", "itinerary_deviation"]
    sequences = []
    for t_id, group in df_scaled.groupby("tourist_id"):
        group = group.sort_values("timestamp")
        for i in range(len(group) - seq_len):
            seq = group.iloc[i:i+seq_len][features].values
            sequences.append(seq)
    return np.array(sequences)

# ---------------------------
# 3️⃣ Build LSTM Autoencoder
# ---------------------------
def build_lstm_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, activation='relu', return_sequences=False)(inputs)
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------------
# 4️⃣ Anomaly Detection
# ---------------------------
def detect_anomalies(model, seq, threshold=0.01):
    pred = model.predict(seq[np.newaxis, ...], verbose=0)
    mse = np.mean((seq - pred[0])**2)
    anomaly = mse > threshold
    return anomaly, mse

# ---------------------------
# 5️⃣ Simulate Real-time WebSocket
# ---------------------------
async def simulate_realtime(df, ws_url="ws://localhost:8000/ws/gps"):
    for t_id, group in df.groupby("tourist_id"):
        group = group.sort_values("timestamp")
        for _, row in group.iterrows():
            async with websockets.connect(ws_url) as ws:
                msg = {
                    "tourist_id": row["tourist_id"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "speed": float(row["speed"]),
                    "itinerary_deviation": int(row["itinerary_deviation"])
                }
                await ws.send(json.dumps(msg))
                resp = await ws.recv()
                print(resp)
            await asyncio.sleep(0.5)  # simulate real-time interval

# ---------------------------
# 6️⃣ Main
# ---------------------------
import os
if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_data(n_tourists=5, n_points=30)

    # ✅ Keep only normal traces (zone_risk=0 and itinerary_deviation=0)
    normal_df = df[(df["zone_risk"] == 0) & (df["itinerary_deviation"] == 0)]

    # Preprocess and create sequences
    df_scaled, scaler = preprocess_features(normal_df)
    sequences = create_sequences(df_scaled, seq_len=10)

    # Train LSTM Autoencoder only on normal behavior
    model = build_lstm_autoencoder((10, 5))
    model.fit(sequences, sequences, epochs=5, batch_size=16, verbose=1)

    # Save normal model
    # Ensure `data/` folder exists
    os.makedirs("data", exist_ok=True)

    # Save model & normal traces
    model.save("models/lstm_autoencoder_normal.h5")
    normal_df.to_csv("data/normal_traces.csv", index=False)

    print("✅ Model trained on NORMAL traces only & saved dataset.")
    print("💾 Saved normal traces to data/normal_traces.csv")

    # Still send all tourists' traces (normal + abnormal) to test anomaly detection
    print("✅ Model trained. Starting WebSocket simulation...")
    asyncio.run(simulate_realtime(df))
