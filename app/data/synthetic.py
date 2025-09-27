# file: ai_anomaly_detection.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# -------------------------
# 1️⃣ Synthetic Data Generator
# -------------------------
def generate_synthetic_data(n_tourists=50, n_points=100):
    data = []
    for t_id in range(1, n_tourists + 1):
        lat, lon = 12.9716, 77.5946  # start in Bangalore, India
        for i in range(n_points):
            timestamp = datetime.utcnow() + timedelta(seconds=i * 60)
            speed = np.random.uniform(0, 10)  # m/s
            # simulate slight drift
            lat += np.random.normal(0, 0.0005)
            lon += np.random.normal(0, 0.0005)
            zone_risk = np.random.choice([0, 0.2, 0.5, 0.8], p=[0.7, 0.1, 0.1, 0.1])
            itinerary_deviation = np.random.uniform(0, 50)  # meters
            data.append([t_id, timestamp, lat, lon, speed, zone_risk, itinerary_deviation])
    df = pd.DataFrame(data, columns=['tourist_id', 'timestamp', 'lat', 'lon', 'speed', 'zone_risk', 'itinerary_deviation'])
    return df

# -------------------------
# 2️⃣ Feature Preprocessing
# -------------------------
def preprocess_features(df):
    df_sorted = df.sort_values(['tourist_id', 'timestamp'])
    features = ['lat', 'lon', 'speed', 'zone_risk', 'itinerary_deviation']
    scaler = MinMaxScaler()
    df_scaled = df_sorted.copy()
    df_scaled[features] = scaler.fit_transform(df_sorted[features])
    return df_scaled, scaler

# -------------------------
# 3️⃣ Create Sequences for LSTM
# -------------------------
def create_sequences(df_scaled, seq_len=10):
    sequences = []
    for t_id, group in df_scaled.groupby('tourist_id'):
        group = group.sort_values('timestamp')
        for i in range(len(group) - seq_len):
            seq = group.iloc[i:i+seq_len][['lat','lon','speed','zone_risk','itinerary_deviation']].values
            sequences.append(seq)
    return np.array(sequences)

# -------------------------
# 4️⃣ Build LSTM Autoencoder
# -------------------------
def build_lstm_autoencoder(seq_len=10, n_features=5, latent_dim=32):
    inputs = Input(shape=(seq_len, n_features))
    encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(inputs)
    decoded = RepeatVector(seq_len)(encoded)
    decoded = LSTM(n_features, activation='relu', return_sequences=True)(decoded)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# -------------------------
# 5️⃣ Train the model
# -------------------------
def train_model(sequences, epochs=20, batch_size=32):
    n_seq, seq_len, n_features = sequences.shape
    model = build_lstm_autoencoder(seq_len, n_features)
    model.fit(sequences, sequences, epochs=epochs, batch_size=batch_size, validation_split=0.1, shuffle=True)
    return model

# -------------------------
# 6️⃣ Detect Anomalies
# -------------------------
def detect_anomalies(model, sequences, threshold=None):
    reconstructions = model.predict(sequences)
    mse = np.mean((sequences - reconstructions)**2, axis=(1,2))
    if threshold is None:
        threshold = np.percentile(mse, 95)  # top 5% as anomalies
    anomalies = mse > threshold
    return anomalies, mse, threshold

# -------------------------
# 7️⃣ Simulate Real-time Feed
# -------------------------
def simulate_realtime(df_scaled, model, seq_len=10, threshold=None):
    alerts = []
    for t_id, group in df_scaled.groupby('tourist_id'):
        group = group.sort_values('timestamp')
        if len(group) < seq_len:
            continue
        for i in range(len(group) - seq_len):
            seq = group.iloc[i:i+seq_len][['lat','lon','speed','zone_risk','itinerary_deviation']].values
            seq = seq[np.newaxis, ...]
            anomaly, mse, _ = detect_anomalies(model, seq, threshold)
            if anomaly[0]:
                alerts.append({
                    'tourist_id': t_id,
                    'timestamp': group.iloc[i+seq_len]['timestamp'],
                    'mse': float(mse[0]),
                    'alert': '🚨 Anomaly detected – possible distress or deviation'
                })
    return pd.DataFrame(alerts)

# -------------------------
# 8️⃣ Example Usage
# -------------------------
if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    df_scaled, scaler = preprocess_features(df)
    sequences = create_sequences(df_scaled)
    print(f"Sequences shape: {sequences.shape}")

    print("Training LSTM Autoencoder...")
    model = train_model(sequences, epochs=10)  # fewer epochs for demo

    print("Simulating real-time detection...")
    alerts_df = simulate_realtime(df_scaled, model)
    print(alerts_df.head())
