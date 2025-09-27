# file: ai_realtime_server.py

from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from app.utils.harvesine import haversine
from fastapi.middleware.cors import CORSMiddleware
from app.data.synthetic_gps_simulator import generate_synthetic_data

# ---------------------------
# 1️⃣ FastAPI App
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 2️⃣ Firebase Initialization
# ---------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("app/data/serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://smart-tourist-safety-monitor-default-rtdb.firebaseio.com"
    })

# ---------------------------
# 3️⃣ In-Memory Storage
# ---------------------------
push_tokens = {}
last_positions = {}
last_risk_levels = {}
zone_risks = {}

# ---------------------------
# 4️⃣ Load Geofencing Zones
# ---------------------------
def load_geofencing_zones():
    ref = db.reference("geofencing_zones")
    zones = ref.get() or {}
    zones_list = []
    for zone_id, z in zones.items():
        zones_list.append({
            "name": z["name"],
            "lat": z["coordinates"][1],
            "lon": z["coordinates"][0],
            "radius": z["radius"],
            "description": z.get("description", "")
        })
    return zones_list

zone_risks = load_geofencing_zones()

# ---------------------------
# 5️⃣ Features & Scaling
# ---------------------------
features = ["lat", "lon", "speed", "zone_risk", "itinerary_deviation"]
scaler = MinMaxScaler()

# ---------------------------
# 6️⃣ Synthetic Data & Scaler Fit
# ---------------------------
print("🔹 Generating synthetic data for scaler fit...")
train_df = generate_synthetic_data(n_tourists=20, n_points=100)
scaler.fit(train_df[features])
print("✅ Scaler fitted on synthetic data")

# ---------------------------
# 7️⃣ LSTM Autoencoder
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

seq_len = 10
model = build_lstm_autoencoder((seq_len, len(features)))

# ---------------------------
# 8️⃣ E-FIR Generator
# ---------------------------
def generate_efir(tourist_id: str, timestamp: datetime, reason: str, lat=None, lon=None):
    ref = db.reference(f"efir/{tourist_id}")
    efir_data = {
        "tourist_id": tourist_id,
        "timestamp": timestamp.isoformat(),
        "reason": reason,
        "latitude": lat,
        "longitude": lon,
        "status": "Pending",
    }
    efir_key = ref.push(efir_data).key
    print(f"🚨 E-FIR generated for tourist {tourist_id} | Reason: {reason} | Key: {efir_key}")
    return efir_key

# ---------------------------
# 9️⃣ Zone Risk Calculator
# ---------------------------
def check_zone_risk(lat, lon):
    risk = 0
    for z in zone_risks:
        distance = haversine(lat, lon, z["lat"], z["lon"])
        if distance <= z["radius"]:
            risk = 1
            break
    return risk

# ---------------------------
# 🔟 Anomaly Detector
# ---------------------------
def detect_anomalies(seq, threshold=0.01):
    seq_scaled = scaler.transform(seq.reshape(-1, len(features))).reshape(seq.shape)
    reconstructed = model.predict(seq_scaled[np.newaxis, ...], verbose=0)
    mse = np.mean((seq_scaled - reconstructed[0])**2)
    return mse > threshold, mse


@app.get("/")
def root():
    return {"message": "Smart Tourist Safety Monitor API running"}

# --------------------------------
# 1️⃣1️⃣ Fetch Tourist Safety Score
# --------------------------------
from fastapi import HTTPException
from fastapi import Query

@app.get("/tourist/safety_score")
def get_tourist_safety_score(blockchain_id: str = Query(...), lat: float = Query(...), lon: float = Query(...)):
    """
    Fetch tourist info by blockchainId (first trip), compute safety score
    """
    # Search tourists/trips for the blockchainId
    tourists_ref = db.reference("tourists")
    tourists = tourists_ref.get() or {}
    tourist_info = None
    for t_id, t_data in tourists.items():
        trips = t_data.get("trips", {})
        for trip_id, trip_data in trips.items():
            if trip_data.get("blockchainId") == blockchain_id:
                tourist_info = t_data
                break
        if tourist_info:
            break

    if not tourist_info:
        raise HTTPException(status_code=404, detail="Tourist not found")

    # Compute zone risk
    zone_risk = check_zone_risk(lat, lon)
    # Safety score: simple example
    safety_score = 100 if zone_risk == 0 else 50

    return {
        "tourist_name": tourist_info.get("name", "Guest"),
        "emergency_contact": tourist_info.get("emergency_contact", "Not Available"),
        "safety_score": safety_score,
        "zone_risk": zone_risk
    }


# ---------------------------
# 1️⃣1️⃣ WebSocket Endpoint
# ---------------------------
@app.websocket("/ws/gps")
async def websocket_gps(websocket: WebSocket):
    await websocket.accept()
    tourist_id = None  # initialize before loop
    try:
        while True:
            data = await websocket.receive_json()
            tourist_id = str(data["tourist_id"])
            lat = data["lat"]
            lon = data["lon"]

            # Compute speed
            timestamp = pd.Timestamp(datetime.utcnow())
            if tourist_id in last_positions:
                prev_lat, prev_lon, prev_time = last_positions[tourist_id]
                delta_distance = haversine(prev_lat, prev_lon, lat, lon)
                delta_t = (timestamp - prev_time).total_seconds()
                speed = delta_distance / delta_t if delta_t > 0 else 0
            else:
                delta_distance = 0
                speed = 0
            last_positions[tourist_id] = (lat, lon, timestamp)

            # Zone risk & itinerary deviation
            zone_risk = check_zone_risk(lat, lon)
            itinerary_deviation = data.get("itinerary_deviation", 0)

            # Prepare sequence
            seq = np.array([[lat, lon, speed, zone_risk, itinerary_deviation]])
            if "seq_buffer" not in last_positions:
                last_positions["seq_buffer"] = {}
            buf = last_positions["seq_buffer"].get(tourist_id, [])
            buf.append(seq[0])
            if len(buf) > seq_len:
                buf = buf[-seq_len:]
            last_positions["seq_buffer"][tourist_id] = buf

            # Anomaly detection
            if len(buf) == seq_len:
                seq_array = np.array(buf)
                anomaly, mse = detect_anomalies(seq_array)
            else:
                anomaly = False
                mse = 0

            risk_level = "🚨 Unsafe" if anomaly or zone_risk else "✅ Safe"
            alerts = []
            if anomaly:
                alerts.append("Anomaly detected – possible distress, missing or deviation")
                generate_efir(tourist_id, timestamp, "Anomaly detected", lat, lon)

            # Save to Firebase
            ref = db.reference(f"gps_updates/{tourist_id}")
            ref.push({
                "lat": lat,
                "lon": lon,
                "speed": speed,
                "zone_risk": zone_risk,
                "itinerary_deviation": itinerary_deviation,
                "safety_score": 100 if risk_level=="✅ Safe" else 50,
                "risk_level": risk_level,
                "alerts": alerts,
                "timestamp": timestamp.isoformat()
            })

            # Send WebSocket update
            await websocket.send_json({
                "tourist_id": tourist_id,
                "lat": lat,
                "lon": lon,
                "speed": speed,
                "safety_score": 100 if risk_level=="✅ Safe" else 50,
                "zone_risk": zone_risk,
                "risk_level": risk_level,
                "alerts": alerts
            })

    except WebSocketDisconnect:
        print(f"❌ Tourist {tourist_id} disconnected")
    except Exception as e:
        print("⚠️ WebSocket error:", e)
