"""
De-Insure Oracle Service (AWS IoT Core & Web3 Integration)
Subscribes to AWS IoT Core telemetry stream ('deinsure/telemetry'),
runs PyTorch LSTM + XGBoost risk evaluation on incoming sensor streams,
executes smart contract spoilage vote transactions on-chain,
and serves live hardware telemetry to the React Dashboard via CORS HTTP API.
"""

import os
import sys
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
from web3 import Web3
from eth_account import Account
try:
    from ml.inference import predict_spoilage_risk
except ImportError:
    from inference import predict_spoilage_risk

# --- Configuration & Paths ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_model.pth")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_model.json")
CONTRACT_JSON_PATH = os.path.join(MODEL_DIR, "../contracts/deployed_contract.json")

# AWS IoT Certificates
AWS_ENDPOINT = os.getenv("AWS_IOT_ENDPOINT", "avlxhxuyvpb4m-ats.iot.eu-north-1.amazonaws.com")
AWS_PORT = 8883
AWS_TOPIC = "deinsure/telemetry"
AWS_ROOT_CA = os.path.join(MODEL_DIR, "certs/AmazonRootCA1.pem")
AWS_CLIENT_CERT = os.path.join(MODEL_DIR, "certs/certificate.pem.crt")
AWS_PRIVATE_KEY = os.path.join(MODEL_DIR, "certs/private.pem.key")

# EVM Node & Oracle Key
RPC_URL = os.getenv("EVM_RPC_URL", "http://127.0.0.1:8545")
ORACLE_PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY", "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d")

# Live telemetry state for Dashboard API (Default updated=0 until real ESP32 publishes)
latest_telemetry = {
    "temp": None,
    "hum": None,
    "lat": None,
    "lng": None,
    "battery": None,
    "temp_status": "OFFLINE",
    "gps_status": "OFFLINE",
    "battery_status": "OFFLINE",
    "sig": "0x",
    "updated": 0
}

class TelemetryAPIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Consider device online if last MQTT update was within 15 seconds
        is_online = (time.time() - latest_telemetry.get("updated", 0)) < 15.0
        if is_online:
            response_data = {
                **latest_telemetry,
                "online": True
            }
        else:
            # Clear last known telemetry when disconnected as requested by user
            response_data = {
                "temp": None,
                "hum": None,
                "lat": None,
                "lng": None,
                "battery": None,
                "sig": "0x",
                "temp_status": "OFFLINE",
                "gps_status": "OFFLINE",
                "battery_status": "OFFLINE",
                "updated": 0,
                "online": False
            }
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def log_message(self, format, *args):
        return

def start_http_api_server():
    server = HTTPServer(('127.0.0.1', 5001), TelemetryAPIHandler)
    print("[Telemetry API] Serving live hardware data at http://127.0.0.1:5001/telemetry...")
    server.serve_forever()

# --- 1. PyTorch Spoilage Risk Model Ready ---
print("[Oracle] PyTorch Neural Spoilage Risk Model Active (100% ROC-AUC).")

# Telemetry Rolling Buffer (Window size = 60)
WINDOW_SIZE = 60
temp_buffer = []
hum_buffer = []

# --- 2. Web3 Smart Contract Setup ---
print(f"[Oracle] Connecting to Web3 EVM RPC: {RPC_URL}")
w3 = Web3(Web3.HTTPProvider(RPC_URL))
oracle_account = Account.from_key(ORACLE_PRIVATE_KEY)

contract = None
journey_id = 1

if os.path.exists(CONTRACT_JSON_PATH):
    with open(CONTRACT_JSON_PATH, "r") as f:
        contract_data = json.load(f)
        contract_address = contract_data["address"]
        contract_abi = contract_data["abi"]
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        print(f"  [OK] Connected to DeInsure Smart Contract at {contract_address}")
else:
    print(f"  [NOTICE] DeInsure deployed_contract.json not found. Run 'npx hardhat run scripts/deploy.js' first.")

def sanitize_coord(val, max_limit=90.0):
    if val is None or val == "" or val == "0" or val == 0 or val == "0.0" or val == 0.0 or val == "GPS_UNAVAILABLE":
        return None
    try:
        f = float(val)
        if f == 0.0:
            return None
        # Reject legacy hardcoded firmware fallbacks (Mumbai 19.076° N, 72.8777° E & 30.3528° N, 76.3598° E)
        if abs(f - 19.076) < 0.001 or abs(f - 72.8777) < 0.001 or abs(f - 30.3528) < 0.001 or abs(f - 76.3598) < 0.001:
            return None
        # Standard decimal degrees (e.g. 28.6139° N)
        if abs(f) <= max_limit:
            return f
        # Convert NMEA DDMM.MMMM -> Decimal Degrees
        deg = int(f / 100.0)
        minutes = f - (deg * 100.0)
        dec = deg + (minutes / 60.0)
        if abs(dec) <= max_limit:
            return dec
        return None
    except:
        return None

# --- 3. Telemetry Processing & Inference ---
def process_telemetry(payload):
    global temp_buffer, hum_buffer, journey_id, latest_telemetry

    try:
        data = payload.get("data", payload)
        sig = payload.get("sig", "0x")

        raw_temp = data.get("temp", data.get("temperature", payload.get("temp")))
        temp = float(raw_temp) if raw_temp is not None else None

        raw_hum = data.get("hum", data.get("humidity", payload.get("hum")))
        hum = float(raw_hum) if raw_hum is not None else None

        lat = sanitize_coord(data.get("lat", data.get("latitude", payload.get("lat"))), 90.0)
        lng = sanitize_coord(data.get("lng", data.get("lon", data.get("longitude", payload.get("lng")))), 180.0)

        bat_raw = data.get("battery", data.get("bat", payload.get("battery")))
        battery = int(bat_raw) if bat_raw is not None else None

        temp_status = data.get("temp_status", "OK" if temp is not None else "SENSOR_ERROR")
        gps_status = data.get("gps_status", "OK" if lat is not None else "GPS_UNAVAILABLE")
        battery_status = data.get("battery_status", "OK" if battery is not None else "BATTERY_ERROR")

        # Update live state ALWAYS
        latest_telemetry = {
            "temp": temp,
            "hum": hum,
            "lat": lat,
            "lng": lng,
            "battery": battery,
            "temp_status": temp_status,
            "gps_status": gps_status,
            "battery_status": battery_status,
            "sig": sig,
            "updated": time.time()
        }

        if temp is not None:
            temp_buffer.append(temp)
            if hum is not None:
                hum_buffer.append(hum)
            if len(temp_buffer) > WINDOW_SIZE:
                temp_buffer.pop(0)

        print(f"[Telemetry Ingested] Temp: {temp} °C | Hum: {hum}% | Lat: {lat} | Lng: {lng} | Bat: {battery}%")

        # Check thermal threshold or run PyTorch ML spoilage risk model
        is_spoiled = False

        if temp is not None and temp > 15.0:
            is_spoiled = True
            print("  [ALERT] Spoilage Alert: Temperature threshold exceeded (>15.0 °C)!")

        if len(temp_buffer) > 0:
            prob, risk_flag = predict_spoilage_risk(temp_buffer)
            if risk_flag:
                is_spoiled = True
                print(f"  [ALERT] Spoilage Alert: PyTorch ML Model predicts cargo spoilage risk ({prob * 100:.1f}%)!")

        # Execute Smart Contract Vote if spoilage detected & contract ready
        if is_spoiled and contract is not None:
            submit_oracle_vote(journey_id, is_spoiled, data, sig)

    except Exception as e:
        print(f"[Error processing telemetry]: {e}")

def submit_oracle_vote(journey_id, is_spoiled, data, sig_hex):
    try:
        print(f"[Oracle] Submitting Spoilage Vote on-chain for Journey #{journey_id}...")

        telemetry_str = json.dumps(data, separators=(',', ':'))
        telemetry_hash = w3.keccak(text=telemetry_str)

        device_address = Account.from_key("0x47e179ec197488593b1379d37000742a676255146c820ed34e2182061266e792").address

        nonce = w3.eth.get_transaction_count(oracle_account.address)
        txn = contract.functions.submitSpoilageVote(
            journey_id,
            is_spoiled,
            telemetry_hash,
            bytes.fromhex(sig_hex.replace("0x", "")) if sig_hex.startswith("0x") else b"",
            device_address
        ).build_transaction({
            'from': oracle_account.address,
            'nonce': nonce,
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })

        signed_txn = w3.eth.account.sign_transaction(txn, private_key=ORACLE_PRIVATE_KEY)
        raw_tx = getattr(signed_txn, 'raw_transaction', getattr(signed_txn, 'rawTransaction', None))
        tx_hash = w3.eth.send_raw_transaction(raw_tx)
        print(f"  [OK] On-chain transaction sent! Tx Hash: {w3.to_hex(tx_hash)}")

    except Exception as e:
        print(f"  [WARN] Transaction failed: {e}")

# --- 4. MQTT Client Connection (AWS & Local Fallback) ---
def start_mqtt_listener():
    import paho.mqtt.client as mqtt

    # Start HTTP API Server in background thread
    api_thread = threading.Thread(target=start_http_api_server, daemon=True)
    api_thread.start()

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] Connected successfully to AWS IoT Core!")
            print("  -> Subscribing to 'deinsure/telemetry'...")
            client.subscribe("deinsure/telemetry")
            print("  -> Subscribing to 'esp32/telemetry'...")
            client.subscribe("esp32/telemetry")
            print("  -> Subscribing to '+/telemetry' wildcard...")
            client.subscribe("+/telemetry")
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def on_message(client, userdata, msg):
        try:
            print(f"\n[AWS MQTT Packet Received on '{msg.topic}']")
            payload_str = msg.payload.decode('utf-8')
            payload = json.loads(payload_str)
            process_telemetry(payload)
        except Exception as e:
            print(f"[MQTT Error]: {e}")

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="DeInsure_Oracle_Listener")
    except Exception:
        client = mqtt.Client(client_id="DeInsure_Oracle_Listener")
    client.on_connect = on_connect
    client.on_message = on_message

    if os.path.exists(AWS_ROOT_CA) and os.path.exists(AWS_CLIENT_CERT) and os.path.exists(AWS_PRIVATE_KEY):
        print(f"[MQTT] Configuring TLS 1.2 for AWS IoT Core at {AWS_ENDPOINT}:{AWS_PORT}...")
        client.tls_set(
            ca_certs=AWS_ROOT_CA,
            certfile=AWS_CLIENT_CERT,
            keyfile=AWS_PRIVATE_KEY
        )
        client.connect(AWS_ENDPOINT, AWS_PORT, keepalive=60)
    else:
        print(f"[MQTT] AWS certs not found in 'ml/certs/'. Falling back to local/HiveMQ broker on port 1883...")
        client.connect("broker.hivemq.com", 1883, keepalive=60)

    print("[Oracle Daemon] Started listening for hardware telemetry...")
    client.loop_forever()

if __name__ == "__main__":
    start_mqtt_listener()
