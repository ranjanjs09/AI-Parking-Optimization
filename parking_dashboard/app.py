"""
app.py
--------
Module 2: Backend + Dashboard server.

Ye Flask server 2 kaam karta hai:
    1. Dashboard webpage serve karta hai (browser me khulti hai)
    2. Ek simple API deta hai jisse:
        - GET  /api/status  -> current parking status (JSON) return karta hai
        - POST /api/status  -> naya status update karta hai (Module 1 ka
          CV script isko call karega jab bhi naya frame process ho)

HOW TO RUN:
    python app.py

Phir browser me kholo:
    http://localhost:5000

Status ko manually bhi test kar sakte ho (dashboard turant update ho jayega):
    curl -X POST http://localhost:5000/api/status \
        -H "Content-Type: application/json" \
        -d '{"spot_1": "occupied", "spot_2": "empty"}'

REAL SYSTEM ME:
    push_live_status.py script Module 1 (detect_occupancy.py) ka CV model use
    karke har kuch second me naya status is /api/status endpoint pe POST
    karega - wahi cheez ye dashboard live dikhayega.
"""

from flask import Flask, jsonify, request, send_from_directory
from datetime import datetime
import os
import json
import threading

app = Flask(__name__, static_folder="static")

STATUS_FILE = os.path.join(os.path.dirname(__file__), "current_status.json")
lock = threading.Lock()

# In-memory state (STATUS_FILE me bhi persist hota hai taaki server restart
# hone pe bhi last known status yaad rahe)
state = {
    "spots": {},          # e.g. {"spot_1": "empty", "spot_2": "occupied"}
    "last_updated": None,
    "total": 0,
    "occupied": 0,
    "available": 0,
}


def load_state_from_disk():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                saved = json.load(f)
                state.update(saved)
        except Exception:
            pass


def save_state_to_disk():
    with open(STATUS_FILE, "w") as f:
        json.dump(state, f, indent=2)


def recompute_counts():
    spots = state["spots"]
    total = len(spots)
    occupied = sum(1 for v in spots.values() if v == "occupied")
    state["total"] = total
    state["occupied"] = occupied
    state["available"] = total - occupied


@app.route("/")
def dashboard():
    return send_from_directory("static", "index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    with lock:
        return jsonify(state)


@app.route("/api/status", methods=["POST"])
def update_status():
    """
    Expects JSON body like: {"spot_1": "empty", "spot_2": "occupied", ...}
    (Ye exactly wahi format hai jo detect_occupancy.py console pe print karta hai)
    """
    new_spots = request.get_json(force=True)
    if not isinstance(new_spots, dict):
        return jsonify({"error": "Body must be a JSON object of spot_id -> status"}), 400

    with lock:
        state["spots"] = new_spots
        state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        recompute_counts()
        save_state_to_disk()

    return jsonify({"message": "Status updated", "state": state})


if __name__ == "__main__":
    load_state_from_disk()
    port = int(os.environ.get("PORT", 5001))  # 5000 macOS AirPlay Receiver se clash karta hai, isliye 5001 default
    print("Starting Parking Dashboard server...")
    print(f"Dashboard: http://localhost:{port}")
    print(f"API:       http://localhost:{port}/api/status")
    app.run(host="0.0.0.0", port=port, debug=True)
