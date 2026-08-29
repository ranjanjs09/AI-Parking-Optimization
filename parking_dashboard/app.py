from flask import Flask, jsonify, request, send_from_directory
from datetime import datetime
import os
import json
import threading

app = Flask(__name__, static_folder="static")

STATUS_FILE = os.path.join(os.path.dirname(__file__), "current_status.json")
lock = threading.Lock()

state = {
    "spots": {},
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
    port = int(os.environ.get("PORT", 5001))
    print("Starting Parking Dashboard server...")
    print(f"Dashboard: http://localhost:{port}")
    print(f"API:       http://localhost:{port}/api/status")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
