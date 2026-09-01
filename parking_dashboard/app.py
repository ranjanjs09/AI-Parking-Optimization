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

predictions_state = {
    "predictions": {},
    "last_updated": None,
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
def welcome():
    return send_from_directory("static", "welcome.html")


@app.route("/dashboard")
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


@app.route("/api/predict", methods=["GET"])
def get_predictions():
    with lock:
        return jsonify(predictions_state)


@app.route("/api/predict", methods=["POST"])
def update_predictions():
    new_predictions = request.get_json(force=True)
    if not isinstance(new_predictions, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400

    with lock:
        predictions_state["predictions"] = new_predictions
        predictions_state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return jsonify({"message": "Predictions updated"})


if __name__ == "__main__":
    load_state_from_disk()
    port = int(os.environ.get("PORT", 5001))
    print("Starting Parking Dashboard server...")
    print(f"Dashboard: http://localhost:{port}")
    print(f"API:       http://localhost:{port}/api/status")
    print(f"Predict API: http://localhost:{port}/api/predict")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
