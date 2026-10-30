from flask import Flask, jsonify, request, send_from_directory
from datetime import datetime
import os
import json
import threading
import uuid

app = Flask(__name__, static_folder="static")

STATUS_FILE = os.path.join(os.path.dirname(__file__), "current_status.json")
BOOKINGS_FILE = os.path.join(os.path.dirname(__file__), "bookings.json")
BOOKING_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "booking_history.json")
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

bookings = {}
booking_history = []


def load_state_from_disk():
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                saved = json.load(f)
                state.update(saved)
        except Exception:
            pass


def load_bookings_from_disk():
    bookings_changed = False
    if os.path.exists(BOOKINGS_FILE):
        with open(BOOKINGS_FILE, "r") as f:
            saved = json.load(f)
        if not isinstance(saved, dict):
            raise ValueError("Bookings file must contain a JSON object")
        bookings.update(saved)

    if os.path.exists(BOOKING_HISTORY_FILE):
        with open(BOOKING_HISTORY_FILE, "r") as f:
            saved_history = json.load(f)
        if not isinstance(saved_history, list) or any(not isinstance(item, dict) for item in saved_history):
            raise ValueError("Booking history file must contain a JSON array of booking objects")
        booking_history.extend(saved_history)

    history_ids = {entry.get("id") for entry in booking_history}
    for spot_id, booking in bookings.items():
        if not isinstance(booking, dict):
            raise ValueError(f"Booking for {spot_id} must be a JSON object")
        if "id" not in booking:
            booking["id"] = uuid.uuid4().hex
            bookings_changed = True
        if booking["id"] not in history_ids:
            booking_history.append({
                **booking,
                "status": "active",
                "cancelled_at": None,
            })
            history_ids.add(booking["id"])
            bookings_changed = True

    if bookings_changed:
        save_bookings_to_disk()
        save_booking_history_to_disk()


def save_state_to_disk():
    with open(STATUS_FILE, "w") as f:
        json.dump(state, f, indent=2)


def save_bookings_to_disk():
    with open(BOOKINGS_FILE, "w") as f:
        json.dump(bookings, f, indent=2)


def save_booking_history_to_disk():
    with open(BOOKING_HISTORY_FILE, "w") as f:
        json.dump(booking_history, f, indent=2)


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


@app.route("/api/bookings", methods=["GET"])
def get_bookings():
    with lock:
        return jsonify(bookings)


@app.route("/api/bookings/history", methods=["GET"])
def get_booking_history():
    with lock:
        return jsonify(list(reversed(booking_history)))


@app.route("/api/bookings", methods=["POST"])
def create_booking():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Body must be a JSON object"}), 400

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        return jsonify({"error": "Name must be between 1 and 80 characters"}), 400

    spot_id = payload.get("spot_id")
    if not isinstance(spot_id, str) or not spot_id:
        return jsonify({"error": "spot_id is required"}), 400

    with lock:
        if spot_id not in state["spots"]:
            return jsonify({"error": "Parking spot not found"}), 404
        if state["spots"][spot_id] != "empty":
            return jsonify({"error": "This parking spot is occupied"}), 409
        if spot_id in bookings:
            return jsonify({"error": "This parking spot is already booked"}), 409

        booking = {
            "id": uuid.uuid4().hex,
            "spot_id": spot_id,
            "name": name.strip(),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        bookings[spot_id] = booking
        booking_history.append({
            **booking,
            "status": "active",
            "cancelled_at": None,
        })
        save_bookings_to_disk()
        save_booking_history_to_disk()

    return jsonify({"message": "Parking spot booked", "booking": booking}), 201


@app.route("/api/bookings/<spot_id>", methods=["DELETE"])
def cancel_booking(spot_id):
    with lock:
        if spot_id not in bookings:
            return jsonify({"error": "Booking not found"}), 404
        booking = bookings[spot_id]
        history_entry = next(
            (entry for entry in reversed(booking_history) if entry.get("id") == booking.get("id")),
            None,
        )
        cancelled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if history_entry is None:
            history_entry = {**booking, "status": "active", "cancelled_at": None}
            booking_history.append(history_entry)
        history_entry["status"] = "cancelled"
        history_entry["cancelled_at"] = cancelled_at
        save_booking_history_to_disk()
        del bookings[spot_id]
        save_bookings_to_disk()

    return jsonify({"message": "Booking cancelled"})


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
    load_bookings_from_disk()
    port = int(os.environ.get("PORT", 5001))
    print("Starting Parking Dashboard server...")
    print(f"Dashboard: http://localhost:{port}")
    print(f"API:       http://localhost:{port}/api/status")
    print(f"Predict API: http://localhost:{port}/api/predict")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
