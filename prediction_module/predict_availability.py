"""
predict_availability.py
--------------------------
Module 3 (Prediction) - STEP 3.

Trained model use karke "abhi se HORIZON minutes baad" har spot ka status
predict karta hai.

Current status kahan se aata hai (in priority order):
    1. --status_json file diya ho, to wahan se
    2. --api_url diya ho (dashboard ka /api/status), to live dashboard se
    3. Dono na diye ho, to sab spots "empty" maan lega (sirf demo ke liye)

HOW TO USE:

Dashboard se live current status leke predict karna (Module 2 ke saath integration):
    python predict_availability.py --model predictor.joblib --spots spots.json \
        --api_url http://127.0.0.1:5001/api/status

Kisi file se:
    python predict_availability.py --model predictor.joblib --spots spots.json \
        --status_json result_status.json

Agar --push_to_dashboard diya, to prediction bhi dashboard ke /api/predict
endpoint pe bhej dega (dashboard pe "predicted in 15 min" dikhega).
"""

import argparse
import json
from datetime import datetime, timedelta

import joblib
import pandas as pd
import requests


def get_current_status(args):
    if args.status_json:
        with open(args.status_json, "r") as f:
            data = json.load(f)
            # detect_occupancy.py ka JSON seedha {spot_id: status} format hota hai,
            # ya app.py ka format {"spots": {...}} bhi ho sakta hai - dono handle karo
            return data.get("spots", data)

    if args.api_url:
        try:
            res = requests.get(args.api_url, timeout=3, proxies={"http": None, "https": None})
            res.raise_for_status()
            return res.json().get("spots", {})
        except Exception as e:
            print(f"WARNING: Dashboard se status fetch nahi ho paya ({e}). Sab spots 'empty' maan rahe hain.")
            return {}

    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="predictor.joblib")
    parser.add_argument("--spots", default="spots.json")
    parser.add_argument("--status_json", default=None)
    parser.add_argument("--api_url", default=None, help="e.g. http://127.0.0.1:5001/api/status")
    parser.add_argument("--push_to_dashboard", action="store_true",
                         help="Prediction ko dashboard ke /api/predict pe bhi POST kar do")
    parser.add_argument("--predict_api_url", default="http://127.0.0.1:5001/api/predict")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    spot_encoder = bundle["spot_encoder"]
    feature_cols = bundle["feature_cols"]
    horizon_minutes = bundle["horizon_minutes"]

    with open(args.spots, "r") as f:
        all_spots = json.load(f)

    current_status = get_current_status(args)

    future_time = datetime.now() + timedelta(minutes=horizon_minutes)
    hour = future_time.hour
    minute = future_time.minute
    day_of_week = future_time.weekday()
    is_weekend = int(day_of_week >= 5)

    rows = []
    spot_ids = []
    for spot_id in all_spots.keys():
        status_now = current_status.get(spot_id, "empty")
        current_status_flag = 1 if status_now == "occupied" else 0

        try:
            spot_encoded = spot_encoder.transform([spot_id])[0]
        except ValueError:
            # ye spot training data me nahi tha (naya spot) - skip karo ya fallback do
            print(f"WARNING: {spot_id} model ki training me nahi tha, skip kar rahe hain.")
            continue

        rows.append({
            "hour": hour, "minute": minute, "day_of_week": day_of_week,
            "is_weekend": is_weekend, "spot_encoded": spot_encoded,
            "current_status": current_status_flag,
        })
        spot_ids.append(spot_id)

    if not rows:
        print("Koi valid spot nahi mila predict karne ke liye.")
        return

    X = pd.DataFrame(rows)[feature_cols]
    probs = model.predict_proba(X)[:, 1]  # probability of "occupied"

    predictions = {}
    print(f"\nPrediction for {future_time.strftime('%H:%M')} ({horizon_minutes} min from now):\n")
    for spot_id, prob_occupied in zip(spot_ids, probs):
        predicted = "occupied" if prob_occupied >= 0.5 else "empty"
        confidence = prob_occupied if predicted == "occupied" else 1 - prob_occupied
        predictions[spot_id] = {
            "current": current_status.get(spot_id, "unknown"),
            "predicted_in_%dmin" % horizon_minutes: predicted,
            "confidence": round(float(confidence), 3),
        }
        print(f"  {spot_id}: now={current_status.get(spot_id, '?'):9s} -> "
              f"in {horizon_minutes}min: {predicted:9s} (confidence: {confidence:.0%})")

    if args.push_to_dashboard:
        try:
            res = requests.post(
                args.predict_api_url, json=predictions, timeout=3,
                proxies={"http": None, "https": None}
            )
            print(f"\nPushed predictions to dashboard: {res.status_code}")
        except Exception as e:
            print(f"\nWARNING: Could not push to dashboard: {e}")


if __name__ == "__main__":
    main()
