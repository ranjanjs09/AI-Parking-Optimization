"""
train_predictor.py
---------------------
Module 3 (Prediction) - STEP 2.

Historical data (history.csv) se ek model train karta hai jo predict karta
hai: "agar spot X abhi [status] hai, [is time/day pe], to 15 minute baad
uska status kya hoga?"

MODEL CHOICE: Random Forest Classifier (scikit-learn) use kiya hai -
Prophet/deep learning jaisi heavy libraries nahi, jo install karne me
dikkat de sakti hain. Random Forest:
    - Lightweight, fast train hota hai
    - Time-of-day patterns aur current-status dono ko easily seekh leta hai
    - macOS pe bina kisi extra dependency issue ke chalta hai

FEATURES used for prediction:
    - hour of day (0-23)
    - minute (0-59)
    - day of week (0=Monday .. 6=Sunday)
    - is_weekend (0/1)
    - spot_id (encoded as integer)
    - current_status (0=empty, 1=occupied)   <- ye sabse important feature hai,
      kyunki current status future status ka sabse bada predictor hota hai

LABEL: status HORIZON minutes baad (default 15 min)

HOW TO USE:
    python train_predictor.py --history history.csv --horizon_minutes 15 --output predictor.joblib
"""

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def build_features(df, spot_encoder):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["spot_encoded"] = spot_encoder.transform(df["spot_id"])
    df["current_status"] = (df["status"] == "occupied").astype(int)
    return df


def make_labels(df, interval_minutes, horizon_minutes):
    """
    Har spot ke liye, HORIZON minutes baad ka status label ke roop me jodta hai.
    Time-series hai isliye per-spot shift karna hoga (dusre spot ka data
    mix nahi hona chahiye).
    """
    steps_ahead = max(round(horizon_minutes / interval_minutes), 1)
    df = df.sort_values(["spot_id", "timestamp"]).reset_index(drop=True)
    df["future_status"] = df.groupby("spot_id")["status"].shift(-steps_ahead)
    df = df.dropna(subset=["future_status"])
    df["label"] = (df["future_status"] == "occupied").astype(int)
    return df


def infer_interval_minutes(df):
    diffs = df.sort_values("timestamp").groupby("spot_id")["timestamp"].diff().dropna()
    return diffs.dt.total_seconds().median() / 60.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="history.csv")
    parser.add_argument("--horizon_minutes", type=int, default=15)
    parser.add_argument("--output", default="predictor.joblib")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                         help="Aakhri X%% time ko validation ke liye rakho (time-based split, random nahi - taaki data leakage na ho)")
    args = parser.parse_args()

    df = pd.read_csv(args.history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    interval_minutes = infer_interval_minutes(df)
    print(f"Detected data interval: ~{interval_minutes:.1f} minutes")

    spot_encoder = LabelEncoder()
    spot_encoder.fit(df["spot_id"])

    df = build_features(df, spot_encoder)
    df = make_labels(df, interval_minutes, args.horizon_minutes)

    feature_cols = ["hour", "minute", "day_of_week", "is_weekend", "spot_encoded", "current_status"]
    X = df[feature_cols]
    y = df["label"]

    # TIME-BASED split (not random!) - taaki model "future ka pata" na chale
    # (real-world me bhi tum future data pe train nahi kar sakte)
    split_time = df["timestamp"].quantile(1 - args.val_fraction)
    train_mask = df["timestamp"] < split_time
    X_train, X_val = X[train_mask], X[~train_mask]
    y_train, y_val = y[train_mask], y[~train_mask]

    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    print(f"\nValidation Accuracy: {acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds, target_names=["will_be_empty", "will_be_occupied"]))

    importances = dict(zip(feature_cols, model.feature_importances_))
    print("Feature importances (report me likhne ke liye achha hai):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    joblib.dump({
        "model": model,
        "spot_encoder": spot_encoder,
        "feature_cols": feature_cols,
        "horizon_minutes": args.horizon_minutes,
    }, args.output)
    print(f"\nModel saved to {args.output}")
    print(f"Ab ye chalao: python predict_availability.py --model {args.output} --spots spots.json")


if __name__ == "__main__":
    main()
