"""
generate_historical_data.py
------------------------------
Module 3 (Prediction) - STEP 1.

Real system me ye data Module 1 (CV detection) se collect hota - har baar
detect_occupancy.py chalta, uska output timestamp ke saath history me log
ho jaata. College project ke time-frame me itna real data collect karna
possible nahi hai (weeks/months lagte), isliye ye script REALISTIC synthetic
historical data generate karta hai - jisse hum turant Module 3 build aur
demo kar sakein.

REALISM: Ye random data nahi hai - har spot ke liye ek Markov-chain style
simulation chalti hai:
    - "Arrival rate" (empty -> occupied hone ka chance) peak hours
      (10am-8pm, jaise ek market/mall) me zyada hota hai, raat ko kam.
    - "Departure rate" (occupied -> empty hone ka chance) average parking
      duration pe depend karta hai.
    - Weekends pe thoda zyada traffic (agar market/commercial area simulate
      kar rahe hain).
    - Har spot ki apni "popularity" hoti hai (kuch spots entrance ke paas
      hone ki wajah se zyada busy rehte hain - jaise real parking lots me).

HOW TO USE:
    python generate_historical_data.py --spots spots.json --days 14 --output history.csv

OUTPUT: history.csv with columns: timestamp, spot_id, status
"""

import argparse
import csv
import json
import random
from datetime import datetime, timedelta


def hourly_arrival_rate(hour, is_weekend):
    """
    Kitna chance hai ki is ghante me ek empty spot occupied ho jaayega.
    Market/commercial area jaisa pattern: subah kam, dopeher-shaam peak,
    raat ko bahut kam.
    """
    base_curve = {
        0: 0.02, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.02,
        6: 0.05, 7: 0.10, 8: 0.18, 9: 0.25, 10: 0.35, 11: 0.40,
        12: 0.45, 13: 0.42, 14: 0.38, 15: 0.40, 16: 0.45, 17: 0.50,
        18: 0.55, 19: 0.50, 20: 0.35, 21: 0.20, 22: 0.10, 23: 0.05,
    }
    rate = base_curve[hour]
    if is_weekend:
        rate *= 1.3  # weekends zyada busy (market/mall assumption)
    return min(rate, 0.9)


def departure_probability(avg_duration_minutes, interval_minutes):
    """
    Ek time-step me occupied spot ke khaali hone ka chance.
    Agar average parking duration 40 min hai aur hum har 5 min check kar
    rahe hain, to roughly har step me 1/8 chance hai ki gaadi nikal jaaye.
    """
    steps_to_leave = max(avg_duration_minutes / interval_minutes, 1)
    return 1.0 / steps_to_leave


def simulate_spot(spot_id, start_time, num_steps, interval_minutes, popularity):
    """
    Ek spot ke liye poori history simulate karta hai - Markov chain:
    agla status sirf current status + time-of-day pe depend karta hai.
    """
    records = []
    status = "empty" if random.random() > 0.3 else "occupied"  # random starting state
    avg_duration = random.uniform(25, 70)  # har spot ka apna average parking duration (minutes)

    current_time = start_time
    for _ in range(num_steps):
        hour = current_time.hour
        is_weekend = current_time.weekday() >= 5

        if status == "empty":
            p_occupy = hourly_arrival_rate(hour, is_weekend) * popularity
            if random.random() < p_occupy:
                status = "occupied"
        else:
            p_leave = departure_probability(avg_duration, interval_minutes)
            if random.random() < p_leave:
                status = "empty"

        records.append((current_time.strftime("%Y-%m-%d %H:%M:%S"), spot_id, status))
        current_time += timedelta(minutes=interval_minutes)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spots", default="spots.json", help="spots.json from Module 1 (defines how many spots)")
    parser.add_argument("--days", type=int, default=14, help="Kitne din ka historical data simulate karna hai")
    parser.add_argument("--interval_minutes", type=int, default=5, help="Har kitne minute pe ek reading")
    parser.add_argument("--output", default="history.csv")
    args = parser.parse_args()

    random.seed(7)

    with open(args.spots, "r") as f:
        spots = json.load(f)
    spot_ids = list(spots.keys())

    num_steps = int((args.days * 24 * 60) / args.interval_minutes)
    start_time = datetime.now() - timedelta(days=args.days)

    print(f"Simulating {args.days} days of history for {len(spot_ids)} spots "
          f"({num_steps} readings per spot, every {args.interval_minutes} min)...")

    all_records = []
    for spot_id in spot_ids:
        popularity = random.uniform(0.7, 1.3)  # kuch spots zyada "popular" (jaise entrance ke paas)
        records = simulate_spot(spot_id, start_time, num_steps, args.interval_minutes, popularity)
        all_records.extend(records)

    all_records.sort(key=lambda r: r[0])

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "spot_id", "status"])
        writer.writerows(all_records)

    print(f"Saved {len(all_records)} records to {args.output}")
    print(f"Ab ye chalao: python train_predictor.py --history {args.output}")


if __name__ == "__main__":
    main()
