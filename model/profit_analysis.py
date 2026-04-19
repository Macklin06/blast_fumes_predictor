import argparse
import math
import os

import joblib
import numpy as np
import pandas as pd


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_ventilation_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURES = [
    "heading_area_m2",
    "heading_length_m",
    "heading_volume_m3",
    "explosive_kg",
    "fan_capacity_m3s",
    "duct_distance_from_face_m",
    "temperature_c",
    "humidity_pct",
]


def summarize_against_baseline(enforced_wait: np.ndarray, baseline_wait: float) -> dict:
    delta = baseline_wait - enforced_wait
    return {
        "baseline_wait": baseline_wait,
        "avg_delta_min": float(np.mean(delta)),
        "median_delta_min": float(np.median(delta)),
        "positive_saving_rate_pct": float(np.mean(delta > 0) * 100.0),
        "negative_delta_rate_pct": float(np.mean(delta < 0) * 100.0),
        "total_positive_saved_min": float(np.clip(delta, 0, None).sum()),
        "net_delta_min": float(delta.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate downtime savings/value vs fixed wait baselines while enforcing a regulatory floor."
    )
    parser.add_argument("--dgms-floor", type=int, default=30, help="Regulatory minimum wait floor in minutes (default: 30)")
    parser.add_argument("--value-per-hour", type=float, default=0.0, help="Estimated value generated per operating hour")
    parser.add_argument("--blasts-per-day", type=float, default=1.0, help="Average number of blasts per day")
    parser.add_argument("--days-per-month", type=float, default=26.0, help="Operating days per month")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    df["heading_volume_m3"] = df["heading_area_m2"] * df["heading_length_m"]

    model = joblib.load(MODEL_PATH)
    preds = model.predict(df[FEATURES])
    model_wait = np.array([math.ceil(x) for x in preds], dtype=float)
    enforced_wait = np.maximum(model_wait, float(args.dgms_floor))

    base_30 = summarize_against_baseline(enforced_wait, 30.0)
    base_45 = summarize_against_baseline(enforced_wait, 45.0)

    print("=== Profitability View (Safety-Constrained) ===")
    print(f"Regulatory floor enforced: {args.dgms_floor} min")
    print("Note: Model recommendations are never allowed below this floor.")
    print("")

    for result in (base_30, base_45):
        b = result["baseline_wait"]
        print(f"--- Versus fixed {int(b)} min method ---")
        print(f"Avg delta per blast (baseline - enforced): {result['avg_delta_min']:.2f} min")
        print(f"Median delta per blast: {result['median_delta_min']:.2f} min")
        print(f"Blasts with time savings: {result['positive_saving_rate_pct']:.1f}%")
        print(f"Blasts with higher wait than baseline: {result['negative_delta_rate_pct']:.1f}%")
        print(f"Total positive-only saved time in dataset: {result['total_positive_saved_min']:.1f} min")
        print(f"Net delta across dataset: {result['net_delta_min']:.1f} min")

        if args.value_per_hour > 0:
            monthly_net_value = (
                result["avg_delta_min"]
                * args.blasts_per_day
                * args.days_per_month
                * (args.value_per_hour / 60.0)
            )
            print(
                "Estimated monthly net value at configured production economics: "
                f"{monthly_net_value:.2f}"
            )
        print("")

    print("Use net delta for honest economics; do not use only positive savings.")


if __name__ == "__main__":
    main()
