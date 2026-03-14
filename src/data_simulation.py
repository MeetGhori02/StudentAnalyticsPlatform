"""
data_simulation.py
Simulates event-like student data arrival at fixed intervals and writes to bronze layer.

Example:
  python data_simulation.py --batches 5 --batch-size 40 --interval-seconds 1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "final_student_dataset.csv"
BRONZE_DIR = BASE_DIR / "data" / "bronze"
OUTPUT_PATH = BRONZE_DIR / "student_events.csv"


def simulate(
    batches: int,
    batch_size: int,
    interval_seconds: float,
    seed: int = 42,
) -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_PATH}")

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    base_df = pd.read_csv(PROCESSED_PATH)
    if base_df.empty:
        raise ValueError("Processed dataset is empty. Run ETL first.")

    if "attendance_rate" not in base_df.columns:
        att_cols = [c for c in ["th_attendance_rate", "pr_attendance_rate"] if c in base_df.columns]
        if att_cols:
            base_df["attendance_rate"] = base_df[att_cols].mean(axis=1)
        else:
            base_df["attendance_rate"] = 0.75

    if "placement_readiness_score" not in base_df.columns:
        base_df["placement_readiness_score"] = (
            pd.to_numeric(base_df.get("mean_cgpa", 0), errors="coerce").fillna(0) * 10
        ).clip(0, 100)

    if "risk_level" not in base_df.columns:
        if "academic_risk_score" in base_df.columns:
            base_df["risk_level"] = pd.cut(
                pd.to_numeric(base_df["academic_risk_score"], errors="coerce").fillna(0),
                bins=[-0.1, 40, 70, 100],
                labels=["Low", "Medium", "High"],
            ).astype(str)
        else:
            base_df["risk_level"] = "Medium"

    rng = np.random.default_rng(seed)
    current_event_id = 1
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_csv(OUTPUT_PATH, usecols=["event_id"])
            if not existing.empty:
                current_event_id = int(existing["event_id"].max()) + 1
        except Exception:
            pass

    write_header = not OUTPUT_PATH.exists()

    for batch_id in range(1, batches + 1):
        sampled = base_df.sample(
            n=min(batch_size, len(base_df)),
            replace=batch_size > len(base_df),
            random_state=int(rng.integers(1, 10_000_000)),
        ).copy()

        # Simulate light drift in observed live metrics.
        sampled["live_study_hours"] = (sampled["study_hours_day"] + rng.normal(0, 0.25, len(sampled))).clip(0, 8)
        sampled["live_stress"] = (sampled["stress_numeric"] + rng.normal(0, 0.35, len(sampled))).clip(0, 3)
        sampled["live_attendance"] = (
            sampled["attendance_rate"] + rng.normal(0, 0.03, len(sampled))
        ).clip(0, 1)

        sampled["event_time"] = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S")
        sampled["batch_id"] = batch_id
        sampled["event_id"] = np.arange(current_event_id, current_event_id + len(sampled))
        current_event_id += len(sampled)

        columns = [
            "event_id",
            "event_time",
            "batch_id",
            "student_id_perf",
            "mean_cgpa",
            "risk_level",
            "attendance_rate",
            "stress_numeric",
            "placement_readiness_score",
            "live_study_hours",
            "live_stress",
            "live_attendance",
        ]
        available_cols = [col for col in columns if col in sampled.columns]
        sampled[available_cols].to_csv(
            OUTPUT_PATH,
            mode="a",
            index=False,
            header=write_header,
        )
        write_header = False

        print(
            f"[SIM] batch {batch_id}/{batches} appended {len(sampled)} events to {OUTPUT_PATH.name}"
        )
        if batch_id < batches and interval_seconds > 0:
            time.sleep(interval_seconds)

    print(f"[SIM] completed. Total output file: {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate interval-based student event data.")
    parser.add_argument("--batches", type=int, default=3, help="Number of batches to generate.")
    parser.add_argument("--batch-size", type=int, default=30, help="Rows per batch.")
    parser.add_argument("--interval-seconds", type=float, default=1.0, help="Sleep between batches.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    simulate(
        batches=max(1, args.batches),
        batch_size=max(1, args.batch_size),
        interval_seconds=max(0.0, args.interval_seconds),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
