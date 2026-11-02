"""Run the saved XGBoost agent on genuine repository PEMS08 observations.

This is a standalone sensor smoke test, not a camera-to-sensor mapping.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.agents import TrafficPredictionControlAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-id", type=int, default=33)
    parser.add_argument("--timestep", type=int, default=15227)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    observations = pd.read_csv(root / "data/raw/traffic_dataset.csv")
    recent = observations[observations.timestep.between(args.timestep - 12, args.timestep)].copy()

    def history_provider(sensor_id, timestep):
        return recent[(recent.location == sensor_id) &
                      recent.timestep.between(timestep - 12, timestep)].copy()

    current = history_provider(args.sensor_id, args.timestep).iloc[-1]
    message = {"request_id": f"pems08-live-{args.sensor_id}-{args.timestep}",
               "sensor_id": args.sensor_id, "timestep": args.timestep,
               "traffic_data": {"current_flow": float(current.flow),
                                "current_speed": float(current.speed),
                                "current_occupancy": float(current.occupy)}}
    agent = TrafficPredictionControlAgent(root, prediction_mode="live",
                                          history_provider=history_provider)
    response = agent.handle_message(message)
    print(json.dumps({"source": "saved PEMS08 observations; no camera mapping",
                      "message": message, "response": response}, indent=2,
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
