"""CLI for video incident inference and an honest prediction handoff boundary."""
import argparse
import json

from src.incident_pipeline import run_traffic_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Highway MP4 path")
    parser.add_argument("--source-id", help="Camera identifier; defaults to filename stem")
    args = parser.parse_args()
    print(json.dumps(run_traffic_pipeline(args.video, source_id=args.source_id), indent=2))


if __name__ == "__main__":
    main()
