"""CLI for video incident inference and an honest prediction handoff boundary."""
import argparse
import json
import sys
from contextlib import redirect_stdout

from src.incident_pipeline import run_traffic_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Highway MP4 path")
    parser.add_argument("--source-id", help="Camera identifier; defaults to filename stem")
    args = parser.parse_args()
    # Ultralytics may print download/progress messages. Keep stdout valid JSON.
    with redirect_stdout(sys.stderr):
        result = run_traffic_pipeline(args.video, source_id=args.source_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
