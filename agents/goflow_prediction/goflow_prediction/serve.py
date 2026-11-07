"""The live process: JSON lines on stdin, JSON lines on stdout (plan, section 4). Started by the town's adapter
(cosim/goflow_prediction.py) with this package's own Python, CPU only.

    python -m goflow_prediction.serve --model DIR [--model DIR_GOFLOW]

* stdout carries only protocol messages (ready, forecast, alert, error), one per line, flushed at once;
* anything else (warnings, tracebacks) goes to stderr, which the adapter writes to tools\\logs\\goflow_prediction.log;
* a bad line gets an `error` message and the loop goes on; end of stdin ends the process cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # never compete with CARLA for the GPU

from .agent import ForecastAgent, ModelBundle
from .schema import check_in, error, ready


def emit(out, msg: dict) -> None:
    out.write(json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n")
    out.flush()


def serve(model_dirs: list[str], inp=sys.stdin, out=sys.stdout, err=sys.stderr) -> int:
    try:
        agent = ForecastAgent([ModelBundle(d) for d in model_dirs])
    except Exception as e:  # noqa: BLE001 - reported, never a hang
        emit(out, error(f"cannot load model: {e}"))
        traceback.print_exc(file=err)
        return 2
    emit(out, ready("+".join(b.version for b in agent.bundles.values()), agent.tb.horizons_s, agent.tb.n_in,
                    agent.N))
    for line in inp:
        line = line.strip()
        if not line:
            continue
        seq = None
        try:
            msg = json.loads(line)
            seq = msg.get("seq") if isinstance(msg, dict) else None
            problems = check_in(msg)
            if problems:
                emit(out, error("; ".join(problems), seq))
                continue
            typ = msg["type"]
            if typ == "hello":
                if [str(n) for n in msg["nodes"]] != agent.nodes:
                    emit(out, error("hello.nodes differ from the model's nodes (order matters)"))
                if abs(float(msg["dt_s"]) - agent.tb.dt_s) > 1e-9:
                    emit(out, error(f"hello.dt_s {msg['dt_s']} differs from the model's {agent.tb.dt_s}"))
                continue
            if typ == "reset":
                agent.reset()
                continue
            t_wall = time.perf_counter()
            res = agent.step(msg)
            for pr in res["problems"]:
                emit(out, error(pr, seq))
            if res["forecast"] is not None:
                res["forecast"]["cpu_ms"] = round((time.perf_counter() - t_wall) * 1000, 1)
                emit(out, res["forecast"])
            for a in res["alerts"]:
                emit(out, a)
        except Exception as e:  # noqa: BLE001
            emit(out, error(f"{type(e).__name__}: {e}", seq))
            traceback.print_exc(file=err)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="a trained model folder (train_lgbm.py); twice for the plain and the GO-FLOW model")
    a = ap.parse_args(argv)
    sys.exit(serve(a.model))


if __name__ == "__main__":
    main()
