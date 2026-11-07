"""Her live process: JSON lines in (hello, obs, reset: the same messages the town sends to goflow_prediction), JSON
lines out (ready, goflow, error). Rules only, CPU only, no LLM (plan, Q-J3).

    python -m goflow_traffic.serve --model DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from .agent import GoFlowTown05


def emit(out, msg):
    out.write(json.dumps(msg, separators=(",", ":"), allow_nan=False) + "\n")
    out.flush()


def err(msg, seq=None):
    return {"type": "error", "v": 1, "seq": seq, "msg": msg}


def serve(folder, inp=sys.stdin, out=sys.stdout, log=sys.stderr) -> int:
    try:
        agent = GoFlowTown05(folder)
    except Exception as e:  # noqa: BLE001
        emit(out, err(f"cannot load model: {e}"))
        traceback.print_exc(file=log)
        return 2
    emit(out, {"type": "ready", "v": 1, "model": agent.version, "horizons_s": [agent.horizon_s],
               "warmup_steps": 13, "nodes": agent.N})
    for line in inp:
        line = line.strip()
        if not line:
            continue
        seq = None
        try:
            msg = json.loads(line)
            if not isinstance(msg, dict) or msg.get("v") != 1:
                emit(out, err("not a v1 message object"))
                continue
            seq = msg.get("seq")
            typ = msg.get("type")
            if typ == "reset":
                agent.reset()
            elif typ == "hello":
                if [str(n) for n in msg.get("nodes", [])] != agent.nodes:
                    emit(out, err("hello.nodes differ from the model's nodes (order matters)"))
                if msg.get("source", "exact") != "exact":
                    emit(out, err("GO-FLOW needs speed and occupancy: run the town with source exact "
                                  "(camera reports have neither); no forecast will be made"))
            elif typ == "obs":
                if not isinstance(msg.get("t"), (int, float)) or isinstance(msg.get("t"), bool):
                    emit(out, err("obs.t must be a number", seq))
                    continue
                res, problems = agent.step(msg)
                for p in problems:
                    emit(out, err(p, seq))
                if res is not None:
                    emit(out, res)
            else:
                emit(out, err(f"unknown type {typ!r}", seq))
        except Exception as e:  # noqa: BLE001
            emit(out, err(f"{type(e).__name__}: {e}", seq))
            traceback.print_exc(file=log)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    sys.exit(serve(ap.parse_args(argv).model))


if __name__ == "__main__":
    main()
