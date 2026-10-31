"""Wait until a CARLA server answers on host:port (or give up). Exit 0 when it does.

    python tools/wait_carla.py [--port 2000] [--max 240]
"""

import argparse
import sys
import time

import carla


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--max", type=float, default=240.0, help="seconds to wait at most")
    a = ap.parse_args(argv)
    t0 = time.time()
    while time.time() - t0 < a.max:
        try:
            c = carla.Client(a.host, a.port)
            c.set_timeout(5.0)
            v = c.get_server_version()
            print(f"CARLA {v} answering after {time.time() - t0:.0f} s", flush=True)
            return 0
        except RuntimeError:
            time.sleep(3)
    print(f"no CARLA on {a.host}:{a.port} after {a.max:.0f} s", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
