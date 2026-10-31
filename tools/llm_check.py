"""Which of the configured free models answer with the keys in .env (a few tokens each).

    python tools/llm_check.py           # every provider in config/llm.json, one tiny call each
    python tools/llm_check.py --status  # only what today used and who is cooling down, no call

Keys are never printed: only whether one was found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim.llm import LLM  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)
    llm = LLM()
    if a.status:
        print(json.dumps(llm.status(), indent=1))
        return 0
    ok = 0
    for p in llm.providers:
        single = LLM.__new__(LLM)
        single.__dict__.update(llm.__dict__)
        single.providers = [p]
        try:
            r = single.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=64, wait_s=0)
            print(f"{p.name:24s} OK   {r.seconds:5.1f} s  {r.usage.get('total_tokens', '?')} tokens  -> {r.text.strip()[:30]!r}")
            ok += 1
        except Exception as e:  # noqa: BLE001 - report and go on
            print(f"{p.name:24s} --   {e}")
    print(f"{ok} of {len(llm.providers)} providers answer")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
