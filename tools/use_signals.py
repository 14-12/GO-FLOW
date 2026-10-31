"""Choose Town05's signal set (27 Oct): which built programs the simulation and the viewer use.

    python tools/use_signals.py approach      # (29 Oct, the default) as split, and at the T junctions too every
                                              # approach goes as one, so each CARLA pole shows one colour, its lamps lit
    python tools/use_signals.py split         # every green a free go, each approach on its own at the four-way
                                              # junctions; at the T junctions a lane may go while its neighbour waits
    python tools/use_signals.py leadlag       # every green a free go, the left lane held red while the opposite
                                              # side goes straight on: a switch more per road, jams at the rush
    python tools/use_signals.py permissive    # the 26 Oct programs: turns give way inside the junction
    python tools/use_signals.py               # which one is in use

Each set is three files in scenarios/town05 (town05.<set>.net.xml, .actuated.add.xml, .webster.add.xml) and, since
27 Oct, its coordinated programs (.coordinated.add.xml, cosim/coordination.py), copied over the ones in use
(town05.net.xml, town05.actuated.add.xml, town05.webster.add.xml, town05.coordinated.add.xml). Nothing is lost: the files
in use are first matched against the sets; if they are none of them, they are kept as town05.backup_<time>.*.
Measured (tools/check_junction_stops.py, the junction agents, 30 min): see README "Signals where no green gives way".
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOWN = ROOT / "scenarios" / "town05"
SETS = ("approach", "leadlag", "split", "permissive")
PARTS = ("net.xml", "actuated.add.xml", "webster.add.xml")
OPTIONAL = ("coordinated.add.xml",)  # since 27 Oct: copied with its set when the set has it
# the permissive set is the 26 Oct build, kept under its old name
FILE_SET = {"approach": "approach", "leadlag": "leadlag", "split": "split", "permissive": "opposites"}


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else ""


def in_use() -> str | None:
    cur = [_md5(TOWN / f"town05.{part}") for part in PARTS]
    for s in SETS:
        if all(_md5(TOWN / f"town05.{FILE_SET[s]}.{part}") == c for part, c in zip(PARTS, cur, strict=True)):
            return s
    return None


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    now = in_use()
    if not argv:
        print(f"signals in use: {now or 'none of ' + ', '.join(SETS)}")
        return 0
    want = argv[0].lower()
    if want not in SETS:
        print(f"unknown set {want!r}: one of {', '.join(SETS)}")
        return 2
    missing = [p for p in PARTS if not (TOWN / f"town05.{FILE_SET[want]}.{p}").is_file()]
    if missing:
        print(f"set {want} is missing {missing} in {TOWN}")
        return 1
    if now == want:
        print(f"signals in use: {want} (nothing to do)")
        return 0
    if now is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for part in PARTS:
            src = TOWN / f"town05.{part}"
            if src.is_file():
                shutil.copy2(src, TOWN / f"town05.backup_{stamp}.{part}")
        print(f"the files in use matched no set: kept as town05.backup_{stamp}.*")
    for part in PARTS:
        shutil.copy2(TOWN / f"town05.{FILE_SET[want]}.{part}", TOWN / f"town05.{part}")
    for part in OPTIONAL:
        src = TOWN / f"town05.{FILE_SET[want]}.{part}"
        if src.is_file():
            if (TOWN / f"town05.{part}").is_file() and now is None:
                shutil.copy2(TOWN / f"town05.{part}", TOWN / f"town05.backup_{time.strftime('%Y%m%d_%H%M%S')}.{part}")
            shutil.copy2(src, TOWN / f"town05.{part}")
    print(f"signals in use: {in_use()} (was {now or 'unknown'})")
    return 0 if in_use() == want else 1


if __name__ == "__main__":
    sys.exit(main())
