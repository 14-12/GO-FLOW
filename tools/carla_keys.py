"""CARLA's own key movement off for the free-fly viewer (31 Oct).

    python tools/carla_keys.py install    CARLA's CarlaUE4/Config/DefaultInput.ini -> tools/carla_config/DefaultInput.ini
    python tools/carla_keys.py restore    CARLA's file as it shipped again (from DefaultInput.qwerty.ini)
    python tools/carla_keys.py status

The viewer flies the camera itself (cosim/fly_proc.py: Z Q S D, A / E, Shift, the arrows). If CARLA's spectator
moved it too, on its own W A S D / Q E, the two would pull the camera back and forth. tools/carla_config/
DefaultInput.ini is CARLA 0.9.16's file with only its six movement lines commented out (and the vehicle reverse
moved from Q to X); the mouse look is CARLA's own and stays.

install: when CARLA's file is the one CARLA ships, it is kept beside it as DefaultInput.qwerty.ini and ours is copied
in (checked by md5). When it is ours already, nothing is done. When it is any other file, it is left alone and a
warning printed. tools/town05_freefly.bat runs `install` before every start.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "tools" / "carla_config" / "DefaultInput.ini"
CONFIG = ROOT / "carla" / "CARLA_0.9.16" / "CarlaUE4" / "Config"
SHIPPED_MD5 = "5ea3fae07273d325a3241b54ba7a1db8"  # CARLA 0.9.16's DefaultInput.ini


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def state(config: Path = CONFIG, ours: Path = OURS, shipped: str = SHIPPED_MD5) -> str:
    f = config / "DefaultInput.ini"
    if not f.is_file():
        return "missing"
    h = md5(f)
    return "ours" if h == md5(ours) else "shipped" if h == shipped else "other"


def install(config: Path = CONFIG, ours: Path = OURS, shipped: str = SHIPPED_MD5) -> str:
    s = state(config, ours, shipped)
    if s == "shipped":
        f, keep = config / "DefaultInput.ini", config / "DefaultInput.qwerty.ini"
        if not keep.exists():
            shutil.copyfile(f, keep)
        shutil.copyfile(ours, f)
        if md5(f) != md5(ours):
            raise OSError(f"copy of {ours} to {f} differs")
        return "installed (CARLA's own kept as DefaultInput.qwerty.ini)"
    if s == "ours":
        return "already in place"
    if s == "missing":
        return f"not found: {config / 'DefaultInput.ini'} (is CARLA in carla/CARLA_0.9.16? tools/download_carla.bat)"
    return ("WARNING: CARLA's DefaultInput.ini is neither as shipped nor ours: left alone. CARLA's own key movement "
            "may fight the viewer's (see tools/carla_config/DefaultInput.ini)")


def restore(config: Path = CONFIG) -> str:
    keep = config / "DefaultInput.qwerty.ini"
    if not keep.is_file():
        return "nothing to restore (no DefaultInput.qwerty.ini)"
    shutil.copyfile(keep, config / "DefaultInput.ini")
    return "CARLA's own DefaultInput.ini put back"


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"
    if cmd == "install":
        print(install())
    elif cmd == "restore":
        print(restore())
    print(f"CARLA's DefaultInput.ini: {state()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
