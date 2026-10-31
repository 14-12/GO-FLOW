"""Locating the SUMO binaries that ship in the `eclipse-sumo` wheel (same pin as the main project)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def sumo_home() -> Path:
    """SUMO_HOME if set, else the directory of the installed `eclipse-sumo` wheel."""
    env = os.environ.get("SUMO_HOME")
    if env:
        return Path(env)
    import sumo  # the eclipse-sumo wheel

    return Path(sumo.SUMO_HOME)


def binary(name: str) -> str:
    """Path of a SUMO executable (`sumo`, `sumo-gui`, `netconvert`)."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    p = sumo_home() / "bin" / exe
    if p.is_file():
        return str(p)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found under {sumo_home()} or on PATH")
