"""Puts a render profile (tools/fps_test/engine_<name>.ini) into CARLA's user Engine.ini, or takes it out.

    python tools/ini_profile.py install medium
    python tools/ini_profile.py remove

The profile's [SystemSettings] keys are read only when CARLA starts. CARLA (Unreal) also writes its user
Engine.ini back when its window is closed normally, keeping whatever [SystemSettings] it read: found on
19 Oct, when the medium profile had stayed in the file after a free-fly session, so a backup taken before
the next install already held it. So nothing is backed up: install takes out every key any profile sets
and adds the profile's; remove takes them out. The rest of the file (Unreal's own sections) is kept.
Both places Unreal may read it from are handled: %LOCALAPPDATA%\\CarlaUE4\\Saved and the CARLA folder's.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "tools" / "fps_test"


def saved_dirs() -> list[Path]:
    out = [ROOT / "carla" / "CARLA_0.9.16" / "CarlaUE4" / "Saved" / "Config" / "WindowsNoEditor"]
    if os.environ.get("LOCALAPPDATA"):
        out.insert(0, Path(os.environ["LOCALAPPDATA"]) / "CarlaUE4" / "Saved" / "Config" / "WindowsNoEditor")
    return out


def profile_lines(name: str) -> list[str]:
    return [ln.strip() for ln in (PROFILES / f"engine_{name}.ini").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith(";") and not ln.strip().startswith("[")]


def managed_keys() -> set[str]:
    keys = set()
    for f in PROFILES.glob("engine_*.ini"):
        keys |= {ln.split("=", 1)[0].strip().lower() for ln in profile_lines(f.stem[len("engine_"):]) if "=" in ln}
    return keys


def strip(text: str, keys: set[str]) -> str:
    """The file without our keys in [SystemSettings], our comment lines, or an emptied section."""
    out, section, block = [], None, []

    def flush():
        if section == "systemsettings":
            body = [ln for ln in block[1:] if ln.strip()]
            if body:
                out.extend(block)
        else:
            out.extend(block)

    for ln in text.splitlines():
        m = re.match(r"\s*\[(.+)\]\s*$", ln)
        if m:
            flush()
            section, block = m.group(1).strip().lower(), [ln]
            continue
        if ln.strip().startswith("; Town05") or (ln.strip().startswith(";") and section == "systemsettings"):
            continue
        if section == "systemsettings" and "=" in ln and ln.split("=", 1)[0].strip().lower() in keys:
            continue
        if section is None and not block:
            block = [ln] if ln.strip() else []
            section = "" if ln.strip() else None
            continue
        block.append(ln)
    flush()
    return "\n".join(out).strip("\n") + "\n" if any(s.strip() for s in out) else ""


def apply(name: str | None) -> None:
    keys = managed_keys()
    for d in saved_dirs():
        f = d / "Engine.ini"
        text = f.read_text(encoding="utf-8", errors="replace") if f.is_file() else ""
        new = strip(text, keys)
        if name:
            new = (new.rstrip("\n") + "\n\n" if new else "") + "[SystemSettings]\n" + "\n".join(profile_lines(name)) + "\n"
        if new != text and (new or f.is_file()):
            d.mkdir(parents=True, exist_ok=True)
            f.write_text(new, encoding="utf-8")
        print(f"{f}: {'profile ' + name if name else 'no profile'}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "install" and len(sys.argv) == 3:
        apply(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "remove":
        apply(None)
    else:
        print(__doc__)
        sys.exit(2)
