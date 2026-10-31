"""Puts the plate files into CARLA's content, or takes them out again (23 Oct).

    python tools/plates/install_plates.py install   the viewer's state: SM_TnPlate (our plate mesh) and SM_TnDeck (the
                                                    recovery truck's deck, 24 Oct), the cars' own
                                                    plates plain black (they show only round the edges of ours)
    python tools/plates/install_plates.py probe     as install, but the cars' own plates magenta (probe_plates2)
    python tools/plates/install_plates.py restore   CARLA exactly as shipped (plate texture, materials, the 126
                                                    unused plate textures) and no SM_TnPlate
    python tools/plates/install_plates.py status

The originals are in carla_backup/ (Licenseplates_original, materials_original); every file is checked against
its md5 after copying. A fresh clone has no carla_backup/: the first `install` copies CARLA's own files there before
it replaces anything (31 Oct, `ensure_backup`). The plate materials are always put back as shipped: the parameter renamed on 23 Oct
morning ("LIcenseText" -> "BaseColor") was for the runtime texture API, which can't reach the cars' plates
(they are part of each car's skeletal mesh).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTENT = ROOT / "carla" / "CARLA_0.9.16" / "CarlaUE4" / "Content" / "Carla" / "Static"
LP = CONTENT / "GenericMaterials" / "Licenseplates"
BACKUP = ROOT / "carla_backup"
MATERIALS = {"M_LicensePlate_Master": CONTENT / "GenericMaterials" / "Vehicles",
             "M_LicensePlate": CONTENT / "GenericMaterials" / "Masters"}
MESH_DIR = CONTENT / "Other"
MESHES = ("SM_TnPlate", "SM_TnDeck")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def put(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if md5(src) != md5(dst):
        raise OSError(f"copy of {src} to {dst} differs")


def ensure_backup() -> None:
    """Copies CARLA's own plate files to carla_backup/ when it is not there yet (a fresh clone: the folder is in
    .gitignore), so `restore` can put them back. Only while CARLA's plate texture is still not one of ours."""
    lp_bak, mat_bak = BACKUP / "Licenseplates_original", BACKUP / "materials_original"
    if lp_bak.is_dir() and mat_bak.is_dir():
        return
    if not LP.is_dir():
        raise SystemExit(f"CARLA's plate folder not found: {LP} (is CARLA in carla/CARLA_0.9.16? tools/download_carla.bat)")
    ours = {md5(f) for k in ("black", "magenta") for f in (HERE / "payload" / f"T_LicensePlate_d_{k}").glob("*.ubulk")}
    if md5(LP / "T_LicensePlate_d.ubulk") in ours:
        print("carla_backup/ missing and CARLA's plate texture is already ours: no backup made")
        return
    if not lp_bak.is_dir():
        shutil.copytree(LP, lp_bak)
    if not mat_bak.is_dir():
        mat_bak.mkdir(parents=True)
        for name, folder in MATERIALS.items():
            for ext in (".uasset", ".uexp"):
                put(folder / f"{name}{ext}", mat_bak / f"{name}{ext}")
    print(f"CARLA's own plate files kept in {BACKUP}")


def restore_shipped() -> int:
    n = 0
    for f in (BACKUP / "Licenseplates_original").rglob("*"):
        if f.is_file():
            dst = LP / f.relative_to(BACKUP / "Licenseplates_original")
            if not dst.exists() or md5(dst) != md5(f):
                put(f, dst)
                n += 1
    for name, folder in MATERIALS.items():
        for ext in (".uasset", ".uexp"):
            src = BACKUP / "materials_original" / f"{name}{ext}"
            dst = folder / f"{name}{ext}"
            if src.exists() and (not dst.exists() or md5(dst) != md5(src)):
                put(src, dst)
                n += 1
    return n


def install(texture: str) -> None:
    ensure_backup()
    n = restore_shipped()
    for mesh in MESHES:  # our plate mesh and the recovery truck's deck (tools/plates/make_deck_mesh.py)
        for f in (HERE / "payload" / mesh).iterdir():
            put(f, MESH_DIR / f.name)
    for f in (HERE / "payload" / f"T_LicensePlate_d_{texture}").iterdir():
        put(f, LP / f.name)
    print(f"installed: {', '.join(MESHES)}, cars' own plates {texture} ({n} shipped files put back first)")


def status() -> None:
    mesh = all(md5(MESH_DIR / f.name) == md5(f) for m in MESHES for f in (HERE / "payload" / m).iterdir()
               if (MESH_DIR / f.name).exists()) and all((MESH_DIR / f"{m}.uasset").exists() for m in MESHES)
    tex = md5(LP / "T_LicensePlate_d.ubulk")
    kinds = {md5(HERE / "payload" / f"T_LicensePlate_d_{k}" / "T_LicensePlate_d.ubulk"): k for k in ("black", "magenta")}
    if not (BACKUP / "materials_original").is_dir():
        print(f"{' + '.join(MESHES)}: {'yes' if mesh else 'no'}; no carla_backup/ yet (made by the first install)")
        return
    kinds[md5(BACKUP / "Licenseplates_original" / "T_LicensePlate_d.ubulk")] = "as shipped"
    mats = all(md5(folder / f"{name}.uasset") == md5(BACKUP / "materials_original" / f"{name}.uasset")
               for name, folder in MATERIALS.items())
    print(f"{' + '.join(MESHES)}: {'yes' if mesh else 'no'}; cars' plate texture: {kinds.get(tex, 'other (test pattern?)')}; "
          f"plate materials as shipped: {mats}")


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"
    if cmd == "install":
        install("black")
    elif cmd == "probe":
        install("magenta")
    elif cmd == "restore":
        n = restore_shipped()
        for mesh in MESHES:
            for ext in (".uasset", ".uexp"):
                p = MESH_DIR / f"{mesh}{ext}"
                if p.exists():
                    dst = p.with_name(p.name + ".removed")
                    if dst.exists():
                        dst.unlink()
                    p.rename(dst)  # renamed out of CARLA's sight
        print(f"restored as shipped ({n} files put back)")
    status()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
