"""What a fresh clone's first start does to CARLA (31 Oct): tools/carla_keys.py and install_plates.ensure_backup."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


keys = _load("carla_keys", ROOT / "tools" / "carla_keys.py")
plates = _load("install_plates", ROOT / "tools" / "plates" / "install_plates.py")


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def test_shipped_keys_are_kept_aside_and_ours_put_in(tmp_path):
    ours = tmp_path / "ours.ini"
    ours.write_text("ours\n")
    cfg = tmp_path / "Config"
    cfg.mkdir()
    shipped = b"carla as shipped\n"
    (cfg / "DefaultInput.ini").write_bytes(shipped)
    assert keys.state(cfg, ours, _md5(shipped)) == "shipped"
    keys.install(cfg, ours, _md5(shipped))
    assert (cfg / "DefaultInput.ini").read_text() == "ours\n"
    assert (cfg / "DefaultInput.qwerty.ini").read_bytes() == shipped
    assert keys.install(cfg, ours, _md5(shipped)) == "already in place"
    keys.restore(cfg)
    assert (cfg / "DefaultInput.ini").read_bytes() == shipped


def test_some_other_keys_file_is_left_alone(tmp_path):
    ours = tmp_path / "ours.ini"
    ours.write_text("ours\n")
    cfg = tmp_path / "Config"
    cfg.mkdir()
    (cfg / "DefaultInput.ini").write_text("edited by hand\n")
    assert keys.install(cfg, ours, _md5(b"carla as shipped\n")).startswith("WARNING")
    assert (cfg / "DefaultInput.ini").read_text() == "edited by hand\n"
    assert not (cfg / "DefaultInput.qwerty.ini").exists()


def test_shipped_keys_file_is_the_one_ours_was_made_from():
    # ours differs from CARLA's only in commented-out movement lines and the reverse key
    ours = (ROOT / "tools" / "carla_config" / "DefaultInput.ini").read_text(encoding="utf-8").splitlines()
    assert any(ln.startswith(';+AxisMappings=(AxisName="MoveForward"') for ln in ours)
    assert not any(ln.startswith('+AxisMappings=(AxisName="MoveForward"') for ln in ours)


def test_first_install_keeps_carlas_own_plate_files(tmp_path, monkeypatch):
    lp = tmp_path / "Licenseplates"
    (lp / "Textures").mkdir(parents=True)
    (lp / "T_LicensePlate_d.ubulk").write_bytes(b"shipped texture")
    (lp / "Textures" / "0ANA101.uasset").write_bytes(b"a plate")
    mats = {}
    for name in plates.MATERIALS:
        folder = tmp_path / name
        folder.mkdir()
        for ext in (".uasset", ".uexp"):
            (folder / f"{name}{ext}").write_bytes(name.encode() + ext.encode())
        mats[name] = folder
    monkeypatch.setattr(plates, "LP", lp)
    monkeypatch.setattr(plates, "MATERIALS", mats)
    monkeypatch.setattr(plates, "BACKUP", tmp_path / "carla_backup")
    plates.ensure_backup()
    bak = tmp_path / "carla_backup"
    assert (bak / "Licenseplates_original" / "Textures" / "0ANA101.uasset").read_bytes() == b"a plate"
    assert (bak / "materials_original" / "M_LicensePlate.uexp").read_bytes() == b"M_LicensePlate.uexp"
    # a second run changes nothing, even after the texture was replaced
    (lp / "T_LicensePlate_d.ubulk").write_bytes(b"ours")
    plates.ensure_backup()
    assert (bak / "Licenseplates_original" / "T_LicensePlate_d.ubulk").read_bytes() == b"shipped texture"
