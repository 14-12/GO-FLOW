"""Assemble le dashboard autonome (un seul fichier HTML, données incluses).

Usage : python -m evaluation.build_dashboard
Sortie : dashboard/dashboard.html (à ouvrir dans un navigateur)
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
RES = ROOT / "results"


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ("emg_detail",) if c in df])
    return json.loads(df.to_json(orient="records"))


def build() -> Path:
    data = {"generated": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
            "corridor": _records(RES / "eval_corridor.csv"), "single": _records(RES / "eval_single.csv"),
            "dispatch": _records(RES / "eval_dispatch.csv")}
    if (RES / "stats_tests.csv").exists():
        data["stats"] = json.loads(pd.read_csv(RES / "stats_tests.csv").to_json(orient="records"))
    prog = ROOT / "outputs" / "training" / "progress.csv"
    if prog.exists():
        p = pd.read_csv(prog)
        p["roll"] = p.reward.rolling(100, min_periods=20).mean()
        p = p.dropna(subset=["roll"])
        step = max(1, len(p) // 300)
        data["training"] = [[int(a), round(float(b), 2)] for a, b in zip(p.timesteps[::step], p.roll[::step])]
    meta = ROOT / "models" / "checkpoints" / "ppo_signal.json"
    if meta.exists():
        data["train_meta"] = json.loads(meta.read_text())
    rp = RES / "replay"
    if (rp / "network.json").exists():
        runs = {}
        files = sorted(rp.glob("*__*.json"), key=lambda f: f.stat().st_mtime)
        latest = files[-1].name.split("__")[0] if files else None   # rediffusion la plus récente seulement
        for f in files:
            if f.name.split("__")[0] != latest:
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            runs[r["config"]] = r
        data["replay"] = {"network": json.loads((rp / "network.json").read_text()), "runs": runs}
    try:
        from importlib.metadata import version
        data["sumo_version"] = version("eclipse-sumo")
    except Exception:
        pass
    tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    body = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    out = ROOT / "dashboard" / "dashboard.html"
    out.write_text('<!doctype html>\n<html lang="fr"><head><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n'
                   + body + "\n</body></html>\n", encoding="utf-8")
    (ROOT / "dashboard" / "dashboard_fragment.html").write_text(body, encoding="utf-8")
    print("dashboard :", out, f"({out.stat().st_size / 1e6:.1f} Mo)")
    return out


if __name__ == "__main__":
    build()
