"""The police office (31 Oct): where an officer reviews the red-light cases and issues the notices.

    python tools/police_office.py [--cases out/enforcement/cases] [--register out/enforcement/register.json]

The camera's agent (cosim/redlight.py) only proposes. As in real life, an officer looks at each case before anything
is sent: the two photos (before the line on red, then in the junction), the clip, the plate as the camera read it and
how sure it was, and what the vehicle register says about that plate: its make, model and colour, to compare with the
photo (a misread or a copied plate shows there), and its registered keeper. Then:
- Accept: the officer confirms (or corrects) the plate; a notice to the registered keeper is made (notice.html, to
  print or send) and the decision logged with the case;
- Reject, with a reason (plate unreadable, vehicle does not match the register, emergency vehicle, signal not red,
  other): nothing is sent; the decision is logged.
Every file of a case carries its SHA-256 in case.json; the office checks them and shows whether the evidence is as
the camera wrote it.

THE REGISTER'S PEOPLE ARE FICTIONAL (cosim/register.py), and so is every notice made here: a simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FINE_TND = 60          # simulated amount: the highest of the 20 / 40 / 60 dinar classes of the 2025 reform proposal
PAY_DAYS = 30
REASONS = ["plate unreadable", "vehicle does not match the register", "emergency vehicle on a run",
           "signal not red / within tolerance", "other"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Office:
    def __init__(self, cases: Path, register: Path):
        from cosim.register import Register

        self.dir = cases
        self.register = Register(register)
        self.reg_path = register

    def all(self) -> list[dict]:
        out = []
        for d in sorted(self.dir.glob("*/case.json")):
            try:
                rec = json.loads(d.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            dec = d.parent / "decision.json"
            rec["_dir"] = d.parent
            rec["_decision"] = json.loads(dec.read_text(encoding="utf-8")) if dec.is_file() else None
            out.append(rec)
        return out

    def integrity(self, rec: dict) -> dict[str, bool]:
        return {name: (rec["_dir"] / name).is_file() and sha256(rec["_dir"] / name) == h
                for name, h in rec.get("files", {}).items()}

    def lookup(self, plate: str | None) -> dict | None:
        if not plate:
            return None
        if self.reg_path.is_file() and self.reg_path.stat().st_mtime > getattr(self, "_reg_t", 0):
            from cosim.register import Register

            self.register = Register(self.reg_path)
            self._reg_t = self.reg_path.stat().st_mtime
        return self.register.lookup(plate)

    def decide(self, rec: dict, accept: bool, plate: str | None, reason: str | None, officer: str) -> dict:
        n = len([r for r in self.all() if r["_decision"] and r["_decision"].get("notice")]) + 1
        dec = {"case": rec["case"]["id"], "accepted": accept, "plate": plate, "reason": reason, "officer": officer,
               "decided_at": time.strftime("%Y-%m-%d %H:%M:%S"), "integrity": self.integrity(rec)}
        if accept:
            entry = self.lookup(plate)
            dec["notice"] = f"SIM-{time.strftime('%Y')}-{n:05d}"
            dec["register"] = entry
            path = rec["_dir"] / "notice.html"
            path.write_text(notice_html(rec, dec, entry), encoding="utf-8")
            dec["notice_file"] = path.name
        (rec["_dir"] / "decision.json").write_text(json.dumps(dec, indent=1, ensure_ascii=False), encoding="utf-8")
        return dec


def notice_html(rec: dict, dec: dict, entry: dict | None) -> str:
    c = rec["case"]
    e = entry or {}
    v, o = e.get("vehicle", {}), e.get("owner", {})
    esc = html.escape
    img = "".join(f'<figure><img src="{n}"><figcaption>{cap}</figcaption></figure>'
                  for n, cap in (("photo_a.jpg", "Photo A: before the stop line, signal red"),
                                 ("photo_b.jpg", "Photo B: in the junction"), ("plate.png", "The plate")) if n in
                  rec.get("files", {}))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Notice {esc(dec['notice'])}</title>
<style>body{{font-family:Arial,sans-serif;max-width:820px;margin:24px auto;color:#111}}
.sim{{background:#fde8e8;border:2px solid #c22;padding:8px 12px;font-weight:bold}}
h1{{font-size:22px}} table{{border-collapse:collapse;width:100%}} td{{border:1px solid #bbb;padding:6px 8px;vertical-align:top}}
td:first-child{{width:34%;background:#f4f4f4}} figure{{display:inline-block;width:48%;margin:6px 1%}} img{{width:100%}}
figcaption{{font-size:12px;color:#444}}</style></head><body>
<p class="sim">SIMULATION: this notice is made by the Urban Traffic CARLA project. The owner, the address and the
identity number are fictional; no real person is concerned.</p>
<h1>Notice of a traffic offence: red light (simulated)</h1>
<table>
<tr><td>Notice number</td><td>{esc(dec['notice'])}</td></tr>
<tr><td>Offence</td><td>Entering a junction against a red signal</td></tr>
<tr><td>Place</td><td>Town05, junction {esc(c['site'].split('-')[0])}, road {esc(c['site'].split('-', 1)[1])}, lane {esc(c['lane'])}</td></tr>
<tr><td>Date and time</td><td>{esc(rec.get('when', ''))}</td></tr>
<tr><td>Measured</td><td>front crossed the stop line {c['red_s']:.2f} s after the signal turned red (tolerance {rec.get('grace_s', 0.5)} s); speed {c['speed_kmh']:.0f} km/h</td></tr>
<tr><td>Registration plate</td><td>{esc(dec['plate'] or '')}</td></tr>
<tr><td>Vehicle (register)</td><td>{esc(v.get('make', '?'))} {esc(v.get('model', ''))}, {esc(v.get('colour', '?'))}, first registered {esc(str(v.get('first_registered', '?')))}</td></tr>
<tr><td>Registered keeper (fictional)</td><td>{esc(o.get('name', '?'))}<br>{esc(o.get('address') or '')}<br>ID {esc(o.get('id') or '-')}</td></tr>
<tr><td>Amount (simulated)</td><td>{FINE_TND} TND, to be paid within {PAY_DAYS} days</td></tr>
<tr><td>Reviewed by</td><td>{esc(dec['officer'])}, {esc(dec['decided_at'])}</td></tr>
<tr><td>Evidence</td><td>case {esc(c['id'])}; files checked against their SHA-256: {'all as recorded' if all(dec['integrity'].values()) else 'NOT ALL AS RECORDED'}</td></tr>
</table>
<p>If you were not driving, you may name the driver; you may also contest this notice within the time given.</p>
{img}
</body></html>"""


def run_gui(office: Office, officer: str) -> None:
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Police office: red-light cases (simulation)")
    root.geometry("1500x900")
    left = ttk.Frame(root, padding=6)
    left.pack(side="left", fill="y")
    right = ttk.Frame(root, padding=6)
    right.pack(side="left", fill="both", expand=True)
    tv = ttk.Treeview(left, columns=("when", "plate", "status"), show="headings", height=40)
    for col, w in (("when", 150), ("plate", 110), ("status", 110)):
        tv.heading(col, text=col)
        tv.column(col, width=w)
    tv.pack(fill="y", expand=True)
    ttk.Button(left, text="Refresh", command=lambda: refresh()).pack(fill="x", pady=4)
    banner = tk.Label(right, text="SIMULATION: the register's people are fictional", bg="#fde8e8", fg="#a00",
                      font=("Arial", 10, "bold"))
    banner.pack(fill="x")
    photos = ttk.Frame(right)
    photos.pack(fill="x")
    pa, pb = tk.Label(photos), tk.Label(photos)
    pa.grid(row=0, column=0, padx=4)
    pb.grid(row=0, column=1, padx=4)
    mid = ttk.Frame(right)
    mid.pack(fill="x", pady=4)
    pv, pp = tk.Label(mid), tk.Label(mid)
    pv.grid(row=0, column=0, padx=4)
    pp.grid(row=0, column=1, padx=4)
    info = tk.Text(right, height=16, font=("Consolas", 10))
    info.pack(fill="both", expand=True)
    bar = ttk.Frame(right)
    bar.pack(fill="x", pady=4)
    ttk.Label(bar, text="Plate:").pack(side="left")
    plate_var = tk.StringVar()
    ttk.Entry(bar, textvariable=plate_var, width=16).pack(side="left", padx=4)
    ttk.Button(bar, text="Look up", command=lambda: show(current[0], plate_var.get())).pack(side="left")
    ttk.Button(bar, text="Play clip", command=lambda: play()).pack(side="left", padx=8)
    ttk.Button(bar, text="Accept and issue notice", command=lambda: decide(True)).pack(side="left", padx=8)
    reason = tk.StringVar(value=REASONS[0])
    ttk.Combobox(bar, textvariable=reason, values=REASONS, width=34, state="readonly").pack(side="left", padx=4)
    ttk.Button(bar, text="Reject", command=lambda: decide(False)).pack(side="left")
    cases: list[dict] = []
    current: list = [None]
    keep: list = []

    def img(path: Path, w: int):
        if not path.is_file():
            return None
        im = Image.open(path)
        im.thumbnail((w, w))
        ph = ImageTk.PhotoImage(im)
        keep.append(ph)
        return ph

    def status(rec):
        d = rec["_decision"]
        if d is None:
            return "PENDING"
        return f"NOTICE {d['notice']}" if d.get("accepted") else "rejected"

    def refresh():
        nonlocal cases
        cases = office.all()
        tv.delete(*tv.get_children())
        for i, rec in enumerate(cases):
            c = rec["case"]
            tv.insert("", "end", iid=str(i), values=(rec.get("when", ""), c.get("plate") or "?", status(rec)))

    def show(rec, plate=None):
        if rec is None:
            return
        current[0] = rec
        keep.clear()
        d = rec["_dir"]
        for lab, name, w in ((pa, "photo_a.jpg", 560), (pb, "photo_b.jpg", 560), (pv, "vehicle.jpg", 300),
                             (pp, "plate.png", 360)):
            ph = img(d / name, w)
            lab.configure(image=ph if ph else "", text="" if ph else f"({name} missing)")
        c = rec["case"]
        p = plate if plate is not None else (rec["_decision"] or {}).get("plate") or c.get("plate") or ""
        plate_var.set(p)
        entry = office.lookup(p)
        ok = office.integrity(rec)
        lines = [f"Case {c['id']}   ({status(rec)})", f"When: {rec.get('when')}   site {c['site']}, lane {c['lane']}",
                 f"Front crossed the stop line {c['red_s']:.2f} s into red (tolerance {rec.get('grace_s')} s), "
                 f"{c['speed_kmh']:.0f} km/h, vehicle seen as: {c['kind']}",
                 f"Plate as read by the camera: {c.get('plate') or '(not read)'}   confidence {c.get('plate_conf', 0):.2f}",
                 "   readings: " + ", ".join(f"{r['text'] or '-'} ({r['conf']:.2f})" for r in c.get("plate_reads", [])[:6]),
                 "Evidence files: " + ", ".join(f"{k} {'OK' if v else 'CHANGED'}" for k, v in ok.items()), ""]
        if entry is None:
            lines.append(f"Register: no vehicle with plate '{p}'.")
        else:
            v, o = entry["vehicle"], entry["owner"]
            lines += [f"Register (FICTIONAL): {v['make']} {v['model']}, {v['colour']}, first registered "
                      f"{v.get('first_registered')}",
                      f"   keeper: {o['name']}   {o.get('address') or ''}   ID {o.get('id') or '-'}",
                      "   Compare the make, model and colour with the photos before accepting."]
        lines += ["", "Rules applied: " + rec.get("rules", "")]
        if rec["_decision"]:
            lines += ["", "Decision: " + json.dumps({k: v for k, v in rec["_decision"].items()
                                                      if k not in ("register", "integrity")})]
        info.delete("1.0", "end")
        info.insert("1.0", "\n".join(lines))

    def on_select(_):
        sel = tv.selection()
        if sel:
            show(cases[int(sel[0])])

    def play():
        rec = current[0]
        if rec is not None and (rec["_dir"] / "clip.mp4").is_file():
            if hasattr(os, "startfile"):
                os.startfile(str(rec["_dir"] / "clip.mp4"))  # noqa: S606 - the officer's own player
            else:
                webbrowser.open((rec["_dir"] / "clip.mp4").as_uri())

    def decide(accept: bool):
        rec = current[0]
        if rec is None or rec["_decision"] is not None:
            return
        dec = office.decide(rec, accept, plate_var.get().strip() or None, None if accept else reason.get(), officer)
        if accept and dec.get("notice_file"):
            webbrowser.open((rec["_dir"] / dec["notice_file"]).as_uri())
        sel = tv.selection()
        refresh()
        if sel:
            tv.selection_set(sel)
            show(cases[int(sel[0])])

    tv.bind("<<TreeviewSelect>>", on_select)
    refresh()

    def poll():  # new cases while the town runs
        n = len(list(office.dir.glob("*/case.json")))
        if n != len(cases):
            sel = tv.selection()
            refresh()
            if sel:
                tv.selection_set(sel)
        root.after(3000, poll)

    root.after(3000, poll)
    root.mainloop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(ROOT / "out" / "enforcement" / "cases"))
    ap.add_argument("--register", default=str(ROOT / "out" / "enforcement" / "register.json"))
    ap.add_argument("--officer", default="Officer on duty (you)")
    ap.add_argument("--list", action="store_true", help="print the cases and quit (no window)")
    a = ap.parse_args(argv)
    office = Office(Path(a.cases), Path(a.register))
    if a.list:
        for rec in office.all():
            c = rec["case"]
            print(c["id"], rec.get("when"), c.get("plate"), c.get("plate_conf"), rec["_decision"])
        return 0
    run_gui(office, a.officer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
