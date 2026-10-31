"""Write what every Town05 signal controls and the checks of its programs (cosim/signal_audit.py) as a report.

    python tools/audit_town05_signals.py                  # out/signal_audit/town05_signals.md and .json

Per junction: the approaches (lanes and the movements each lane may make, the CARLA pole showing them), every
signal link (movement or crossing) with the links it conflicts with and those it gives way to (SUMO's junction
logic, the right of way the simulated vehicles obey), each phase spelled out per approach, the crossings with their
length and the pedestrian clearance they need, the left turns' volumes against the oncoming traffic (cross product:
the usual test for a protected left-turn phase), and the result of every check for the static, actuated and Webster
programs. Exits 1 if any program fails a check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOWN05 = ROOT / "scenarios" / "town05"
OUT = ROOT / "out" / "signal_audit"
ROUTES = ("passenger", "delivery", "truck", "bus", "motorcycle")
# Left-turn phasing may be considered above these peak-hour cross products (left turns x opposing through, veh/h),
# urban street with 2 / 4 / 6 lanes: ADOT Traffic Engineering Guidelines and Processes, section 612 (2015)
CROSS_PRODUCT = {1: 75_000, 2: 150_000, 3: 225_000}
DIRS = {"s": "straight", "r": "right", "l": "left", "t": "U-turn"}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _phase_text(jn, state: str) -> str:
    parts = []
    by_app: dict[str, list[str]] = {}
    for lk in jn.links:
        if lk.kind == "vehicle" and state[lk.index] in "Gg":
            tag = DIRS.get(lk.dir, lk.dir) + ("" if state[lk.index] == "G" else " (yield)")
            if tag not in by_app.setdefault(lk.approach, []):
                by_app[lk.approach].append(tag)
    for a in sorted(by_app):
        parts.append(f"{a}: " + ", ".join(by_app[a]))
    walks = [lk.crossing.split("_")[-1] for lk in jn.links if lk.kind == "crossing" and state[lk.index] == "G"]
    if walks:
        parts.append("walk " + ", ".join(walks))
    flash = [lk.crossing.split("_")[-1] for lk in jn.links if lk.kind == "crossing" and state[lk.index] == "u"]
    if flash:
        parts.append("flashing don't walk " + ", ".join(flash))
    if not parts:
        parts.append("yellow" if "y" in state else "all red")
    return "; ".join(parts)


def main(argv=None) -> int:
    from cosim.signal_audit import PED_CLEAR_SPEED, audit, load
    from cosim.town import link_flows

    net = TOWN05 / "town05.net.xml"
    adds = [TOWN05 / f"town05.{p}.add.xml" for p in ("actuated", "webster", "coordinated")
            if (TOWN05 / f"town05.{p}.add.xml").is_file()]
    rep = audit(net, adds)
    js = load(net, adds)
    flows = link_flows(net, [TOWN05 / f"town05.{k}.rou.xml" for k in ROUTES if (TOWN05 / f"town05.{k}.rou.xml").is_file()])
    md = ["# Town05 signals: what each one controls, and the checks", "",
          "From `scenarios/town05/town05.net.xml` and the actuated and Webster programs (cosim/signal_audit.py). "
          "Approaches are named by the side vehicles come from. Conflicts and who gives way are SUMO's own junction "
          "logic. Checks: every link has a state in every phase; conflicting links green together only when one gives "
          "way; a priority green 'G' never gives way to a green; yellow >= 3 s and all red >= 1 s before a conflicting "
          f"green; flashing don't walk + buffer >= crossing / {PED_CLEAR_SPEED} m/s (MUTCD 3.5 ft/s), buffer >= 2 s; "
          "and (27 Oct) no green has to give way: no 'g', no two links that cross each other (cars or a crossing) "
          "green at once.", ""]
    failed = 0
    cp_rows = []
    for tid in sorted(rep, key=lambda t: int(t) if t.isdigit() else t):
        v, jn = rep[tid], js[tid]
        md += [f"## Junction {tid}", "", "| approach | lane: movements | CARLA pole (landmark) |", "|---|---|---|"]
        for a, ap in v["approaches"].items():
            lanes = "; ".join(f"{ln}: {'+'.join(DIRS.get(d, d) for d in ds)}" for ln, ds in ap["lanes"].items())
            md.append(f"| {a} | {lanes} | {', '.join(ap['landmarks']) or 'none (no CARLA light on this approach)'} |")
        md += ["", "| link | movement | conflicts with | gives way to |", "|---|---|---|---|"]
        for lk in jn.links:
            if lk.kind not in ("vehicle", "crossing"):
                continue
            what = (f"{lk.approach} {DIRS.get(lk.dir, lk.dir)} ({lk.from_lane} -> {lk.to_lane})" if lk.kind == "vehicle"
                    else f"crossing {lk.crossing} over {' / '.join(lk.crosses)}, {lk.length:.1f} m")
            md.append(f"| {lk.index} | {what} | {', '.join(map(str, v['conflicts'].get(str(lk.index), [])))} | "
                      f"{', '.join(map(str, v['gives_way_to'].get(str(lk.index), [])))} |")
        md += ["", "| phase | state | seconds (actuated min-max) | what runs |", "|---|---|---|---|"]
        act = jn.programs.get("actuated", [])
        for k, (s, d, _, nm) in enumerate(jn.programs["0"]):
            rng = ""
            if k < len(act):
                a_ph = v["programs"]["actuated"]["phases"][k]
                rng = f" ({a_ph['min']:g}-{max(a_ph['duration'], a_ph['min']):g})" if nm == "green" else ""
            md.append(f"| {k} {nm} | `{s}` | {d:g}{rng} | {_phase_text(jn, s)} |")
        md += ["", "| crossing | length (m) | clearance needed (s) | conflicting vehicle links |", "|---|---|---|---|"]
        for c in v["crossings"]:
            md.append(f"| {c['crossing']} | {c['length_m']} | {c['clearance_needed_s']} | "
                      f"{', '.join(map(str, c['conflicting_vehicle_links']))} |")
        md.append("")
        for pid, p in v["programs"].items():
            errs = p["errors"] + p.get("protected_errors", [])
            if errs:
                failed += 1
                md.append(f"**Program {pid}: {len(errs)} problem(s)**: " + "; ".join(errs[:6]))
            else:
                md.append(f"Program {pid}: all checks pass, no green has to give way"
                          + (f" (heads of one pole differ in {len(p['pole_disagreements'])} phase-pole cases: each "
                             "direction its own signal, cosim/signal_heads.py)." if p["pole_disagreements"] else "."))
        md.append("")
        f = flows.get(tid, {})
        vol: dict[str, dict[str, float]] = {}
        lanes_in: dict[str, set] = {}
        for lk in jn.links:
            if lk.kind == "vehicle":
                vol.setdefault(lk.approach, {}).setdefault(lk.dir, 0.0)
                vol[lk.approach][lk.dir] += f.get(lk.index, 0.0)
                lanes_in.setdefault(lk.approach, set()).add(lk.from_lane)
        for a, vv in vol.items():
            o = OPP[a]
            if vv.get("l", 0.0) > 0 and o in vol:
                opposing = vol[o].get("s", 0.0) + vol[o].get("r", 0.0)
                n_opp = len([ln for ln in lanes_in[o] if any(d in ("s", "r") for d in v["approaches"][o]["lanes"].get(ln, []))])
                cp_rows.append((tid, a, vv["l"], opposing, n_opp, v["approaches"][a]["exclusive_left_lane"]))
    md += ["## Left turns: protected phase or permissive?", "",
           "Scenario demand (x1.0, the routes' own hour). Cross product = left turns/h x opposing straight-on and "
           "right turns/h; left-turn phasing may be considered above 75 000 / 150 000 / 225 000 with 1 / 2 / 3 "
           "opposing lanes (urban; ADOT TGP 612). A protected left needs a lane of its own (FHWA Signal Timing Manual "
           "ch. 4: from a shared lane it can only be split phasing, one approach at a time).", "",
           "| junction | approach | left /h | opposing /h | opposing lanes | cross product x1.0 | x1.2 | threshold | "
           "left lane |", "|---|---|---|---|---|---|---|---|---|"]
    for tid, a, lft, opp, n, excl in sorted(cp_rows, key=lambda r: -r[2] * r[3]):
        cp = lft * opp
        thr = CROSS_PRODUCT.get(max(1, min(3, n)), 150_000)
        md.append(f"| {tid} | {a} | {lft:.0f} | {opp:.0f} | {n} | {cp:,.0f} | {cp * 1.44:,.0f} | {thr:,} | "
                  f"{'own' if excl else 'shared with straight on'} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "town05_signals.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "town05_signals.json").write_text(json.dumps(rep, indent=1, default=list), encoding="utf-8")
    print(f"{len(rep)} junctions, {sum(len(v['programs']) for v in rep.values())} programs, {failed} with problems; "
          f"report {OUT / 'town05_signals.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
