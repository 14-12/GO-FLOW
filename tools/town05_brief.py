"""The first AI agent on top of steps 1 and 2: a short operator brief, written by a free LLM.

    python tools/town05_brief.py --dry-run     # shows the prompt and its size, calls nothing
    python tools/town05_brief.py               # one call (~1.5k tokens in, ~0.5k out)

It reads what steps 1 and 2 wrote (out/baseline_town05/baseline.json, out/town05_cameras/
coverage.json, and camera reports: out/town05_cameras/live/reports.jsonl if the live run was
made, else the simulated ones a max_pressure_walk_observed baseline run saved), folds them into
a few compact tables here in Python (the model never sees raw reports: that is what keeps a
call small), and asks the model for a brief in a fixed JSON shape. The model only words and
ranks; every number comes from the tables. A junction the model names that is not in the data
is flagged in the output.

The agent never touches the signals. Control stays with the junction controllers
(cosim/controllers.py); an LLM takes seconds per answer and can be wrong, which a signal cannot.
Written to out/reports/town05_brief.md and .json (with the provider, model and tokens used).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out" / "reports"

SCHEMA = {"name": "town05_brief", "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["headline", "findings", "hotspots", "next_checks"],
    "properties": {
        "headline": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["claim", "evidence"],
            "properties": {"claim": {"type": "string"}, "evidence": {"type": "string"}}}},
        "hotspots": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["junction", "reason"],
            "properties": {"junction": {"type": "string"}, "reason": {"type": "string"}}}},
        "next_checks": {"type": "array", "items": {"type": "string"}}}}}

SYSTEM = ("You are the traffic operations analyst for Town05, a simulated town (SUMO traffic, CARLA cameras). "
          "Write for the engineer who runs it: short, plain sentences, no marketing words. Use only the numbers in "
          "the tables; quote them as evidence. If the tables cannot answer something, say so instead of guessing. "
          "At most 5 findings, 4 hotspots, 4 next checks. Take each number from the row it belongs to (controller "
          "and demand both match), with its unit: stuck is a count of vehicles, not seconds.")


def reports_file() -> Path | None:
    live = ROOT / "out" / "town05_cameras" / "live" / "reports.jsonl"
    if live.is_file():
        return live
    sims = sorted((ROOT / "out" / "baseline_town05" / "runs").glob("max_pressure_walk_observed_*.reports.jsonl"))
    return sims[0] if sims else None


def camera_table(path: Path) -> tuple[str, str, float]:
    """Per junction over the reports: mean and peak queue on its incoming lanes, parked cars
    seen (median), emergency confirmations."""
    per: dict[str, dict] = {}
    was_on: dict[str, bool] = {}
    minutes = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        reps = row["reports"] if "reports" in row else [row]
        minutes = max(minutes, float(row.get("t", 0.0)) / 60.0)
        moment: dict[str, dict] = {}
        for r in reps:
            m = moment.setdefault(r["tls"], {"q_in": 0, "in": 0, "parked": 0, "ems": 0})
            m["q_in"] += r["q_in"]
            m["in"] += r["in"]
            m["parked"] += r["parked"]
            # a siren episode: a camera confirms a siren that it did not confirm in its previous report
            on = any(e["confirmed"] for e in r["ems"])
            if on and not was_on.get(r["cam"]):
                m["ems"] += 1
            was_on[r["cam"]] = on
        for cam in list(was_on):
            if cam not in {r["cam"] for r in reps}:  # sample files keep only cameras that saw something
                was_on[cam] = False
        for tls, m in moment.items():
            p = per.setdefault(tls, {"q": [], "n": [], "parked": [], "ems": 0})
            p["q"].append(m["q_in"])
            p["n"].append(m["in"])
            p["parked"].append(m["parked"])
            p["ems"] += m["ems"]
    lines = ["junction,mean_incoming_vehicles,mean_queue,peak_queue,parked_seen_median,siren_episodes"]
    for tls, p in sorted(per.items(), key=lambda kv: -st.mean(kv[1]["q"])):
        lines.append(f"{tls},{st.mean(p['n']):.1f},{st.mean(p['q']):.1f},{max(p['q'])},"
                     f"{int(st.median(p['parked']))},{p['ems']}")
    return "\n".join(lines), path.relative_to(ROOT).as_posix(), minutes


def build_prompt() -> tuple[list[dict], dict]:
    base = json.loads((ROOT / "out" / "baseline_town05" / "baseline.json").read_text(encoding="utf-8"))
    cov = json.loads((ROOT / "out" / "town05_cameras" / "coverage.json").read_text(encoding="utf-8"))
    runs = ROOT / "out" / "baseline_town05" / "runs"
    t1 = ["demand,controller,seeds,veh_waiting_s,veh_time_loss_s,throughput_per_h,ambulance_time_loss_s,"
          "ped_time_loss_s,stuck_mean,stuck_worst_seed"]
    for r in base:
        worst = max((json.loads(f.read_text(encoding="utf-8")).get("stuck_over_300s_at_end", 0)
                     for f in runs.glob(f"{r['controller']}_x{r['scale']:g}_seed*.json")), default=0)
        t1.append(f"x{r['scale']:g},{r['controller']},{len(r['seeds'])},{r['veh_waiting_s'][0]},"
                  f"{r['veh_time_loss_s'][0]},{r['throughput_per_h'][0]},{r['ems_time_loss_s'][0]},"
                  f"{r['ped_time_loss_s'][0]},{r['stuck'][0]},{worst}")
    # The recommendation is decided here, not by the model (it once recommended max-pressure for having 0
    # vehicles stuck at x0.5, which actuated also has, at half the waiting): the controller that never locks
    # (at most 10 vehicles stuck on its worst seed, every demand tested), then the lowest mean waiting.
    by_ctl: dict[str, list[tuple[float, int, float]]] = {}
    for r in base:
        worst = max((json.loads(f.read_text(encoding="utf-8")).get("stuck_over_300s_at_end", 0)
                     for f in runs.glob(f"{r['controller']}_x{r['scale']:g}_seed*.json")), default=0)
        by_ctl.setdefault(r["controller"], []).append((r["scale"], worst, r["veh_waiting_s"][0]))
    demands = {r["scale"] for r in base}
    safe = {c: v for c, v in by_ctl.items() if {x[0] for x in v} == demands and all(w <= 10 for _, w, _ in v)}
    rec = min(safe, key=lambda c: st.mean(x[2] for x in safe[c])) if safe else None
    decision = ("none of the controllers stays unlocked on every seed and demand" if rec is None else
                f"{rec}: the only controller with at most 10 vehicles stuck on every seed at every demand tested"
                if len(safe) == 1 else
                f"{rec}: of the controllers with at most 10 vehicles stuck on every seed at every demand "
                f"({', '.join(sorted(safe))}), the lowest mean waiting")
    t2 = ["junction,cameras,incoming_seen_share,outgoing_seen_share"]
    for tls, v in sorted(cov["junctions"].items(), key=lambda kv: kv[1]["in_share"] or 0):
        t2.append(f"{tls},{v['cameras']},{v['in_share']},{v['out_share']}")
    rf = reports_file()
    t3, src, minutes = camera_table(rf) if rf else ("(no camera reports yet)", None, 0.0)
    live = bool(src and "/live/" in src)
    # Table 4: the junctions worth naming, picked here, not by the model (it once called a junction
    # with 4 cameras, the most any has, "limited coverage")
    flags: dict[str, list[str]] = {}
    most = max(v["cameras"] for v in cov["junctions"].values())
    for tls, v in cov["junctions"].items():
        if (v["in_share"] or 0) < 0.70:
            flags.setdefault(tls, []).append(f"weak camera coverage: {v['in_share']:.0%} of incoming seen, "
                                             f"{v['cameras']} cameras (the best-equipped have {most})")
    rows = [ln.split(",") for ln in t3.splitlines()[1:]] if src else []
    for row in sorted(rows, key=lambda x: (-int(x[3]), -float(x[2])))[:2]:
        flags.setdefault(row[0], []).append(f"longest queue in the camera reports: peak {row[3]}, mean {row[2]}")
    for row in sorted(rows, key=lambda x: -int(x[5]))[:1]:
        if int(row[5]) > 0:
            flags.setdefault(row[0], []).append(f"most siren episodes: {row[5]} (ambulance routes, not a fault)")
    t4 = "\n".join(f"{tls}: " + "; ".join(why) for tls, why in sorted(flags.items()))
    user = (
        "Table 1, signal controllers compared in SUMO, 1 simulated hour per run. Times are seconds per trip; "
        "stuck = vehicles standing still over 300 s at the end of the hour (a locked network); demand = scale of "
        "the scenario's routes. max_pressure_walk_observed is max-pressure fed by the cameras' reports instead of "
        "exact counts.\n" + "\n".join(t1) +
        f"\n\nTable 2, share of the last {cov['reach_m']:g} m before each stop line that lies inside some camera's "
        "picture. It depends only on how many cameras a junction has and where they point (road geometry; "
        "buildings ignored), not on the detector.\n" + "\n".join(t2) +
        f"\n\nTable 3, camera reports per junction over {minutes:.0f} minutes, one report per camera every 2 s. "
        + ("From the real detector on CARLA images." if live else
           "Simulated: SUMO's truth seen through each camera's visible lane stretches with the detector's measured "
           "miss and false-box rates, during the max_pressure_walk_observed run (x0.5, seed 1), so these queues "
           "are that controller's, not actuated's.")
        + " siren_episodes = times a camera started confirming a siren (one ambulance passing two cameras counts "
          "twice); ambulances follow their routes, so this shows where they drove, not a fault.\n" + t3 +
        "\n\nDecision, computed from Table 1: run " + decision + "." +
        "\n\nTable 4, the junctions flagged from the tables above (the only ones to list as hotspots, with "
        "these reasons):\n" + t4 +
        "\n\nWhat the engineer can do next in this project: run the live camera check at junctions 1126, 1882, "
        "2296 (real detector vs truth); add or move cameras at a junction; build step 3, the junction agents "
        "(switching cost, neighbours warning each other about full exits, actuated fallback where coverage is "
        "weak, emergency pre-emption); rerun the baseline at other demands or seeds.\n\n"
        "Write the brief. headline: one sentence that states the decision above and why, with its numbers from "
        "Table 1. "
        "Findings: the most important first, each with its numbers; say when a result holds on some seeds but "
        "not all (stuck_mean vs stuck_worst_seed). Hotspots: only the junctions of Table 4, with its reason. Next checks: chosen from the list above, each saying what it would settle.")
    return ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            {"known": set(flags), "reports": src, "text": user, "recommended": rec})


def _num(x: str) -> float:
    return round(float(x), 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=3000, help="room for the reply, the model's reasoning included")
    a = ap.parse_args(argv)
    messages, meta = build_prompt()
    from cosim.llm import LLM, estimate_tokens

    if a.dry_run:
        for m in messages:
            print(f"--- {m['role']}\n{m['content']}\n")
        print(f"about {estimate_tokens(messages, a.max_tokens)} tokens with the reply")
        return 0
    llm = LLM()
    r = llm.chat(messages, max_tokens=a.max_tokens, json_schema=SCHEMA)
    brief = r.json()
    unknown = [h["junction"] for h in brief["hotspots"] if h["junction"] not in meta["known"]]
    choice = meta["recommended"]
    off_decision = bool(choice) and not re.search(rf"\b{re.escape(choice)}\b", brief["headline"], re.I)
    # every number the model quotes must be in what it was given
    given = {_num(x) for x in re.findall(r"\d+(?:\.\d+)?", meta["text"])}
    quoted = [x for f in brief["findings"] for x in re.findall(r"\d+(?:\.\d+)?", f["claim"] + " " + f["evidence"])]
    made_up = sorted({x for x in quoted if _num(x) not in given and not re.fullmatch(r"0\.[58]", x)})
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"brief": brief, "provider": r.provider, "model": r.model, "usage": r.usage, "seconds": round(r.seconds, 1),
           "camera_reports": meta["reports"], "hotspots_not_flagged": unknown, "numbers_not_in_data": made_up, "recommended": choice,
           "headline_off_decision": off_decision,
           "written": time.strftime("%Y-%m-%d %H:%M")}
    (OUT / "town05_brief.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    md = [f"# Town05 brief, {rec['written']}", "", f"**{brief['headline']}**", "", "## Findings", ""]
    md += [f"- {f['claim']} ({f['evidence']})" for f in brief["findings"]]
    md += ["", "## Junctions to watch", ""] + [f"- {h['junction']}: {h['reason']}" for h in brief["hotspots"]]
    md += ["", "## Next checks", ""] + [f"- {c}" for c in brief["next_checks"]]
    md += ["", f"_Written by {r.model} ({r.provider}), {r.usage.get('total_tokens', '?')} tokens, "
               f"{r.seconds:.1f} s. Data: out/baseline_town05/baseline.json, out/town05_cameras/coverage.json"
               + (f", {meta['reports']}" if meta["reports"] else "") + "._"]
    if unknown:
        md.append(f"\n**Warning:** the model listed junctions the data does not flag: {', '.join(unknown)}.")
    if off_decision:
        md.append(f"\n**Warning:** the headline does not name the computed choice, {choice}.")
    if made_up:
        md.append(f"\n**Warning:** numbers in the findings that are not in the data: {', '.join(made_up)}.")
    (OUT / "town05_brief.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
