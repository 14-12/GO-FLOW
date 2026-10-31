"""Why do cars still stop inside a junction under the protected signals? (27 Oct, diagnostic)

    python tools/diag_junction_stops.py --controller agents --scale 0.8 --seconds 900

For every vehicle that stands (< 0.1 m/s) on a signalised junction's internal lane with no standing vehicle just
ahead of it, records once: the junction, the link it is on, that link's signal state now, the persons on the
junction's crossings that cross its path, the foe links green now, and SUMO's own reason (the leader, if any).
Also the signal monitor's checks (cosim/signal_monitor.py) and SUMO's collisions, kept. Writes
out/signal_checks/diag_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="agents")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--port", type=int, default=9900)
    ap.add_argument("--tag", default="diag")
    a = ap.parse_args(argv)

    import traci.constants as tc

    from cosim.agents_actuated import make as make_agents
    from cosim.signal_audit import load
    from cosim.signal_monitor import SignalMonitor
    from cosim.sumo_side import SumoSide

    town = ROOT / "scenarios" / "town05"
    js = load(town / "town05.net.xml", [])
    out_dir = ROOT / "out" / "signal_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    coll = out_dir / f"diag_{a.tag}.collisions.xml"
    policy = make_agents(a.controller) if a.controller.startswith("agents") else None
    s = SumoSide(town, a.controller if policy is None else "agents", seed=a.seed, port=a.port, label=f"diag{a.port}",
                 policy=policy, extras=True, emergency_every=120.0, vehicle_vars=[tc.VAR_LANE_ID],
                 extra_args=["--scale", f"{a.scale:g}", "--end", f"{a.seconds:g}", "--collision.check-junctions",
                             "true", "--collision.action", "warn", "--intermodal-collision.action", "warn",
                             "--collision-output", str(coll)])
    s.extras.config.pedestrian_trips = True
    if policy is not None:
        ex = s.extras
        policy.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    mon = SignalMonitor(town / "town05.net.xml")
    c = s.conn
    seen: set[str] = set()
    events = []
    # internal lane -> (tls, link index) from the links' via lanes
    via: dict[str, tuple[str, int]] = {}
    conns = list(ET.parse(town / "town05.net.xml").getroot().iter("connection"))
    for e in conns:
        if e.get("tl") and e.get("via") and e.get("tl") in js:
            via[e.get("via")] = (e.get("tl"), int(e.get("linkIndex")))
    for _ in range(3):  # internal lanes continuing an internal lane (an internal junction inside)
        for e in conns:
            fr = f"{e.get('from')}_{e.get('fromLane')}"
            if e.get("from", "").startswith(":") and fr in via and e.get("via"):
                via.setdefault(e.get("via"), via[fr])
    try:
        while s.time < a.seconds:
            r = s.step()
            tls = r.tls if hasattr(r, "tls") else {}
            try:
                mon.step(c, s.time, tls)
            except Exception:  # noqa: BLE001, S110 - a diagnostic: the monitor's own trouble is not the point
                pass
            for vid, sub in s.vehicle_subs.items():
                lane = sub.get(tc.VAR_LANE_ID, "")
                if not lane.startswith(":") or sub.get(tc.VAR_SPEED, 1.0) >= 0.1 or vid in seen:
                    continue
                j = lane[1:].rsplit("_", 2)[0]
                if j not in js or "_w" in lane or "_c" in lane:
                    continue
                leader = c.vehicle.getLeader(vid, 30.0)
                if leader is not None and leader[1] < 8.0 and c.vehicle.getSpeed(leader[0]) < 0.5:
                    continue
                seen.add(vid)
                tl_link = via.get(lane)
                st = tls.get(j, "")
                ev = {"t": round(s.time, 1), "veh": vid, "junction": j, "lane": lane,
                      "pos": round(c.vehicle.getLanePosition(vid), 1), "lane_len": round(c.lane.getLength(lane), 1),
                      "link": tl_link[1] if tl_link else None,
                      "link_state": st[tl_link[1]] if tl_link and tl_link[1] < len(st) else None,
                      "leader": [leader[0], round(leader[1], 1)] if leader else None,
                      "next_tls": [(x[0], x[1], round(x[2], 1), x[3]) for x in c.vehicle.getNextTLS(vid)[:1]],
                      "route_next": c.vehicle.getRoute(vid)[c.vehicle.getRouteIndex(vid) + 1:][:2]}
                # persons on this junction's crossings and walking areas
                peds = []
                for pid in c.person.getIDList():
                    pl = c.person.getLaneID(pid)
                    if pl.startswith(f":{j}_c"):
                        peds.append((pid, pl, round(c.person.getSpeed(pid), 2)))
                ev["persons_on_crossings"] = peds
                if tl_link:
                    ev["foes_green"] = [lk.index for lk in js[j].links if lk.index != tl_link[1]
                                        and lk.index < len(st) and st[lk.index] in "Gg"
                                        and js[j].conflict(tl_link[1], lk.index)]
                events.append(ev)
    finally:
        s.close()
    colls = []
    if coll.is_file():
        colls = [dict(e.attrib) for e in ET.parse(coll).getroot().iter("collision")]
    rep = mon.report()
    kinds = Counter()
    for e in events:
        if e["persons_on_crossings"]:
            kinds["ped_on_crossing"] += 1
        elif e["leader"]:
            kinds["leader_far_or_moving"] += 1
        elif e["link_state"] in ("r", "y"):
            kinds[f"link_{e['link_state']}"] += 1
        else:
            kinds["other"] += 1
    res = {"tag": a.tag, "controller": a.controller, "scale": a.scale, "seconds": a.seconds, "kinds": kinds,
           "events": events, "collisions": colls, "monitor": rep}
    (out_dir / f"diag_{a.tag}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"kinds": kinds, "events": len(events), "collisions": len(colls),
                      "monitor_counts": {k: (len(v) if isinstance(v, list) else v)
                                         for k, v in rep.items() if not isinstance(v, dict)}}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
