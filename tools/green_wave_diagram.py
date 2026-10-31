"""Time-space diagram of a Town05 corridor: are the signals in step? (27 Oct)

    python tools/green_wave_diagram.py                       # out/signal_checks/green_wave_<corridor>.png
    python tools/green_wave_diagram.py --corridor 53,1126,1292 --controllers coordinated,agents

For each controller, a SUMO run (as tools/check_junction_stops.py runs the town, x0.8, after a warm-up) records every
second the colour of each signal's straight-on movement along the corridor, both ways, and where every vehicle
driving along it is. Drawn the way traffic engineers check a green wave: distance along the corridor across, time up;
at each signal a bar green / amber / red for the way the panel shows; each vehicle's path a line. With the signals in
step a car leaving on green runs through the next signals' greens (a band of lines with no flat parts); out of step
the lines stop (flat) at each red.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOWN05 = ROOT / "scenarios" / "town05"
OUT = ROOT / "out" / "signal_checks"
CORRIDORS = {"row200": "838,979,1427,1574", "row290": "1882,53,1126,1292", "col145": "1882,53,979,1722,2240"}


def corridor_geometry(net, nodes: list[str]):
    """For each direction (forward, back): [(junction, position m)], the edges between them with their start
    position, and per junction the straight-on link indices that carry the corridor."""
    out = {}
    for name, seq in (("forward", nodes), ("back", nodes[::-1])):
        pos = {seq[0]: 0.0}
        edges = {}
        x = 0.0
        for a, b in zip(seq, seq[1:], strict=False):
            # a path a -> b over plain edges
            start = net.getNode(a)
            best = None
            for e in start.getOutgoing():
                path, length, cur = [e], e.getLength(), e
                for _ in range(4):
                    if cur.getToNode().getID() == b:
                        break
                    nxt = [o for o in cur.getToNode().getOutgoing() if o.getToNode().getID() != cur.getFromNode().getID()]
                    if len(nxt) != 1:
                        break
                    cur = nxt[0]
                    path.append(cur)
                    length += cur.getLength()
                if cur.getToNode().getID() == b:
                    best = (path, length)
                    break
            if best is None:
                raise SystemExit(f"no road from {a} to {b}")
            off = x
            for e in best[0]:
                edges[e.getID()] = off
                off += e.getLength()
            x += best[1]
            pos[b] = x
        links = {}
        ids = list(edges)
        for j in seq:
            inc = [e for e in ids if net.getEdge(e).getToNode().getID() == j]
            outg = [e for e in ids if net.getEdge(e).getFromNode().getID() == j]
            idx = []
            tl = net.getTLS(j) if j in {t.getID() for t in net.getTrafficLights()} else None
            if tl is not None:
                for ln_in, ln_out, li in tl.getConnections():
                    con = [c_ for c_ in ln_in.getOutgoing() if c_.getToLane() == ln_out]
                    straight = bool(con) and con[0].getDirection() == "s"
                    in_ok = ln_in.getEdge().getID() in inc if inc else straight
                    out_ok = ln_out.getEdge().getID() in outg if outg else straight
                    if in_ok and out_ok:
                        idx.append(li)
            links[j] = sorted(set(idx))
        out[name] = {"seq": seq, "pos": pos, "edges": edges, "links": links, "length": x}
    return out


def record(controller: str, nodes: list[str], scale: float, seed: int, warmup: float, seconds: float, port: int):
    import sumolib
    import traci.constants as tc

    from cosim.agents_actuated import make as make_agents
    from cosim.sumo_side import SumoSide

    net = sumolib.net.readNet(str(TOWN05 / "town05.net.xml"))
    geo = corridor_geometry(net, nodes)
    policy = make_agents(controller) if controller.startswith("agents") else None
    s = SumoSide(TOWN05, controller if policy is None else "agents", seed=seed, port=port, label=f"gw{port}",
                 policy=policy, extras=True, emergency_every=120.0, vehicle_vars=[tc.VAR_LANE_ID, tc.VAR_LANEPOSITION],
                 extra_args=["--scale", f"{scale:g}", "--end", f"{warmup + seconds + 5:g}"])
    s.extras.config.pedestrian_trips = True
    if policy is not None:
        ex = s.extras
        policy.emergency = lambda: [v for v, r in ex.emergency.items() if r.on_run]
    states = []   # (t, {tls: state})
    paths = {"forward": {}, "back": {}}
    try:
        while s.time < warmup + seconds:
            r = s.step()
            if r.time < warmup or int(r.time * 2) % 2:
                continue
            t = r.time - warmup
            states.append((t, {j: r.tls.get(j, "") for j in nodes}))
            for vid, sub in s.vehicle_subs.items():
                lane = sub.get(tc.VAR_LANE_ID, "")
                edge = lane.rsplit("_", 1)[0]
                for d in ("forward", "back"):
                    g = geo[d]
                    if edge in g["edges"]:
                        paths[d].setdefault(vid, []).append((t, g["edges"][edge] + sub.get(tc.VAR_LANEPOSITION, 0.0)))
    finally:
        s.close()
    return geo, states, paths


def draw(results: dict, nodes: list[str], out: Path, seconds: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(len(names), 2, figsize=(14, 6 * len(names)), squeeze=False)
    col = {"G": "#16a34a", "g": "#16a34a", "y": "#f59e0b", "u": "#dc2626", "r": "#dc2626"}
    for row, ctl in enumerate(names):
        geo, states, paths = results[ctl]
        for c_, d in enumerate(("forward", "back")):
            ax = axes[row][c_]
            g = geo[d]
            for j in g["seq"]:
                x = g["pos"][j]
                idx = g["links"].get(j, [])
                if not idx:
                    continue
                prev_t, prev_c = None, None
                for t, st in states:
                    s_ = st.get(j, "")
                    chars = [s_[i] for i in idx if i < len(s_)]
                    cc = "G" if any(ch in "Gg" for ch in chars) else ("y" if "y" in chars else "r")
                    if cc != prev_c:
                        if prev_c is not None:
                            ax.plot([x, x], [prev_t, t], color=col[prev_c], lw=7, solid_capstyle="butt")
                        prev_t, prev_c = t, cc
                if prev_c is not None:
                    ax.plot([x, x], [prev_t, states[-1][0]], color=col[prev_c], lw=7, solid_capstyle="butt")
                ax.text(x, seconds * 1.005, j, ha="center", va="bottom", fontsize=9, color="#334155")
            for pts in paths[d].values():
                if len(pts) < 3 or max(x for _, x in pts) - min(x for _, x in pts) < 30.0:
                    continue  # not along the corridor (turned in or out at once)
                ts, xs = zip(*pts, strict=True)
                ax.plot(xs, ts, color="#1e3a8a", lw=0.6, alpha=0.55)
            ax.set_xlim(-5, g["length"] + 5)
            ax.set_ylim(0, seconds)
            ax.set_xlabel(f"distance along the corridor (m), {'→'.join(g['seq'])}")
            ax.set_ylabel("time (s)")
            ax.set_title(f"{ctl}: {' → '.join(g['seq'])}", fontsize=11, pad=16)
    fig.suptitle("Town05: signals along the corridor (bars: the straight-on signal each way; lines: vehicles)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corridor", default="row200", help=f"a name ({', '.join(CORRIDORS)}) or junction ids")
    ap.add_argument("--controllers", default="coordinated,agents")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--warmup", type=float, default=600.0)
    ap.add_argument("--seconds", type=float, default=360.0)
    ap.add_argument("--port", type=int, default=9700)
    a = ap.parse_args(argv)
    nodes = CORRIDORS.get(a.corridor, a.corridor).split(",")
    results = {}
    for k, ctl in enumerate(a.controllers.split(",")):
        results[ctl] = record(ctl, nodes, a.scale, a.seed, a.warmup, a.seconds, a.port + k)
    OUT.mkdir(parents=True, exist_ok=True)
    name = a.corridor if a.corridor in CORRIDORS else "_".join(nodes)
    out = OUT / f"green_wave_{name}.png"
    draw(results, nodes, out, a.seconds)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
