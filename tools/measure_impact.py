"""How a crash looks as it happens: SUMO's own motion of the cars that crash, against what the viewer draws (3 Nov).

The GO-FLOW team saw crashing cars go from driving to standing over 2-3 s, like hard braking ("soft impact"), and
asked whether the viewer's smoothing causes it. This runs the viewer's traffic and accidents in SUMO alone
(tools/check_accidents.py, no CARLA), keeps each wreck's SUMO states from 8 s before the crash to 5 s after, and
replays them through the viewer's own motion (cosim/interp.Timed, as cosim/pose_proc.Poser.vehicles does, 60 poses a
second) and the wreck's damage ramp (cosim/pose_proc.lay_wreck). For each wreck it reports:

* SUMO: its speed over the last steps before the crash, and how long it took from 90 % of its speed to the crash
  (braking that SUMO itself did, before any viewer smoothing);
* drawn: how long the drawn car takes from 90 % of its speed to under 0.5 m/s, and whether it slides back after;
* the damage ramp (push and turn): when it starts and ends, in seconds from the moment the drawn cars touch.

    python tools/measure_impact.py [--minutes 30] [--share 0.3] [--seed 7] [--ring] [--json out/impact.json]

--share raises the careless share only to get more crashes in less time; it changes how often crashes happen, not
how one happens.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cosim.interp import Timed  # noqa: E402

PRE_S, POST_S = 8.0, 5.0
POSES_PER_S = 60
LEAD_S = 0.2  # cosim/freefly.FrameFeeder.LEAD_S: a step's segment (and the scene update) goes this long before it starts


def drawn_track(states: list[tuple], dt: float, t_crash: float, rest: tuple | None = None,
                t_hit: float | None = None) -> list[tuple]:
    """states: [(t, x, y, speed)] at each SUMO step. -> [(t, x, y)] as the viewer draws it, 60 poses a second.

    The segment for step k (shown from t_k to t_k + dt, one step late) runs from the state at k-1 to the state at
    k, with the states at k-2 and k+1 shaping the curve (cosim/freefly.FrameFeeder.segment). t here is the SUMO time
    the picture shows: the state at step k is reached at t_k.

    rest, t_hit (3 Nov, the viewer since then): the crash step ends where the car came to rest (rest: SUMO x, y;
    cosim/freefly.produce, IncidentManager.hit_poses) and is drawn stopping dead there at t_hit (Timed stop_at,
    cosim/pose_proc.hit_fraction); the step before still runs on towards where SUMO had the car."""
    if t_hit is not None:
        from cosim.pose_proc import hit_fraction
    out = []
    pos = {round(t, 3): (x, y) for t, x, y, _ in states}
    seen = dict(pos)  # what the next step's curve looks ahead to (unchanged by the rest pose)
    times = [round(t, 3) for t, *_ in states]
    tc = round(t_crash, 3)
    if rest is not None and tc in pos:
        pos[tc] = rest
    for j in range(1, len(times)):
        t1, t2 = times[j - 1], times[j]
        p1, p2 = pos[t1], pos[t2]
        p0 = pos[times[j - 2]] if j >= 2 else (2 * p1[0] - p2[0], 2 * p1[1] - p2[1])
        p3 = seen[times[j + 1]] if j + 1 < len(times) else (2 * p2[0] - p1[0], 2 * p2[1] - p1[1])
        if j + 1 < len(times) and times[j + 1] != tc:
            p3 = pos[times[j + 1]]
        hit = None
        if t_hit is not None:
            hit = hit_fraction({"t_hit": t_hit}, t2, dt)
        tm = Timed(p0, p1, p2, p3) if hit is None else Timed(p0, p1, p2, p3, stop_at=hit)
        n = int(round(dt * POSES_PER_S))
        for i in range(n):
            s = i / n
            x, y = tm.at(s)
            out.append((t1 + s * dt, x, y))
    return out


def stop_time(track: list[tuple], v_ref: float, t_crash: float) -> dict:
    """From the drawn track: speed per pose; when it falls under 90 % of v_ref (last time before the crash), when it is
    under 0.5 m/s (first time after that), and how far it moves backwards after the crash."""
    sp = []
    for (ta, xa, ya), (tb, xb, yb) in zip(track, track[1:], strict=False):
        sp.append((tb, math.hypot(xb - xa, yb - ya) / (tb - ta), (xa, ya), (xb, yb)))
    # direction of travel just before the crash
    before = [s for s in sp if t_crash - 1.5 <= s[0] <= t_crash - 0.5]
    hx = hy = 0.0
    for _, _, a, b in before:
        hx += b[0] - a[0]
        hy += b[1] - a[1]
    n = math.hypot(hx, hy) or 1.0
    hx, hy = hx / n, hy / n
    t90 = None
    for t, v, *_ in sp:
        if t <= t_crash + 0.6 and v >= 0.9 * v_ref:
            t90 = t
    t_stop = None
    for t, v, *_ in sp:
        if t90 is not None and t > t90 and v < 0.5:
            t_stop = t
            break
    back = 0.0
    fwd_after = 0.0
    for t, _, a, b in sp:
        if t > t_crash:
            d = (b[0] - a[0]) * hx + (b[1] - a[1]) * hy
            if d < 0:
                back -= d
            else:
                fwd_after += d
    return {"t90": t90, "t_stop": t_stop, "drawn_stop_s": None if t90 is None or t_stop is None else round(t_stop - t90, 2),
            "slides_back_m": round(back, 2), "moves_on_after_m": round(fwd_after, 2),
            "peak_decel_ms2": round(max((max(0.0, (sp[i - 1][1] - sp[i][1]) * POSES_PER_S) for i in range(1, len(sp))
                                         if t_crash - 2 <= sp[i][0] <= t_crash + 1), default=0.0), 1)}


def analyse(case: dict, dt: float) -> dict:
    t_c = case["t_crash"]
    out = {"incident": case["incident"], "kind": case["kind"], "cause": case.get("cause"), "vid": case["vid"], "role": case["role"],
           "impact_kmh": case["impact_kmh"], "wreck_impact_speed_ms": case["speed"]}
    st = case["states"]  # [(t, x, y, speed)]
    pre = [s for s in st if s[0] <= t_c]
    if len(pre) < 4:
        out["skip"] = "too few steps before the crash"
        return out
    v_ref = max(s[3] for s in pre if s[0] >= t_c - 4.0)
    out["sumo_speed_before"] = [round(s[3], 1) for s in pre[-9:]]  # the last 4 s, 0.5 s steps, ending at the crash
    out["sumo_speed_after"] = [round(s[3], 1) for s in st if s[0] > t_c][:4]
    t90 = max((s[0] for s in pre if s[3] >= 0.9 * v_ref), default=None)
    out["v_ref_ms"] = round(v_ref, 1)
    out["sumo_brake_before_crash_s"] = None if t90 is None else round(t_c - t90, 2)
    out["sumo_speed_at_crash_ms"] = round(pre[-1][3], 1)
    new = case.get("t_hit") is not None and case.get("rest") is not None
    tr = drawn_track(st, dt, t_c, case.get("rest") if new else None, case.get("t_hit") if new else None)
    d = stop_time(tr, v_ref, t_c)
    out.update({f"drawn_{k}" if not k.startswith("drawn") else k: v for k, v in d.items() if k not in ("t90", "t_stop")})
    if d["t90"] is not None:
        out["drawn_from_90pct_s_before_contact"] = round(t_c - d["t90"], 2)
    # the stop at the hit as drawn: from the last pose at 90 % of the speed it hit with to the first under 0.5 m/s
    # (a car that hit at under 2 m/s is left out: there is no stop to see)
    v_hit = pre[-1][3]
    if v_hit >= 2.0:
        sp = [(tb, math.hypot(xb - xa, yb - ya) / (tb - ta)) for (ta, xa, ya), (tb, xb, yb) in zip(tr, tr[1:])]
        t_a = max((t for t, v in sp if t <= t_c + 0.6 and v >= 0.9 * v_hit), default=None)
        t_b = next((t for t, v in sp if t_a is not None and t > t_a and v < 0.5), None)
        out["drawn_stop_at_hit_s"] = None if t_a is None or t_b is None else round(t_b - t_a, 3)
    if d["t_stop"] is not None:
        out["drawn_stopped_s_after_contact"] = round(d["t_stop"] - t_c, 2)
    # the damage ramp, in seconds from the moment the drawn car stops at the hit
    ramp = case.get("ramp_s")
    if ramp is not None and not new:
        # before 3 Nov: from when the scene saw the incident, LEAD_S before the crash step's segment starts; that
        # segment ends on the contact (t_c). So the ramp started at t_c - dt - LEAD_S.
        start = -dt - LEAD_S
        out["damage_ramp_from_contact_s"] = [round(start, 2), round(start + ramp, 2)]
    elif ramp is not None:
        from cosim.pose_proc import damage_share

        dmg = {"ramp_s": ramp, "t_hit": case["t_hit"]}
        t_drawn_stop = d["t_stop"] if d["t_stop"] is not None else case["t_hit"]
        first = next((t for t, *_ in tr if damage_share(dmg, None, t) > 0.0), None)
        full = next((t for t, *_ in tr if damage_share(dmg, None, t) >= 1.0), None)
        out["damage_ramp_from_contact_s"] = [None if first is None else round(first - case["t_hit"], 2),
                                             None if full is None else round(full - case["t_hit"], 2)]
        out["t_hit_in_step"] = round(case["t_hit"] - (t_c - dt), 2)
        out["drawn_stop_vs_hit_s"] = round(t_drawn_stop - case["t_hit"], 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=8997)
    ap.add_argument("--ring", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from cosim import incidents as inc_mod
    from cosim import sumo_side
    from tools import check_accidents

    hist: collections.deque = collections.deque(maxlen=int(PRE_S / 0.5) + 2)
    cases: dict[tuple, dict] = {}   # (incident, vid) -> case
    holder: dict = {}
    seen: set = set()
    dt_box = [0.5]

    orig_init = inc_mod.IncidentManager.__init__

    def init(self, *args, **kw):
        orig_init(self, *args, **kw)
        holder["im"] = self

    orig_step = sumo_side.SumoSide.step

    def step(self, *args, **kw):
        im = holder.get("im")
        if im is not None:  # the incidents of the step before (im.step ran after it): new wrecks
            for inc in list(im.incidents) + list(im.done):
                for w in inc.wrecks:
                    key = (inc.id, w.vid)
                    if key in seen:
                        continue
                    seen.add(key)
                    t_c = hist[-1][0] if hist else 0.0
                    states = [(t, *vs[w.vid]) for t, vs in hist if w.vid in vs]
                    cases[key] = {"incident": inc.id, "kind": inc.kind, "cause": getattr(inc, "cause", None), "vid": w.vid, "role": w.role,
                                  "impact_kmh": round(inc.severity * 3.6), "speed": round(w.speed, 2), "t_crash": t_c,
                                  "states": states, "ramp_s": (w.damage or {}).get("ramp_s"), "until": t_c + POST_S,
                                  "t_hit": getattr(w, "t_hit", None), "rest": (w.x, w.y), "rest_angle": w.angle,
                                  "length": w.length, "width": 1.85}
        r = orig_step(self, *args, **kw)
        dt_box[0] = self.step_length
        snap = {vid: (v.x, v.y, v.speed) for vid, v in r.vehicles.items()}
        hist.append((r.time, snap))
        for cs in cases.values():
            if r.time <= cs["until"] and cs["vid"] in snap:
                cs["states"].append((r.time, *snap[cs["vid"]]))
        return r

    inc_mod.IncidentManager.__init__ = init
    sumo_side.SumoSide.step = step
    # check_accidents imported IncidentManager by name: patch its reference too
    check_accidents.IncidentManager.__init__ = init
    argv2 = ["--minutes", str(a.minutes), "--seed", str(a.seed), "--port", str(a.port), "--share", str(a.share),
             "--json", str(ROOT / "out" / "freefly" / f"measure_impact_check_{a.seed}.json")]
    if a.ring:
        argv2.append("--ring")
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        check_accidents.main(argv2)
    dt = dt_box[0]
    rows = [analyse(c, dt) for c in cases.values()]
    # inside each other? The two first wrecks of each incident, their bodies shrunk by 0.15 m all round: where SUMO
    # saw them (what the viewer drew at the hit before 3 Nov) and where they came to rest (drawn since)
    try:
        from cosim.incidents import body_box, boxes_touch
    except ImportError:  # (the code before 3 Nov, measured for comparison: no overlap check)
        body_box = boxes_touch = None
    by_inc: dict = {}
    for cs in cases.values():
        by_inc.setdefault(cs["incident"], []).append(cs)
    overlap = {"at_sumo_step": 0, "at_rest": 0, "pairs": 0}
    for lst in by_inc.values():
        if body_box is None or len(lst) < 2 or lst[0]["kind"] == "breakdown":
            continue
        ca, cb = lst[0], lst[1]
        sa = next((s_ for s_ in ca["states"] if abs(s_[0] - ca["t_crash"]) < 1e-6), None)
        sb = next((s_ for s_ in cb["states"] if abs(s_[0] - cb["t_crash"]) < 1e-6), None)
        if sa is None or sb is None or abs(ca["t_crash"] - cb["t_crash"]) > 1e-6:
            continue
        overlap["pairs"] += 1
        ang = {id(ca): ca["rest_angle"], id(cb): cb["rest_angle"]}

        def box(cs, x, y):
            return body_box(x, y, ang[id(cs)], cs["length"] - 0.3, cs["width"] - 0.3)
        overlap["at_sumo_step"] += boxes_touch(box(ca, sa[1], sa[2]), box(cb, sb[1], sb[2]))
        overlap["at_rest"] += boxes_touch(box(ca, *ca["rest"]), box(cb, *cb["rest"]))
    ok = [r for r in rows if "skip" not in r]

    def med(xs):
        xs = sorted(x for x in xs if x is not None)
        return None if not xs else xs[len(xs) // 2]

    colliders = [r for r in ok if r["role"] == "collider"]
    summary = {
        "minutes": a.minutes, "seed": a.seed, "share": a.share, "ring": a.ring, "step_s": dt, "wrecks": len(ok),
        "incidents": len({r["incident"] for r in ok}),
        "colliders": len(colliders),
        "collider_sumo_brake_before_crash_s_median": med([r["sumo_brake_before_crash_s"] for r in colliders]),
        "collider_sumo_brake_before_crash_s_over_1s": sum(1 for r in colliders
                                                          if (r["sumo_brake_before_crash_s"] or 0) > 1.0),
        "collider_drawn_from_90pct_s_before_contact_median": med([r.get("drawn_from_90pct_s_before_contact")
                                                                  for r in colliders]),
        "collider_drawn_stopped_s_after_contact_median": med([r.get("drawn_stopped_s_after_contact") for r in colliders]),
        "collider_slides_back_m_median": med([r["drawn_slides_back_m"] for r in colliders]),
        "damage_ramp_ends_before_contact": sum(1 for r in ok if r.get("damage_ramp_from_contact_s")
                                               and r["damage_ramp_from_contact_s"][1] < 0),
        "inside_each_other": overlap,
        "rows": rows,
    }
    txt = json.dumps(summary, indent=1)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
