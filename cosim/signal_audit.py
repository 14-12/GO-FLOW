"""What each Town05 signal controls and whether its programs are safe (no server, no simulation).

Read from the SUMO network itself (the same file the simulation and the CARLA bridge use):

* movements: every signal link (tls id, link index) with its approach (N/E/S/W, from the heading of the lane
  that enters the junction), lane, direction (s/r/l/t), the lane it leaves by, and the CARLA landmark (pole) that
  shows it (``linkSignalID``, cosim/tlmap.py); crossings with their length and the roads they cross;
  links on lanes no vehicle may use (shoulders, ``disallow="all"``) or only pedestrians (sidewalk lanes) are
  marked, since they never carry a vehicle;
* conflicts: SUMO's own junction logic (``<request foes=... response=...>``): link i and j conflict when j is in
  i's foes; i gives way to j when j is in i's response. This is what the simulation enforces, so the audit checks
  the programs against the right of way the vehicles actually obey;
* checks, phase by phase (``check_program``):
    - every phase gives every link a state, from r y g G only (no stale, missing or unknown states);
    - no two conflicting links green at once unless one of them gives way to the other ("primary" conflicts,
      e.g. two crossing through streams, never share a green; a permissive left "g" may share the green of the
      oncoming through "G" because it yields to it; a turning car may share the green of a crossing it turns over
      because it yields to the people on it);
    - a priority green "G" never gives way to another green (a "G" that must yield is mislabelled);
    - after a vehicle link's green ends it shows yellow, then red, before any link conflicting with it turns
      green (intergreen >= its yellow + the all red);
    - after a crossing's walk ends, conflicting vehicles stay red for at least the pedestrian clearance time,
      crossing length / PED_CLEAR_SPEED (MUTCD 4I.06: 3.5 ft/s), counting the flashing don't-walk and the
      steady don't-walk buffer (the yellow and all red that follow it), and the buffer is at least BUFFER_MIN_S;
    - each CARLA pole shows links that agree in every phase (one actor has one state for all its heads).

``audit(net_file, programs)`` returns a dict; tools/audit_town05_signals.py writes it as Markdown and JSON.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

PED_CLEAR_SPEED = 1.07   # m/s = 3.5 ft/s (MUTCD 11th ed., 4I.06)
BUFFER_MIN_S = 2.0       # steady don't walk before conflicting traffic is released (MUTCD 11th ed., 4I.06 para 4:
#                          "at least 2 seconds"; the change interval plus the buffer >= the pedestrian clearance time)
MIN_YELLOW_S = 3.0       # MUTCD 4F.17 (yellow change interval 3-6 s)
MIN_ALLRED_S = 1.0
GREEN = "Gg"
KNOWN = set("rygGu")  # "u": a crossing in its walk clearance (red to walkers; the lamps flash, 27 Oct)


@dataclass
class Link:
    index: int
    kind: str                  # vehicle | crossing | unused | sidewalk
    approach: str = ""         # N E S W (vehicles: the side they come from)
    from_lane: str = ""
    to_lane: str = ""
    dir: str = ""
    via: str = ""
    landmark: str | None = None
    crossing: str = ""         # crossing edge id (kind crossing)
    length: float = 0.0        # crossing length (kind crossing), internal lane length (vehicles)
    crosses: tuple[str, ...] = ()  # roads a crossing crosses
    speed: float = 13.89       # approach lane speed (vehicles)
    lane_type: str = ""        # the approach lane's OpenDRIVE type (driving, parking, ...)


@dataclass
class Junction:
    tls: str
    links: list[Link]
    foes: list[set[int]]
    response: list[set[int]]
    programs: dict[str, list[tuple[str, float, float, str]]] = field(default_factory=dict)  # id -> (state, dur,
    #   min dur, name)

    def conflict(self, i: int, j: int) -> bool:
        return j in self.foes[i] or i in self.foes[j]

    def yields(self, i: int, j: int) -> bool:
        """i gives way to j."""
        return j in self.response[i]


def _bits(s: str) -> set[int]:
    n = len(s)
    return {n - 1 - k for k, ch in enumerate(s) if ch == "1"}


def _heading(shape: str) -> float:
    pts = [tuple(map(float, p.split(",")[:2])) for p in shape.split()]
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0  # 0 north, clockwise


def side(heading: float) -> str:
    """The side of the junction a vehicle travelling at `heading` (0 north, clockwise) comes from."""
    return "SWNE"[int(((heading + 45.0) % 360.0) // 90.0)]


def load(net_file: Path, add_files: list[Path] = ()) -> dict[str, Junction]:
    root = ET.parse(net_file).getroot()
    lanes = {}
    crossings = {}
    for e in root.iter("edge"):
        if e.get("function") == "crossing":
            ln = e.find("lane")
            crossings[e.get("id")] = (float(ln.get("length")), tuple(e.get("crossingEdges", "").split()))
        for ln in e.findall("lane"):
            lanes[ln.get("id")] = ln
    tls_ids = [t.get("id") for t in root.iter("tlLogic")]
    out: dict[str, Junction] = {}
    conns: dict[str, list[ET.Element]] = {}
    for c in root.iter("connection"):
        if c.get("tl"):
            conns.setdefault(c.get("tl"), []).append(c)
    reqs: dict[str, list[ET.Element]] = {}
    for j in root.iter("junction"):
        if j.get("type") in ("traffic_light", "traffic_light_right_on_red", "traffic_light_unregulated"):
            reqs[j.get("id")] = j.findall("request")
    landmarks: dict[tuple[str, int], str] = {}
    for tl in root.iter("tlLogic"):
        for p in tl.iter("param"):
            if p.get("key", "").startswith("linkSignalID:"):
                landmarks[(tl.get("id"), int(p.get("key").split(":")[1]))] = p.get("value")
    for tid in dict.fromkeys(tls_ids):
        cs = sorted(conns.get(tid, []), key=lambda c: int(c.get("linkIndex")))
        n = 1 + max(int(c.get("linkIndex")) for c in cs)
        links: list[Link | None] = [None] * n
        for c in cs:
            i = int(c.get("linkIndex"))
            if links[i] is not None:
                continue
            frm = f"{c.get('from')}_{c.get('fromLane')}"
            to = f"{c.get('to')}_{c.get('toLane')}"
            if c.get("to", "").startswith(":") and "_c" in c.get("to"):
                length, crosses = crossings.get(c.get("to"), (0.0, ()))
                links[i] = Link(i, "crossing", from_lane=frm, to_lane=to, crossing=c.get("to"), length=length,
                                crosses=crosses, landmark=landmarks.get((tid, i)))
                continue
            ln = lanes.get(frm)
            allow, dis = (ln.get("allow") or "") if ln is not None else "", (ln.get("disallow") or "") if ln is not None else ""
            kind = "vehicle"
            if dis.strip() == "all":
                kind = "unused"
            elif allow.strip() == "pedestrian":
                kind = "sidewalk"
            via = c.get("via", "")
            vlen = float(lanes[via].get("length")) if via in lanes else 0.0
            links[i] = Link(i, kind, approach=side(_heading(ln.get("shape"))) if ln is not None else "?",
                            from_lane=frm, to_lane=to, dir=c.get("dir", ""), via=via,
                            landmark=landmarks.get((tid, i)), length=vlen,
                            speed=float(ln.get("speed")) if ln is not None else 13.89,
                            lane_type=(ln.get("type") or "") if ln is not None else "")
        req = reqs.get(tid, [])
        foes = [set() for _ in range(n)]
        resp = [set() for _ in range(n)]
        for r in req:
            k = int(r.get("index"))
            if k < n:
                foes[k] = _bits(r.get("foes"))
                resp[k] = _bits(r.get("response"))
        jn = Junction(tid, [lk or Link(i, "unused") for i, lk in enumerate(links)], foes, resp)
        out[tid] = jn
    for tl in list(root.iter("tlLogic")) + [t for f in add_files for t in ET.parse(f).getroot().iter("tlLogic")]:
        jn = out.get(tl.get("id"))
        if jn is None:
            continue
        ph = []
        for p in tl.findall("phase"):
            d = float(p.get("duration"))
            ph.append((p.get("state"), d, float(p.get("minDur", d)), p.get("name", "")))
        jn.programs[tl.get("programID")] = ph
    return out


def _active(jn: Junction, i: int) -> bool:
    return jn.links[i].kind in ("vehicle", "crossing")


def check_program(jn: Junction, phases: list[tuple[str, float, float, str]], use_min: bool = True) -> list[str]:
    """Problems of one program (empty: none). Durations: each phase's minimum (actuated) when use_min, the
    shortest it can run, so the intergreens are checked at their shortest."""
    errs = []
    n = len(jn.links)
    for k, (s, *_rest) in enumerate(phases):
        if len(s) != n:
            errs.append(f"phase {k}: {len(s)} states for {n} links")
            continue
        bad = set(s) - KNOWN
        if bad:
            errs.append(f"phase {k}: unknown states {sorted(bad)}")
        bad_u = [i for i, ch in enumerate(s) if ch == "u" and jn.links[i].kind != "crossing"]
        if bad_u:
            errs.append(f"phase {k}: 'u' (walk clearance) on links {bad_u} that are not crossings")
        green = [i for i, ch in enumerate(s) if ch in GREEN and _active(jn, i)]
        for a in green:
            for b in green:
                if a >= b or not jn.conflict(a, b):
                    continue
                if not (jn.yields(a, b) or jn.yields(b, a)):
                    errs.append(f"phase {k}: links {a} and {b} conflict and are green together, neither gives way")
        for a in green:
            if s[a] == "G" and jn.links[a].kind == "vehicle":
                for b in green:
                    if b != a and jn.yields(a, b):
                        errs.append(f"phase {k}: link {a} is 'G' (priority) but gives way to green link {b}")
                        break
    if errs:
        return errs
    # intergreens, over two cycles so the wrap-around is seen
    dur = [p[2] if use_min else p[1] for p in phases]
    seq = [(p[0], d) for p, d in zip(phases, dur, strict=True)] * 2
    t = 0.0
    starts = []
    for _, d in seq:
        starts.append(t)
        t += d
    for i in range(n):
        if not _active(jn, i):
            continue
        for k in range(len(phases)):
            s, s_next = seq[k][0], seq[k + 1][0]
            if not (s[i] in GREEN and s_next[i] not in GREEN):
                continue
            end = starts[k + 1]
            # what i shows until a conflicting green
            yellow = 0.0
            m = k + 1
            released = None
            while m < len(seq):
                st = seq[m][0]
                if st[i] in GREEN:
                    break  # i green again before any foe
                if st[i] == "y":
                    yellow += seq[m][1]
                # a conflicting link whose green starts here (one that was green with i already gives way to it,
                # checked phase by phase above: a permissive turn goes on yielding to those still on the crossing)
                before = seq[m - 1][0]
                foes_green = [j for j in range(n) if j != i and _active(jn, j) and st[j] in GREEN
                              and before[j] not in GREEN and jn.conflict(i, j)]
                if foes_green:
                    released = (m, foes_green)
                    break
                m += 1
            if released is None:
                continue
            m, fg = released
            inter = starts[m] - end
            if jn.links[i].kind == "vehicle":
                if yellow < MIN_YELLOW_S - 1e-6:
                    errs.append(f"link {i}: yellow {yellow:g} s before conflicting link(s) {fg} (phase {m % len(phases)})")
                if inter - yellow < MIN_ALLRED_S - 1e-6:
                    errs.append(f"link {i}: all red {inter - yellow:g} s before conflicting link(s) {fg} "
                                f"(phase {m % len(phases)})")
            else:
                need = jn.links[i].length / PED_CLEAR_SPEED
                # the buffer: from the end of the vehicles' green that ran with the flashing don't walk
                # (the phases between the end of the walk and the release in which no vehicle has a green: the
                # yellow and the all red after the flashing don't walk, shown as steady don't walk)
                # (since 27 Oct the parallel cars may keep their green through it: a "walk buffer" phase)
                names = [p[3] for p in phases] * 2
                buffer = sum(seq[q][1] for q in range(k + 1, m)
                             if names[q] == "walk buffer"
                             or not any(seq[q][0][x] in GREEN for x in range(n) if jn.links[x].kind == "vehicle"))
                if inter + 1e-6 < need:
                    errs.append(f"crossing link {i} ({jn.links[i].length:.1f} m): {inter:g} s from end of walk to "
                                f"conflicting green {fg}, needs {need:.1f} s")
                if buffer + 1e-6 < BUFFER_MIN_S:
                    errs.append(f"crossing link {i}: buffer {buffer:g} s before conflicting green {fg}")
    return errs


def check_protected(jn: Junction, phases) -> list[str]:
    """The 27 Oct rule (cosim/signal_plan.py): no green may have to give way. No "g" anywhere, and no two links in
    conflict (vehicles or crossings, SUMO's foes) green at the same time, whoever would give way."""
    errs = []
    act = [i for i in range(len(jn.links)) if _active(jn, i)]
    for k, (s, *_r) in enumerate(phases):
        if "g" in s:
            errs.append(f"phase {k}: a green that must give way ('g') at links "
                        f"{[i for i, ch in enumerate(s) if ch == 'g']}")
        green = [i for i in act if s[i] in GREEN]
        for a in green:
            for b in green:
                if a < b and jn.conflict(a, b):
                    errs.append(f"phase {k}: links {a} and {b} cross each other and are green together")
    return errs


def served(jn: Junction, phases) -> list[str]:
    """Every vehicle movement and every crossing gets a green in the cycle."""
    s_all = [p[0] for p in phases]
    return [f"link {lk.index} ({lk.kind} {lk.approach}{lk.dir}{lk.crossing}) never green" for lk in jn.links
            if _active(jn, lk.index) and not any(s[lk.index] in GREEN for s in s_all)]


def poles(jn: Junction) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for lk in jn.links:
        if lk.landmark:
            out.setdefault(lk.landmark, []).append(lk.index)
    return out


def pole_disagreements(jn: Junction, phases) -> list[str]:
    """Phases in which one CARLA pole would have to show two colours (cosim/convert.tl_state shows the most
    restrictive one then). Only links vehicles use count."""
    col = {"G": "green", "g": "green", "y": "yellow", "r": "red"}
    out = []
    for lm, idx in poles(jn).items():
        idx = [i for i in idx if jn.links[i].kind == "vehicle"]
        for k, (s, *_r) in enumerate(phases):
            cs = {col.get(s[i], s[i]) for i in idx}
            if len(cs) > 1:
                out.append(f"pole {lm}, phase {k}: {sorted(cs)} on links {idx}")
    return out


def audit(net_file: Path, add_files: list[Path] = ()) -> dict:
    js = load(net_file, add_files)
    rep = {}
    for tid, jn in js.items():
        approaches: dict[str, dict] = {}
        for lk in jn.links:
            if lk.kind != "vehicle":
                continue
            ap = approaches.setdefault(lk.approach, {"lanes": {}, "landmarks": set()})
            ap["lanes"].setdefault(lk.from_lane, set()).add(lk.dir)
            if lk.landmark:
                ap["landmarks"].add(lk.landmark)
        progs = {}
        for pid, ph in jn.programs.items():
            progs[pid] = {"phases": [{"state": s, "duration": d, "min": mn, "name": nm} for s, d, mn, nm in ph],
                          "errors": check_program(jn, ph, use_min=True) + served(jn, ph),
                          "protected_errors": check_protected(jn, ph),
                          "pole_disagreements": pole_disagreements(jn, ph)}
        cross = []
        for lk in jn.links:
            if lk.kind != "crossing":
                continue
            conf = [j for j in range(len(jn.links)) if jn.links[j].kind == "vehicle" and jn.conflict(lk.index, j)]
            cross.append({"link": lk.index, "crossing": lk.crossing, "length_m": round(lk.length, 2),
                          "crosses": list(lk.crosses), "conflicting_vehicle_links": conf,
                          "clearance_needed_s": round(lk.length / PED_CLEAR_SPEED, 1)})
        rep[tid] = {
            "links": [lk.__dict__ | {"crosses": list(lk.crosses)} for lk in jn.links],
            "approaches": {a: {"lanes": {ln: sorted(d) for ln, d in v["lanes"].items()},
                               "landmarks": sorted(v["landmarks"]),
                               "shared_left_lane": any("l" in d and len(d) > 1 for d in v["lanes"].values()),
                               "exclusive_left_lane": any(d == {"l"} for d in v["lanes"].values())}
                           for a, v in sorted(approaches.items())},
            "conflicts": {str(i): sorted(j for j in jn.foes[i] | {x for x in range(len(jn.links)) if i in jn.foes[x]}
                                         if _active(jn, j))
                          for i in range(len(jn.links)) if _active(jn, i)},
            "gives_way_to": {str(i): sorted(j for j in jn.response[i] if _active(jn, j))
                             for i in range(len(jn.links)) if _active(jn, i) and jn.response[i]},
            "crossings": cross,
            "programs": progs,
        }
    return rep
