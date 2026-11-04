"""Exécute un épisode d'évaluation pour une configuration et calcule toutes les métriques.

Configurations :
  B1_fixe          feux à cycle fixe (programme SUMO, 90 s)
  B2_actuated      feux actionnés par détecteurs (SUMO type="actuated")
  B2P_preemption   cycle fixe + préemption classique locale (urgence à < 300 m -> vert immédiat)
  B2AP_actuated_preemption  feux actionnés + la même préemption (référence industrielle la plus forte)
  B3_ppo           PPO seul, sans information de priorité (contexte toujours NORMAL)
  B3L_ppo_local    PPO + priorité détectée localement (V2I 300 m, pas d'anticipation)
  B4_ppo_regles    PPO + coordinateur stratégique déterministe (corridor, arbitrage, récupération)
  B5_ppo_llm       PPO + agent LLM (LangChain/Ollama) + Safety Controller (architecture complète)

Préemption garantie (FORCE_PREEMPTION) : dans B3L, B4 et B5, quand le coordinateur met un carrefour en
EMERGENCY_PREEMPTION / CONFLICT_RESOLUTION, le vert de l'approche de l'urgence est imposé (comme dans B2AP),
sous les contraintes du Safety Controller ; le PPO gère le reste du temps. B3 (PPO seul) n'est pas concerné.
"""
from __future__ import annotations

import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from coordination.rules import LocalCoordinator, RuleCoordinator
from environment.runtime import DECISION_S, Context, Runtime
from simulation.scenarios import write_scenario

CONFIGS = ["B1_fixe", "B2_actuated", "B2P_preemption", "B2AP_actuated_preemption", "B3_ppo", "B3L_ppo_local", "B4_ppo_regles", "B5_ppo_llm"]
PPO_CONFIGS = {"B3_ppo", "B3L_ppo_local", "B4_ppo_regles", "B5_ppo_llm"}
EMERGENCY_TYPES = {"ambulance", "fire_truck", "police"}
FORCE_PREEMPTION = True  # B3L/B4/B5 : vert imposé pendant une urgence (sinon le PPO seul décide)
B4_HORIZON_S = 30.0  # horizon d'anticipation du corridor, réglé sur les graines de validation 11-15


def emergency_ids(spec: dict) -> set[str]:
    """Vraies urgences sirène active sur tout le trajet (exclut fausses demandes et sirènes coupées)."""
    off = {e["vehicle"] for e in spec.get("events", []) if e["type"] == "siren_off"}
    return {p["id"] for p in spec["priority"]
            if p["type"] in EMERGENCY_TYPES and p["id"] not in off
            and p.get("params", {}).get("confirmed", "true") != "false"}


def run_episode(config: str, spec: dict, model=None, llm=None, llm_name: str = "?", gui: bool = False,
                record: bool = False, out_dir: Path | None = None, backend: str = "libsumo",
                verbose: bool = False, gui_delay_ms: int = 100, horizon_s: float = B4_HORIZON_S,
                llm_timeout_s: float = 90.0, force_preemption: bool = FORCE_PREEMPTION) -> dict:
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="adsp_eval_"))
    cfg = write_scenario(spec, out_dir, tripinfo=True, actuated=config in ("B2_actuated", "B2AP_actuated_preemption"))
    rt = Runtime(cfg, seed=spec["seed"], backend=backend, gui=gui, events=[dict(e) for e in spec["events"]],
                 keep_records=record, label=f"{config}_{spec['name']}",
                 extra_args=["--start", "--delay", str(gui_delay_ms), "--quit-on-end"] if gui else None).start()
    c = rt.c
    tls_ids = list(rt.inters)

    coord = None
    if config in ("B2P_preemption", "B2AP_actuated_preemption", "B3L_ppo_local"):
        coord = LocalCoordinator(recovery_s=10.0, mode_safety=rt.mode_safety, inters=rt.inters)
    elif config == "B4_ppo_regles":
        coord = RuleCoordinator(horizon_s=horizon_s, recovery_s=10.0, mode_safety=rt.mode_safety, inters=rt.inters)
    elif config == "B5_ppo_llm":
        from coordination.llm_agent import LLMCoordinator
        coord = LLMCoordinator(llm, rt, model_name=llm_name, horizon_s=horizon_s, timeout_s=llm_timeout_s,
                               dispatch=spec.get("dispatch"))

    ctx = {tls: Context() for tls in tls_ids}
    mode_timeline = {tls: [] for tls in tls_ids}
    last_mode = {tls: None for tls in tls_ids}
    degraded = {tls: False for tls in tls_ids}
    queue_sum, queue_max, n_steps = 0.0, 0, 0
    green_start = {tls: (c.trafficlight.getPhase(tls), 0.0) for tls in tls_ids}
    min_green_viol = 0
    was_external = {tls: False for tls in tls_ids}  # le changement de phase vient-il de notre contrôleur ?
    lanes = {tls: [ln for lns in inter.incoming_lanes.values() for ln in lns] for tls, inter in rt.inters.items()}
    frames = []
    decide_every = DECISION_S
    wall0 = time.perf_counter()
    end_t = spec["duration"] + spec.get("drain", 600)
    pred_time = 0.0
    gui_state = {"active": [], "tracked": None, "slow": False}
    if gui:
        try:
            c.gui.setDelay(GUI_VIEW, 0)  # avance rapide tant qu'il n'y a que des voitures
        except Exception:
            pass

    while c.simulation.getMinExpectedNumber() > 0 and rt.t < end_t:
        rt.step()
        t = rt.t
        if gui:  # sumo-gui : cercle de couleur, caméra qui suit l'urgence, ralenti seulement quand il se passe quelque chose
            _gui_follow(c, t, gui_state, gui_delay_ms)
        # ---- coordination stratégique (contexte par intersection)
        if coord is not None:
            ctx = dict(coord.update(t, rt.tracker.vehicles))
        # ---- watchdog : capteurs perdus -> DEGRADED (programme fixe, sans PPO)
        for tls in tls_ids:
            d = rt.dropout.get(tls)
            is_deg = bool(d and d[0] <= t < d[1]) and config in PPO_CONFIGS
            if is_deg and not degraded[tls]:
                rt.safety_log.fallbacks += 1
                rt.safety_log.add("fallback", t, tls, "observation périmée -> DEGRADED (programme fixe)")
            degraded[tls] = is_deg
        # ---- contrôle des feux
        if config in ("B2P_preemption", "B2AP_actuated_preemption"):
            for tls in tls_ids:
                ctl, cx = rt.ctls[tls], ctx[tls]
                rt.set_priority(tls, cx)
                if cx.local_mode == "EMERGENCY":
                    ctl.take_control(t)
                    greens = rt.inters[tls].green_phases
                    target = rt.inters[tls].approach_phase[cx.approach]
                    mask = ctl.mask(t)
                    if not mask[greens.index(target)]:  # vert max atteint : bascule imposée par la sécurité
                        target = greens[int(np.flatnonzero(mask)[0])]
                    ctl.propose(target, t, "preemption")
                elif ctl.external:
                    ctl.release(t)
        elif config in PPO_CONFIGS:
            decide = int(t) % decide_every == 0
            for tls in tls_ids:
                ctl = rt.ctls[tls]
                cx = ctx[tls] if config != "B3_ppo" else Context()
                # préemption garantie : pendant une urgence, le vert de son approche est imposé (sous les
                # contraintes du Safety Controller : vert min, jaune, vert max), le PPO reprend ensuite
                if force_preemption and cx.local_mode == "EMERGENCY" and cx.approach and not degraded[tls]:
                    ctl.take_control(t)
                    rt.set_priority(tls, cx)
                    greens = rt.inters[tls].green_phases
                    target = rt.inters[tls].approach_phase[cx.approach]
                    mask = ctl.mask(t)
                    if not mask[greens.index(target)]:  # vert max atteint : bascule imposée par la sécurité
                        target = greens[int(np.flatnonzero(mask)[0])]
                    ctl.propose(target, t, "preemption")
                    continue
                if not decide:
                    continue
                obs, age = rt.observation(tls, cx)
                if degraded[tls] or age > rt.safety_cfg.max_obs_age_s:
                    ctl.release(t)
                    continue
                ctl.take_control(t)
                rt.set_priority(tls, cx)
                mask = ctl.mask(t)
                p0 = time.perf_counter()
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                pred_time += time.perf_counter() - p0
                ctl.propose(rt.inters[tls].green_phases[int(action)], t, "ppo", obs_age=age)
        # ---- mesures
        for tls in tls_ids:
            m = "DEGRADED" if degraded[tls] else ctx[tls].mode
            if m != last_mode[tls]:
                mode_timeline[tls].append({"t": t, "mode": m, "vehicle": ctx[tls].vehicle_id})
                if verbose and last_mode[tls] is not None:
                    print(f"  t={t:5.0f}s  {tls}: {last_mode[tls]:>20} -> {m:<20} {ctx[tls].vehicle_id or ''}", flush=True)
                last_mode[tls] = m
            ph = c.trafficlight.getPhase(tls)
            prev_ph, since = green_start[tls]
            if ph != prev_ph:
                if prev_ph in rt.inters[tls].green_phases and was_external[tls] and \
                        t - since < rt.safety_cfg.min_green_s - 0.5 and since > 0:
                    min_green_viol += 1
                green_start[tls] = (ph, t)
            was_external[tls] = rt.ctls[tls].external
        h = [c.lane.getLastStepHaltingNumber(ln) for tls in tls_ids for ln in lanes[tls]]
        queue_sum += sum(h)
        queue_max = max(queue_max, max(h))
        n_steps += 1
        if record:
            frames.append(_frame(c, rt, ctx, t, degraded))
    wall = time.perf_counter() - wall0
    rt.close()

    trips = _read_trips(out_dir / f"{spec['name']}.tripinfo.xml")
    res = _metrics(config, spec, trips, rt, queue_sum / max(n_steps, 1), queue_max, min_green_viol)
    res["wall_time_s"] = round(wall, 2)
    res["ppo_inference_ms"] = round(1000 * pred_time / max(1, n_steps / DECISION_S * len(tls_ids)), 3) if model else None
    extra = {"mode_timeline": mode_timeline, "safety_reasons": rt.safety_log.reasons[:200],
             "events": rt.event_log}
    if coord is not None and hasattr(coord, "decisions"):
        extra["decisions"] = coord.decisions[:300]
    if config == "B5_ppo_llm":
        res.update(coord.summary())
        extra["llm_transcripts"] = coord.stats.transcripts
    if record:
        extra["frames"] = frames
        extra["control_records"] = {tls: rt.ctls[tls].stats.records for tls in tls_ids}
        extra["trips"] = trips
    return {"metrics": res, "extra": extra}


HIGHLIGHT = {"ambulance": ((255, 255, 255, 255), "ambulance (blanc)"), "fire_truck": ((255, 40, 40, 255), "pompiers (rouge)"),
             "police": ((40, 80, 255, 255), "police (bleu foncé)"), "bus_transit": ((0, 170, 255, 255), "bus (bleu clair)")}


GUI_VIEW = "View #0"


def _gui_follow(c, t, st, slow_ms):
    """Ralenti + suivi caméra pendant le passage des véhicules prioritaires, avance rapide le reste du temps."""
    st["active"] += _highlight_new(c, t)
    alive = set(c.vehicle.getIDList())
    st["active"] = [v for v in st["active"] if v in alive]
    try:
        if st["active"]:
            if not st["slow"]:
                c.gui.setDelay(GUI_VIEW, slow_ms)
                c.gui.setZoom(GUI_VIEW, 700)
                st["slow"] = True
            if st["tracked"] not in st["active"]:
                st["tracked"] = st["active"][0]
                c.gui.trackVehicle(GUI_VIEW, st["tracked"])
                print(f"  t={t:5.0f}s  [caméra] suit {st['tracked']}", flush=True)
        elif st["slow"]:
            c.gui.trackVehicle(GUI_VIEW, "")
            c.gui.setDelay(GUI_VIEW, 0)
            st["slow"], st["tracked"] = False, None
            print(f"  t={t:5.0f}s  [caméra] plus d'urgence en circulation : avance rapide", flush=True)
    except Exception:
        pass


def _highlight_new(c, t):
    new = []
    for vid in c.simulation.getDepartedIDList():
        vtype = c.vehicle.getTypeID(vid)
        is_fake = vtype == "car" and c.vehicle.getParameter(vid, "priority_class") != ""
        if vtype not in HIGHLIGHT and not is_fake:
            continue
        color, label = HIGHLIGHT.get(vtype, ((255, 170, 0, 255), "fausse demande (voiture, orange)"))
        try:
            c.vehicle.highlight(vid, color, size=12)
        except Exception:
            pass
        print(f"  t={t:5.0f}s  >>> {vid} entre sur le réseau : {label}, depuis {c.vehicle.getRoadID(vid)}", flush=True)
        if vtype != "bus_transit":
            new.append(vid)
    return new


def _frame(c, rt, ctx, t, degraded):
    veh = []
    for vid in c.vehicle.getIDList():
        x, y = c.vehicle.getPosition(vid)
        veh.append([vid, round(x, 1), round(y, 1), round(c.vehicle.getAngle(vid)), c.vehicle.getTypeID(vid),
                    round(c.vehicle.getSpeed(vid), 1)])
    sig = {tls: c.trafficlight.getRedYellowGreenState(tls) for tls in rt.inters}
    modes = {tls: ("DEGRADED" if degraded[tls] else ctx[tls].mode) for tls in rt.inters}
    return {"t": t, "v": veh, "s": sig, "m": modes}


def _read_trips(path: Path) -> list[dict]:
    out = []
    for tr in ET.parse(path).getroot().findall("tripinfo"):
        out.append({"id": tr.get("id"), "vtype": tr.get("vType"), "depart": float(tr.get("depart")),
                    "arrival": float(tr.get("arrival")), "duration": float(tr.get("duration")),
                    "waiting": float(tr.get("waitingTime")), "stops": int(tr.get("waitingCount")),
                    "time_loss": float(tr.get("timeLoss"))})
    return out


def _metrics(config, spec, trips, rt, queue_mean, queue_max, min_green_viol) -> dict:
    emg = emergency_ids(spec)
    late_bus = {b["id"] for b in spec.get("buses", []) if b.get("late_s", 0) >= 60}

    def m(sel, key):
        vals = [x[key] for x in sel]
        return round(float(np.mean(vals)), 2) if vals else None

    e = [x for x in trips if x["id"] in emg]
    cars = [x for x in trips if x["vtype"] == "car" and x["id"] not in {p["id"] for p in spec["priority"]}]
    buses_late = [x for x in trips if x["vtype"] == "bus_transit" and x["id"].split(".")[0] in late_bus]
    log = rt.safety_log
    return {
        "config": config, "scenario": spec.get("scenario", spec["name"]), "seed": spec["seed"],
        "emg_n": len(e), "emg_time_loss": m(e, "time_loss"), "emg_travel_time": m(e, "duration"),
        "emg_waiting": m(e, "waiting"), "emg_stops": sum(x["stops"] for x in e) if e else None,
        "emg_detail": {x["id"]: round(x["time_loss"], 1) for x in e},
        "critical_time_loss": next((round(x["time_loss"], 1) for x in e if x["id"] == spec.get("critical")), None),
        "bus_late_time_loss": m(buses_late, "time_loss"),
        "car_n": len(cars), "car_time_loss": m(cars, "time_loss"), "car_waiting": m(cars, "waiting"),
        "car_travel_time": m(cars, "duration"), "car_stops": m(cars, "stops"),
        "throughput_900s": sum(1 for x in trips if x["arrival"] <= spec["duration"]),
        "queue_mean": round(queue_mean, 2), "queue_max_lane": int(queue_max),
        "safety_rejected_actions": log.rejected_actions, "safety_rejected_modes": log.rejected_modes,
        "safety_state_violations": log.state_violations, "safety_min_green_violations": min_green_viol,
        "safety_fallbacks": log.fallbacks,
        "switches": sum(ctl.stats.executed_switches for ctl in rt.ctls.values()),
    }
