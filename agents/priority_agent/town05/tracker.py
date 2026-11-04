"""PriorityTracker for Town05: builds coordination.priority.PriorityVehicle objects from the town's
``state`` messages instead of reading TraCI itself.

Mapping (integration plan, section 2 row 4):
  vClass "emergency"            -> EMERGENCY_MEDICAL, level 3
  vClass "authority"            -> POLICE, level 2
  role "tow" (incident truck)   -> POLICE class, level 1 (open question Q4)
  siren_on  = on_run            (the vehicle's own request, as the town's agents use it)
  confirmed = True with confirm="v2i"; with confirm="camera", once a camera has confirmed its siren
A stop is TLSStop(tls_id, approach=str(link index), edge=lane, distance_m, eta_s, rank).
"""
from __future__ import annotations

from coordination.priority import PRIORITY_LEVEL, PriorityVehicle, TLSStop

V_REF_MIN = 8.0  # m/s, as coordination.priority (ETA = distance / max(speed, 8 m/s))
CLASS_OF = {"emergency": "EMERGENCY_MEDICAL", "authority": "POLICE"}
TOW_LEVEL = 1


class TownTracker:
    def __init__(self, confirm: str = "v2i"):
        if confirm not in ("v2i", "camera"):
            raise ValueError(f"confirm must be v2i or camera, not {confirm!r}")
        self.confirm = confirm
        self.vehicles: dict[str, PriorityVehicle] = {}
        self._first_seen: dict[str, float] = {}
        self._confirmed: set[str] = set()
        self._seen_tls: dict[str, list[str]] = {}   # vid -> signals it has had ahead, in order
        self._passed: dict[str, list[str]] = {}

    @staticmethod
    def classify(v: dict) -> tuple[str, int] | None:
        if v.get("role") == "tow":
            return "POLICE", TOW_LEVEL
        cls = CLASS_OF.get(v.get("vclass", ""))
        return (cls, PRIORITY_LEVEL[cls]) if cls else None

    def update(self, t: float, vehicles: list[dict]) -> dict[str, PriorityVehicle]:
        out: dict[str, PriorityVehicle] = {}
        for v in vehicles:
            vid = v["id"]
            c = self.classify(v)
            if c is None:
                continue
            cls, level = c
            self._first_seen.setdefault(vid, t)
            if v.get("cam_confirmed"):
                self._confirmed.add(vid)
            confirmed = True if self.confirm == "v2i" else vid in self._confirmed
            speed = float(v.get("speed", 0.0))
            pv = PriorityVehicle(vid, cls, level, bool(v.get("on_run")), confirmed, 0.0,
                                 self._first_seen[vid], speed, v.get("edge", ""))
            ahead = []
            for rank, s in enumerate(v.get("next", [])):
                dist = float(s["dist"])
                pv.upcoming.append(TLSStop(s["tls"], str(int(s["link"])), v.get("lane", ""), dist,
                                           dist / max(speed, V_REF_MIN), rank))
                ahead.append(s["tls"])
            # passed = signals it had ahead before and no longer has (RuleCoordinator then gives RECOVERY)
            prev = self._seen_tls.get(vid, [])
            passed = self._passed.setdefault(vid, [])
            for tls in prev:
                if tls not in ahead and tls not in passed:
                    passed.append(tls)
            self._seen_tls[vid] = ahead
            pv.passed = list(passed)
            out[vid] = pv
        for vid in set(self._first_seen) - set(out):  # gone, or no longer a priority vehicle
            for d in (self._first_seen, self._seen_tls, self._passed):
                d.pop(vid, None)
            self._confirmed.discard(vid)
        self.vehicles = out
        return out
