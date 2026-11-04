"""The B4 rule coordinator (coordination.rules.RuleCoordinator) on Town05's junctions.

Unchanged rules: class first (ambulance > police > tow), then first seen, then ETA; every signal
whose ETA is under the horizon is prepared (corridor); RECOVERY for recovery_s after the passage;
ModeSafety (siren on and confirmed, rate limit per junction, valid junction and link).
Changed for Town05: a conflict is judged on link indices (no car green serves both links) instead
of the N/E/S/W approach's best phase, and the plan carries every requested link that one green can
serve together with the chosen vehicle's link.
"""
from __future__ import annotations

from coordination.rules import RuleCoordinator
from safety.validator import ModeSafety, SafetyConfig, SafetyLog
from town05.conflicts import LinkMap, TownJunction

EMERGENCY_MODES = ("EMERGENCY_PREEMPTION", "CONFLICT_RESOLUTION")


class TownRuleCoordinator(RuleCoordinator):
    def __init__(self, links: LinkMap, horizon_s: float = 30.0, recovery_s: float = 10.0,
                 rate_limit: bool = True, safety_cfg: SafetyConfig | None = None):
        cfg = safety_cfg or SafetyConfig()
        if not rate_limit:
            cfg.max_preemptions = 10 ** 6
        self.safety_log = SafetyLog()
        inters = {tls: TownJunction(tls, links) for tls in links.greens}
        super().__init__(horizon_s=horizon_s, recovery_s=recovery_s,
                         mode_safety=ModeSafety(cfg, self.safety_log), inters=inters)
        self.links = links
        self.plan_links: dict[str, list[int]] = {}

    def update(self, t: float, vehicles: dict):
        cands = self.candidates(t, vehicles)
        cands = {tls: lst for tls, lst in cands.items() if tls in self.inters}
        chosen, links = {}, {}
        for tls, lst in cands.items():
            lst = [(pv, st) for pv, st in lst if self.links.known(tls, int(st.approach))]
            if not lst:
                continue
            lst.sort(key=lambda x: (-(x[0].priority_level + self.bonus.get(x[0].vehicle_id, 0.0)),
                                    x[0].first_seen, x[1].eta_s))
            best_pv, best_stop = lst[0]
            mine = int(best_stop.approach)
            together = [mine]
            n_conflict = 0
            for pv, st in lst[1:]:
                other = int(st.approach)
                if self.links.compatible(tls, mine, other):
                    if other not in together:
                        together.append(other)
                elif pv.is_emergency:
                    n_conflict += 1
            chosen[tls] = (best_pv, best_stop, n_conflict)
            links[tls] = together
        for tls in self.inters:
            self.ctx[tls] = self._finish(t, tls, chosen, vehicles)
        self.plan_links = {tls: links.get(tls, []) for tls, c in self.ctx.items() if c.mode in EMERGENCY_MODES}
        return self.ctx
