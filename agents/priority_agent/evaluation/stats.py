"""Tests appariés (mêmes scénarios, mêmes graines) entre configurations du corridor.

Différence moyenne a - b, intervalle de confiance bootstrap à 95 % et test de Wilcoxon.
Métrique pondérée S7 : weighted_emg_loss = CRITICAL_WEIGHT x critique + transfert (CRITICAL_WEIGHT = 3).
Inclut S7 (message du centre de régulation, results/eval_dispatch.csv) : temps perdu du patient critique,
de l'ambulance en transfert non urgent, de toutes les urgences et des voitures.
Avec n graines, le plus petit p de Wilcoxon possible est 2/2^n (0,0625 pour n = 5) : il faut au moins
6 graines pour pouvoir atteindre p < 0,05.
Usage : python -m evaluation.stats   ->  results/stats_tests.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).parents[1] / "results"
CRITICAL_WEIGHT = 3.0  # S7 : une seconde perdue par le patient en arrêt cardiaque compte 3 fois plus
PAIRS = [
    ("B3L_ppo_local", "B1_fixe", "emg_time_loss", True),
    ("B3L_ppo_local", "B3_ppo", "emg_time_loss", True),
    ("B3L_ppo_local", "B2AP_actuated_preemption", "emg_time_loss", True),
    ("B4_ppo_regles", "B3L_ppo_local", "emg_time_loss", True),
    ("B4_ppo_regles", "B2AP_actuated_preemption", "emg_time_loss", True),
    ("B5_ppo_llm", "B4_ppo_regles", "emg_time_loss", True),
    ("B3L_ppo_local", "B1_fixe", "car_time_loss", False),
    ("B3L_ppo_local", "B2_actuated", "car_time_loss", False),
    ("B3L_ppo_local", "B2AP_actuated_preemption", "car_time_loss", False),
    ("B5_ppo_llm", "B4_ppo_regles", "car_time_loss", False),
    ("B3L_ppo_local", "B2AP_actuated_preemption", "bus_late_time_loss", False),
]
DISPATCH_PAIRS = [(a, b, m) for a, b in (("B5_ppo_llm", "B4_ppo_regles"), ("B5_ppo_llm", "B2AP_actuated_preemption"))
                  for m in ("critical_time_loss", "routine_time_loss", "emg_time_loss", "weighted_emg_loss", "car_time_loss")]


def _test(x, y, keys, metric):
    m = x.merge(y, on=keys, suffixes=("_a", "_b")).dropna(subset=[f"{metric}_a", f"{metric}_b"])
    diff = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
    if len(diff) == 0:
        return None
    rng = np.random.default_rng(0)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    return {"n": len(diff), "mean_diff": round(diff.mean(), 2),
            "ci95_low": round(np.percentile(bs, 2.5), 2), "ci95_high": round(np.percentile(bs, 97.5), 2),
            "wilcoxon_p": round(p, 4), "p_min_possible": round(2 / 2 ** len(diff), 4)}


def main():
    d = pd.read_csv(RES / "eval_corridor.csv")
    rows = []
    for a, b, metric, prio in PAIRS:
        x, y = d[d.config == a], d[d.config == b]
        if x.empty or y.empty:
            continue
        if prio:
            x, y = x[x.scenario != "S1_normal"], y[y.scenario != "S1_normal"]
        r = _test(x, y, ["scenario", "seed"], metric)
        if r:
            rows.append({"a": a, "b": b, "scenario": "S2-S6", "metric": metric, **r})
    disp = RES / "eval_dispatch.csv"
    if disp.exists():
        d7 = pd.read_csv(disp)
        # temps perdu pondéré : w x patient critique + transfert non urgent
        d7["weighted_emg_loss"] = CRITICAL_WEIGHT * d7["critical_time_loss"] + d7["routine_time_loss"]
        for a, b, metric in DISPATCH_PAIRS:
            x, y = d7[d7.config == a], d7[d7.config == b]
            if x.empty or y.empty or metric not in d7:
                continue
            r = _test(x, y, ["seed"], metric)
            if r:
                rows.append({"a": a, "b": b, "scenario": "S7_dispatch", "metric": metric, **r})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "stats_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    if (out.n < 6).any():
        print("\nAttention : avec moins de 6 graines, p < 0,05 est impossible (voir p_min_possible).")


if __name__ == "__main__":
    main()
