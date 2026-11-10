"""Affinage de PPO directement sur le corridor (partage de paramètres entre J1, J2, J3).

Chaque carrefour est un « agent » ; les trois partagent le même réseau de neurones. Un VecEnv
expose les 3 carrefours d'UNE même simulation comme 3 environnements : PPO apprend ainsi avec
l'aval réel (files qui remontent, arrivées en pelotons) au lieu d'un carrefour isolé.
Scénarios aléatoires (graines 200000+), jamais utilisés en évaluation.

Usage : python -m models.finetune_corridor --steps 600000
Sortie : models/checkpoints/ppo_signal_corridor.zip (+ outputs/training/progress_corridor.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
import time
from pathlib import Path

import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv

from coordination.rules import LocalCoordinator, RuleCoordinator
from environment.runtime import DECISION_S, OBS_DIM, Context, Runtime, reward
from simulation.scenarios import CORRIDOR_ENTRIES, CORRIDOR_EXITS, EMERGENCY_TYPES, corridor_flows, write_scenario

ROOT = Path(__file__).parents[1]
TLS = ["J1", "J2", "J3"]


def random_corridor_scenario(seed: int, duration: int = 900) -> dict:
    rng = random.Random(seed)
    level = rng.uniform(0.6, 1.45)
    prio, events = [], []
    ents, exs = list(CORRIDOR_ENTRIES), list(CORRIDOR_EXITS)
    for i in range(rng.choice([0, 1, 2, 2, 3, 3])):
        a = rng.choice(ents)
        b = rng.choice([x for x in exs if x != a])
        params = {}
        if rng.random() < 0.1:
            params["confirmed"] = "false"
        prio.append({"id": f"emg_{i}", "type": rng.choice(EMERGENCY_TYPES), "from": CORRIDOR_ENTRIES[a],
                     "to": CORRIDOR_EXITS[b], "depart": rng.uniform(30, duration - 150), "params": params})
        if rng.random() < 0.1:
            events.append({"time": prio[-1]["depart"] + rng.uniform(5, 30), "type": "siren_off", "vehicle": f"emg_{i}"})
    buses = []
    if rng.random() < 0.6:
        buses.append({"id": "bus", "from": "W02J1", "to": "J32E0", "begin": rng.uniform(0, 100),
                      "period": rng.uniform(120, 240), "late_s": rng.choice([0, 120])})
    return {"name": f"corr_{seed}", "network": "corridor", "duration": duration, "drain": 0,
            "flows": corridor_flows(level), "buses": buses, "priority": prio, "events": events,
            "seed": seed, "demand_level": level}


class CorridorVecEnv(VecEnv):
    def __init__(self, seed0: int = 200000):
        super().__init__(len(TLS), spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32), spaces.Discrete(2))
        self.seed0, self.ep = seed0, 0
        self.tmp = Path(tempfile.mkdtemp(prefix="adsp_corr_"))
        self.rt = None
        self.actions = None
        self.ep_reward = 0.0

    def _new_episode(self):
        if self.rt is not None:
            self.rt.close()
        seed = self.seed0 + self.ep
        self.ep += 1
        self.spec = random_corridor_scenario(seed)
        self.rt = Runtime(write_scenario(self.spec, self.tmp), seed=seed, events=self.spec["events"]).start()
        rng = random.Random(seed)
        # coordinateur d'entraînement : local (B3L) ou corridor à horizon court (B4), au hasard
        if rng.random() < 0.5:
            self.coord = LocalCoordinator(recovery_s=rng.uniform(10, 40), inters=self.rt.inters)
        else:
            self.coord = RuleCoordinator(horizon_s=rng.uniform(20, 40), recovery_s=rng.uniform(10, 40),
                                         inters=self.rt.inters)
        self.ctx = {t: Context() for t in TLS}
        for t in TLS:
            self.rt.ctls[t].take_control(self.rt.t)
        self.ep_reward = 0.0
        return np.stack([self.rt.observation(t, self.ctx[t])[0] for t in TLS])

    def reset(self):
        return self._new_episode()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        rt = self.rt
        for i, t in enumerate(TLS):
            rt.ctls[t].propose(rt.inters[t].green_phases[int(self.actions[i])], rt.t, "ppo")
        before = [rt.ctls[t].stats.executed_switches for t in TLS]
        r = np.zeros(len(TLS), dtype=np.float32)
        for _ in range(DECISION_S):
            rt.step()
            self.ctx = dict(self.coord.update(rt.t, rt.tracker.vehicles))
            for i, t in enumerate(TLS):
                r[i] += reward(rt.c, rt.inters[t], self.ctx[t]) / DECISION_S
        for i, t in enumerate(TLS):
            r[i] -= 0.05 * (rt.ctls[t].stats.executed_switches - before[i])
        self.ep_reward += float(r.mean())
        obs = np.stack([rt.observation(t, self.ctx[t])[0] for t in TLS])
        done = rt.t >= self.spec["duration"]
        infos = [{} for _ in TLS]
        if done:
            for i in range(len(TLS)):
                infos[i]["terminal_observation"] = obs[i]
            infos[0]["episode_stats"] = {"reward": self.ep_reward, "demand": self.spec["demand_level"]}
            obs = self._new_episode()
        return obs, r, np.array([done] * len(TLS)), infos

    def action_masks(self):
        out = []
        for t in TLS:
            self.rt.set_priority(t, self.ctx[t])
            out.append(self.rt.ctls[t].mask(self.rt.t))
        return out

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name == "action_masks":
            masks = self.action_masks()
            idx = range(self.num_envs) if indices is None else indices
            return [masks[i] for i in idx]
        raise NotImplementedError(method_name)

    def get_attr(self, attr_name, indices=None):
        n = self.num_envs if indices is None else len(list(indices))
        return [getattr(self, attr_name, None)] * n

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def close(self):
        if self.rt is not None:
            self.rt.close()

    def seed(self, seed=None):
        return [None] * self.num_envs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--init", default=str(ROOT / "models/checkpoints/ppo_signal.zip"))
    a = ap.parse_args()
    env = CorridorVecEnv()
    model = MaskablePPO.load(a.init, env=env, device="cpu")
    model.learning_rate = 1e-4
    from stable_baselines3.common.utils import get_schedule_fn
    model.lr_schedule = get_schedule_fn(1e-4)
    out = ROOT / "outputs/training/progress_corridor.csv"
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["agent_timesteps", "episode", "reward", "demand", "elapsed_s"])
    t0 = time.time()

    from stable_baselines3.common.callbacks import BaseCallback

    class Log(BaseCallback):
        n = 0

        def _on_step(self):
            st = self.locals["infos"][0].get("episode_stats")
            if st:
                Log.n += 1
                w.writerow([self.num_timesteps, Log.n, round(st["reward"], 3), round(st["demand"], 3),
                            round(time.time() - t0, 1)])
                if Log.n % 20 == 0:
                    f.flush()
                    print(f"[{self.num_timesteps:>8}] épisodes={Log.n} reward={st['reward']:.1f}", flush=True)
            return True

    model.learn(total_timesteps=a.steps, callback=Log(), reset_num_timesteps=True)
    model.save(ROOT / "models/checkpoints/ppo_signal_corridor")
    meta = json.loads((ROOT / "models/checkpoints/ppo_signal.json").read_text())
    meta.update({"finetune_corridor_agent_steps": a.steps, "finetune_time_s": round(time.time() - t0),
                 "finetune_lr": 1e-4})
    (ROOT / "models/checkpoints/ppo_signal_corridor.json").write_text(json.dumps(meta, indent=1))
    f.close()
    env.close()
    print("terminé")


if __name__ == "__main__":
    main()
