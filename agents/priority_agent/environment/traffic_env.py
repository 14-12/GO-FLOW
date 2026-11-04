"""Environnement Gymnasium d'entraînement : un carrefour isolé, contrôlé par PPO.

- Chaque épisode = un scénario aléatoire (demande, urgences, bus, demandes non confirmées,
  sirènes coupées) -> PPO rencontre tous les modes locaux pendant l'entraînement.
- Le contexte stratégique est produit par un coordinateur à règles aux paramètres tirés au
  hasard (horizon d'anticipation, durée de récupération), pour couvrir les situations que
  créeront les coordinateurs B3L / B4 / LLM au déploiement.
- Le masque d'actions vient du Safety Controller (MaskablePPO).
- Une décision toutes les 5 s ; action = phase verte suivante (0 : N-S, 1 : E-O).
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np

from coordination.rules import RuleCoordinator
from environment.runtime import DECISION_S, OBS_DIM, Context, Runtime, reward
from simulation.scenarios import random_training_scenario, write_scenario

TLS = "C"


class TrafficSignalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed_offset: int = 0, backend: str = "libsumo", log_dir: str | None = None):
        super().__init__()
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.backend = backend
        self._episode = 0
        self._seed_offset = seed_offset
        self._tmp = Path(tempfile.mkdtemp(prefix="adsp_train_"))
        self.rt: Runtime | None = None
        self.log_dir = log_dir

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.rt is not None:
            self.rt.close()
        scen_seed = (options or {}).get("scenario_seed")
        if scen_seed is None:
            scen_seed = self._seed_offset * 100000 + self._episode
        self._episode += 1
        self.spec_ = random_training_scenario(scen_seed)
        cfg = write_scenario(self.spec_, self._tmp)
        self.rt = Runtime(cfg, seed=scen_seed, backend=self.backend, events=self.spec_["events"],
                          label=f"train{self._seed_offset}").start()
        rng = random.Random(scen_seed)
        inter = self.rt.inters[TLS]
        self.coord = RuleCoordinator(horizon_s=rng.uniform(20, 90), recovery_s=rng.uniform(10, 90),
                                     scheduled=[dict(p) for p in self.spec_["priority"]] if rng.random() < 0.5 else [],
                                     edge_tls={e: (TLS, a) for a, e in inter.incoming.items()},
                                     inters=self.rt.inters)
        self.ctl = self.rt.ctls[TLS]
        self.ctl.take_control(self.rt.t)
        self.ctx = Context()
        self.ep_stats = {"switches": 0, "reward": 0.0, "emg_stopped_s": 0.0}
        obs, _ = self.rt.observation(TLS, self.ctx)
        return obs, {}

    def action_masks(self) -> np.ndarray:
        self.rt.set_priority(TLS, self.ctx)
        return self.ctl.mask(self.rt.t)

    def step(self, action):
        rt = self.rt
        inter = rt.inters[TLS]
        target = inter.green_phases[int(action)]
        before = self.ctl.current_green
        self.ctl.propose(target, rt.t, "ppo")
        switched = target != before
        r = -0.05 if switched else 0.0
        for _ in range(DECISION_S):
            rt.step()
            self.ctx = self.coord.update(rt.t, rt.tracker.vehicles)[TLS]
            r += reward(rt.c, inter, self.ctx) / DECISION_S
        self.ep_stats["switches"] += int(switched)
        self.ep_stats["reward"] += r
        obs, _ = rt.observation(TLS, self.ctx)
        terminated = rt.t >= self.spec_["duration"]
        info = {}
        if terminated:
            info["episode_stats"] = dict(self.ep_stats, demand=self.spec_["demand_level"])
        return obs, float(r), terminated, False, info

    def close(self):
        if self.rt is not None:
            self.rt.close()
            self.rt = None
