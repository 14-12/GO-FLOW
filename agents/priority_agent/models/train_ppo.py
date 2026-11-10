"""Entraînement de MaskablePPO sur le carrefour isolé (scénarios aléatoires).

Usage : python -m models.train_ppo --steps 800000
Sorties : models/checkpoints/ppo_signal.zip, outputs/training/progress.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from environment.traffic_env import TrafficSignalEnv

ROOT = Path(__file__).parents[1]
CKPT = ROOT / "models" / "checkpoints"
OUT = ROOT / "outputs" / "training"


class EpisodeLogger(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.t0 = time.time()
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["timesteps", "episode", "reward", "switches", "demand", "elapsed_s"])
        self.n = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            st = info.get("episode_stats")
            if st:
                self.n += 1
                self.w.writerow([self.num_timesteps, self.n, round(st["reward"], 3), st["switches"],
                                 round(st["demand"], 3), round(time.time() - self.t0, 1)])
                if self.n % 50 == 0:
                    self.f.flush()
                    print(f"[{self.num_timesteps:>8}] épisodes={self.n} reward={st['reward']:.1f}", flush=True)
        return True

    def _on_training_end(self) -> None:
        self.f.close()


def make_env(i: int):
    return lambda: TrafficSignalEnv(seed_offset=i + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800_000)
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    CKPT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    vec = SubprocVecEnv([make_env(i) for i in range(args.envs)]) if args.envs > 1 else DummyVecEnv([make_env(0)])
    hp = dict(n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.98, gae_lambda=0.95,
              ent_coef=0.01, clip_range=0.2, policy_kwargs=dict(net_arch=[128, 128]))
    model = MaskablePPO("MlpPolicy", vec, verbose=0, seed=args.seed, device="cpu", **hp)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=EpisodeLogger(OUT / "progress.csv"))
    model.save(CKPT / "ppo_signal")
    meta = {"algo": "MaskablePPO (sb3-contrib)", "timesteps": args.steps, "envs": args.envs,
            "train_time_s": round(time.time() - t0), "hyperparameters": {k: str(v) for k, v in hp.items()},
            "obs_dim": int(np.prod(vec.observation_space.shape)), "decision_interval_s": 5}
    (CKPT / "ppo_signal.json").write_text(json.dumps(meta, indent=1))
    vec.close()
    print("terminé", meta)


if __name__ == "__main__":
    main()
