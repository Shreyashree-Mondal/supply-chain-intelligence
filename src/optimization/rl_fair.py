"""Fair comparison: classical policies tuned on a MIX of demand levels vs the PPO agent."""
from __future__ import annotations

import math
import sys

import numpy as np
from stable_baselines3 import PPO

from src import config
from src.optimization.rl_env import InventoryEnv, N_ACTIONS, evaluate, load_product, sq_policy

SCALES = [0.7, 1.0, 1.3]
VAL = dict(episodes=10, seed0=5000)
TEST = dict(episodes=30, seed0=1000)


def make(prod):
    def sq(s_mult, q_units):
        def pol(env, obs):
            return q_units if env.on_hand + sum(env.pipeline) <= prod["rop"] * s_mult else 0
        return pol

    def base_stock(k):
        L = prod["lead_days"]
        S = prod["daily_mean"] * (L + 1) + k * prod["daily_sigma"] * math.sqrt(L + 1)
        def pol(env, obs):
            pos = env.on_hand + sum(env.pipeline)
            return int(min(N_ACTIONS - 1, max(0, math.ceil((S - pos) / env.unit))))
        return pol
    return sq, base_stock


def mixed_cost(prod, pol):
    return np.mean([evaluate(lambda: InventoryEnv(prod, demand_scale=s), pol, **VAL)["annual_cost"] for s in SCALES])


for pid in [int(a) for a in sys.argv[1:]] or [365]:
    prod = load_product(pid)
    sq, base_stock = make(prod)
    best_sq = min(((m, q) for m in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5] for q in range(1, N_ACTIONS)),
                  key=lambda a: mixed_cost(prod, sq(*a)))
    best_bs = min(np.arange(0, 5.01, 0.5), key=lambda k: mixed_cost(prod, base_stock(k)))
    model = PPO.load(config.MODELS_DIR / f"rl_ppo_{pid}", device="cpu")
    ppo = lambda env, obs: int(model.predict(obs, deterministic=True)[0])
    pols = {"Current (ROP, EOQ)": sq_policy, "Mix-tuned (s, Q)": sq(*best_sq),
            "Mix-tuned base-stock": base_stock(best_bs), "PPO agent": ppo}
    print(f"\nproduct {pid} (classical tuned on demand x0.7/1.0/1.3; s x{best_sq[0]}, Q={best_sq[1]}, k={best_bs:.1f})")
    for scale in SCALES:
        print(f"  demand x{scale}")
        for name, pol in pols.items():
            r = evaluate(lambda: InventoryEnv(prod, demand_scale=scale), pol, **TEST)
            print(f"    {name:22s} cost ${r['annual_cost']:9,.0f} | fill {r['fill_rate']:.3f}")
