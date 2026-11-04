"""Robustness check: policies tuned/trained at normal demand, tested when demand shifts."""
from __future__ import annotations

import math
import sys

import numpy as np
from stable_baselines3 import PPO

from src import config
from src.optimization.rl_env import InventoryEnv, N_ACTIONS, evaluate, load_product, sq_policy

pid = int(sys.argv[1]) if len(sys.argv) > 1 else 365
prod = load_product(pid)
VAL = dict(episodes=20, seed0=5000)
TEST = dict(episodes=30, seed0=1000)
normal = lambda: InventoryEnv(prod)


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


best_sq = min(((m, q) for m in [0.5, 0.75, 1.0, 1.25, 1.5] for q in range(1, N_ACTIONS)),
              key=lambda a: evaluate(normal, sq(*a), **VAL)["annual_cost"])
best_bs = min(np.arange(0, 4.01, 0.25), key=lambda k: evaluate(normal, base_stock(k), **VAL)["annual_cost"])
model = PPO.load(config.MODELS_DIR / f"rl_ppo_{pid}", device="cpu")
ppo = lambda env, obs: int(model.predict(obs, deterministic=True)[0])
policies = {"Current (ROP, EOQ)": sq_policy, "Tuned (s, Q)": sq(*best_sq),
            "Tuned base-stock": base_stock(best_bs), "PPO agent": ppo}

print("product", pid, "(policies tuned/trained at demand x1.0)")
for scale in [0.7, 1.0, 1.3]:
    print(f"\n-- demand x{scale} --")
    for name, pol in policies.items():
        r = evaluate(lambda: InventoryEnv(prod, demand_scale=scale), pol, **TEST)
        print(f"{name:22s} cost ${r['annual_cost']:9,.0f} | fill {r['fill_rate']:.3f} | orders/yr {r['orders_per_year']:6.1f}")
