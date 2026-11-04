"""Compare PPO with tuned classical baselines on identical unseen demand."""
from __future__ import annotations

import math
import sys

import numpy as np
from stable_baselines3 import PPO

from src import config
from src.optimization.rl_env import InventoryEnv, N_ACTIONS, evaluate, load_product, sq_policy

pid = int(sys.argv[1]) if len(sys.argv) > 1 else 365
prod = load_product(pid)
factory = lambda: InventoryEnv(prod)
VAL = dict(episodes=20, seed0=5000)     # tuning seeds
TEST = dict(episodes=30, seed0=1000)    # same unseen seeds used for PPO


def sq(s_mult, q_units):
    def pol(env, obs):
        pos = env.on_hand + sum(env.pipeline)
        return q_units if pos <= prod["rop"] * s_mult else 0
    return pol


def base_stock(k):
    L = prod["lead_days"]
    S = prod["daily_mean"] * (L + 1) + k * prod["daily_sigma"] * math.sqrt(L + 1)
    def pol(env, obs):
        pos = env.on_hand + sum(env.pipeline)
        return int(min(N_ACTIONS - 1, max(0, math.ceil((S - pos) / env.unit))))
    return pol


best_sq = min(((m, q) for m in [0.5, 0.75, 1.0, 1.25, 1.5] for q in range(1, N_ACTIONS)),
              key=lambda a: evaluate(factory, sq(*a), **VAL)["annual_cost"])
best_bs = min(np.arange(0, 4.01, 0.25),
              key=lambda k: evaluate(factory, base_stock(k), **VAL)["annual_cost"])
model = PPO.load(config.MODELS_DIR / f"rl_ppo_{pid}", device="cpu")
ppo = lambda env, obs: int(model.predict(obs, deterministic=True)[0])

rows = {"Current (ROP, EOQ)": sq_policy,
        f"Tuned (s, Q) [s x{best_sq[0]}, Q={best_sq[1]} units]": sq(*best_sq),
        f"Tuned base-stock [k={best_bs:.2f}]": base_stock(best_bs),
        "PPO agent": ppo}
res = {name: evaluate(factory, pol, **TEST) for name, pol in rows.items()}
print("product", pid)
for name, r in res.items():
    print(f"{name:48s} cost ${r['annual_cost']:9,.0f} | fill {r['fill_rate']:.3f} | orders/yr {r['orders_per_year']:6.1f}")
best_classic = min(v["annual_cost"] for k, v in res.items() if k != "PPO agent")
print("PPO vs best classical baseline: %.1f%%" % (100 * (res["PPO agent"]["annual_cost"] - best_classic) / best_classic))
