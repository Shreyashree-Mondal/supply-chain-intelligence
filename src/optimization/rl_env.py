"""Gymnasium inventory-replenishment environment (RL extension of the policy table)."""
from __future__ import annotations

import math
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src import config

STOCKOUT_COST_FRAC = 0.25   # ASSUMPTION: lost-sale penalty per unit = 25% of unit price (not in config)
EPISODE_DAYS = 365
N_ACTIONS = 7               # order 0..6 units, where one unit = EOQ / 2


def load_product(product_id: int, policy_csv=None) -> dict:
    p = pd.read_csv(policy_csv or config.POLICY_CSV)
    r = p[p["Product Card Id"] == product_id].iloc[0]
    lead = max(1, math.ceil(float(r["Lead_Time_Days"])))
    mean = float(r["Latest_Forecast"]) / 30.0
    sigma = float(r["Monthly_Sigma"]) / math.sqrt(30.0)
    # same formula as src/optimization/inventory.py, but with the whole-day lead time used in the simulation
    rop = round(mean * lead + config.Z_SCORE * sigma * math.sqrt(lead))
    return {"id": int(r["Product Card Id"]), "daily_mean": mean, "daily_sigma": sigma,
            "lead_days": lead, "price": float(r["Product Price"]), "rop": float(rop),
            "eoq": max(1.0, float(r["EOQ"]))}


class InventoryEnv(gym.Env):
    def __init__(self, prod: dict, demand_scale=1.0):
        self.rand_scale = demand_scale is None
        self.p, self.scale, self.L = prod, (1.0 if demand_scale is None else demand_scale), prod["lead_days"]
        self.unit = max(1.0, round(prod["eoq"] / 2))
        self.hold_per_day = config.HOLDING_RATE * prod["price"] / 365.0
        self.stockout_cost = STOCKOUT_COST_FRAC * prod["price"]
        self.norm = max(prod["daily_mean"], 1.0)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.L + 2,), dtype=np.float32)

    def _obs(self):
        return (np.array([self.on_hand, *self.pipeline, self.on_hand + sum(self.pipeline)]) / self.norm).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.rand_scale:
            self.scale = float(self.np_random.uniform(0.7, 1.3))
        if self.rand_scale:
            self.scale = float(self.np_random.uniform(0.7, 1.3))
        if self.rand_scale:
            self.scale = float(self.np_random.uniform(0.7, 1.3))
        if self.rand_scale:
            self.scale = float(self.np_random.uniform(0.7, 1.3))
        self.on_hand = self.p["rop"] + self.p["eoq"] / 2
        self.pipeline = [0.0] * self.L
        self.day = 0
        return self._obs(), {}

    def step(self, action):
        order = float(action) * self.unit
        cost = config.ORDERING_COST if order > 0 else 0.0
        self.on_hand += self.pipeline[0]                    # stock ordered L days ago arrives
        self.pipeline = self.pipeline[1:] + [order]
        mu, sd = self.p["daily_mean"] * self.scale, self.p["daily_sigma"] * self.scale
        demand = max(0.0, round(float(self.np_random.normal(mu, sd))))
        sold = min(self.on_hand, demand)
        lost = demand - sold
        self.on_hand -= sold
        cost += self.hold_per_day * self.on_hand + self.stockout_cost * lost
        self.day += 1
        info = {"demand": demand, "lost": lost, "ordered": order}
        return self._obs(), -cost, False, self.day >= EPISODE_DAYS, info


def sq_policy(env, obs):
    """Your current policy: when inventory position <= reorder point, order one EOQ."""
    position = env.on_hand + sum(env.pipeline)
    if position <= env.p["rop"]:
        return int(min(N_ACTIONS - 1, round(env.p["eoq"] / env.unit)))
    return 0


def evaluate(env_factory, policy_fn, episodes: int = 30, seed0: int = 1000) -> dict:
    cost = lost = demand = orders = 0.0
    for ep in range(episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy_fn(env, obs))
            cost -= r
            lost += info["lost"]
            demand += info["demand"]
            orders += info["ordered"] > 0
            done = term or trunc
    return {"annual_cost": cost / episodes, "fill_rate": 1 - lost / max(demand, 1.0),
            "orders_per_year": orders / episodes}


if __name__ == "__main__":
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    prod = load_product(pid)
    print("product:", prod)
    print("(ROP, EOQ) baseline:", evaluate(lambda: InventoryEnv(prod), sq_policy))
