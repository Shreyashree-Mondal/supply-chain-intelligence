"""Train a PPO replenishment agent and compare it with the (ROP, EOQ) policy on identical demand."""
from __future__ import annotations

import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from src import config
from src.optimization.rl_env import InventoryEnv, evaluate, load_product, sq_policy

pid = int(sys.argv[1]) if len(sys.argv) > 1 else 365
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
prod = load_product(pid)

venv = make_vec_env(lambda: InventoryEnv(prod, demand_scale=None), n_envs=4, seed=0)
venv = VecNormalize(venv, norm_obs=False, norm_reward=True)   # scales rewards only
model = PPO("MlpPolicy", venv, seed=0, device="cpu", verbose=0, gamma=0.999, n_steps=2048, batch_size=512, ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128]))
model.learn(total_timesteps=steps)

ppo = lambda env, obs: int(model.predict(obs, deterministic=True)[0])
base = evaluate(lambda: InventoryEnv(prod), sq_policy)
rl = evaluate(lambda: InventoryEnv(prod), ppo)

config.MODELS_DIR.mkdir(exist_ok=True)
model.save(config.MODELS_DIR / f"rl_ppo_{pid}")
print("product", pid, "| timesteps", steps)
print("(ROP, EOQ):", {k: round(v, 3) for k, v in base.items()})
print("PPO agent :", {k: round(v, 3) for k, v in rl.items()})
print("cost change vs baseline: %.1f%%" % (100 * (rl["annual_cost"] - base["annual_cost"]) / base["annual_cost"]))
