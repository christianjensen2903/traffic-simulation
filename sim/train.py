import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sumo_env import SumoEnv
import numpy as np
from sumo_wrappers import BinVehicles, DiscritizeSignal
from wandb.integration.sb3 import WandbCallback
import wandb
from stable_baselines3.common.vec_env import SubprocVecEnv

LOG_WANDB = False

if LOG_WANDB:
    wandb.init(project="sumo-rl")


if __name__ == "__main__":

    def create_env():
        env = SumoEnv(intersection_path="intersections")
        env = BinVehicles(env)
        env = DiscritizeSignal(env)

        return env

    env = SubprocVecEnv([create_env for _ in range(4)])

    if LOG_WANDB:
        callback = WandbCallback()
    else:
        callback = None

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(
        total_timesteps=50000,
        log_interval=1,
        progress_bar=True,
        callback=callback,
    )

    # ave the model
    model.save("ppo_sumo")
