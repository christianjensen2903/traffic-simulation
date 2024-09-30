import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sumo_env import SumoEnv
import numpy as np
from generate_road import RoadDirection
from sumo_wrappers import BinVehicles, DiscritizeSignal


env = SumoEnv(config_path="intersections/2")

env = BinVehicles(env)
env = DiscritizeSignal(env)


model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
