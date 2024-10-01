import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sumo_env import SumoEnv
import numpy as np
from generate_road import RoadDirection
from sumo_wrappers import BinVehicles, DiscritizeSignal
import time
import os
from stable_baselines3.common.callbacks import (
    EveryNTimesteps,
    BaseCallback,
    CallbackList,
)
from wandb.integration.sb3 import WandbCallback

env = SumoEnv(intersection_path="intersections")

env = BinVehicles(env)
env = DiscritizeSignal(env)

model = PPO.load("ppo_sumo")


def play_game():
    env.render()
    obs, info = env.reset()
    done = False
    cum_reward = 0
    while not done:
        action = np.ones(env.action_space.shape)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info, _ = env.step(action)
        cum_reward += reward

    env.toggle_visualize()

    print(f"Total reward: {cum_reward}")
    return cum_reward


class PlayCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        play_game()

        return True


model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(
    total_timesteps=100000,
    log_interval=3,
    progress_bar=True,
    callback=EveryNTimesteps(
        n_steps=5000, callback=CallbackList([PlayCallback(), WandbCallback()])
    ),
)

# Save the model
model.save("ppo_sumo")

# Load the model
# model = PPO.load("ppo_sumo")


n_games = 1

print("Playing games...")

rewards = []
for i in range(n_games):
    start_time = time.time()
    cum_reward = play_game()
    end_time = time.time()
    rewards.append(cum_reward)
    print(f"Game {i+1}: {cum_reward}")

print(f"Average reward: {np.mean(rewards)}")
