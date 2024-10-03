from stable_baselines3 import PPO
from sumo_env import SumoEnv

from sumo_wrappers import (
    DiscritizeSignal,
    DiscretizeLegs,
    SimpleObs,
    DiscretizeAndTrackLanes,
    TrackLanes,
)
import time


model = PPO.load("ppo_sumo")
env = SumoEnv(intersection_path="intersections")
env.visualize = True
env = DiscritizeSignal(env)
env = TrackLanes(env, "intersection_3")
env = SimpleObs(env)

obs, _ = env.reset()
done = False
cum_reward = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, _, info = env.step(action)
    time.sleep(0.25)
    cum_reward += rewards
print(f"Total reward: {cum_reward}")
