import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sumo_env import SumoEnv
import numpy as np
from torch import nn
import torch
from sumo_wrappers import BinVehicles, DiscritizeSignal, DiscretizeLegs
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EveryNTimesteps,
    CallbackList,
    EvalCallback,
)
import wandb
from typing import Callable, Tuple
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from feature_extractor import FeatureExtractor, CustomActorCriticPolicy

LOG_WANDB = True


if LOG_WANDB:
    run = wandb.init(project="sumo", sync_tensorboard=True)


if __name__ == "__main__":

    def create_env():
        env = SumoEnv(intersection_path="intersections")
        env = BinVehicles(env)
        env = DiscritizeSignal(env)
        env = DiscretizeLegs(env)

        return env

    visualize_env = create_env()
    visualize_env.render()

    # env = SubprocVecEnv([create_env for _ in range(4)])
    env = create_env()

    visualize_game_callback = EvalCallback(
        visualize_env, deterministic=True, eval_freq=50000, n_eval_episodes=1
    )

    callbacks = [visualize_game_callback]

    if LOG_WANDB:
        callbacks.append(WandbCallback(verbose=2))

    tensorboard_log = f"runs/{run.id}" if LOG_WANDB else None

    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(
            input_channels=4,  # Replace with the actual input channels
            hidden_channels=32,
            hidden_kernel_size=3,
            hidden_stride=1,
            blocks=4,  # Number of residual blocks
        ),
        net_arch=dict(
            hidden_channels=32,
            hidden_kernel_size=3,
            hidden_stride=1,
            blocks=4,  # Number of residual blocks
            feature_dim=2,
            last_layer_dim_pi=304,
            last_layer_dim_vf=304,
        ),
    )
    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
    )

    model.learn(
        total_timesteps=200000,
        log_interval=1,
        progress_bar=True,
        callback=CallbackList(callbacks),
    )

    # ave the model
    model.save("ppo_sumo")
