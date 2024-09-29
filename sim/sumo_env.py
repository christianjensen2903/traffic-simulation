from time import time
from time import sleep
from uuid import uuid4
import threading

from sumolib import checkBinary  # noqa
import traci  # noqa

import os
import sys
from multiprocessing import Queue
import yaml
import pathlib

from loguru import logger

from dtos import (
    TrafficSimulationPredictRequestDto,
    VehicleDto,
    SignalDto,
    LegDto,
    AllowedGreenSignalCombinationDto,
)
import gymnasium as gym


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        groups,
        connections,
        junction,
        signal_groups,
        legs,
        allowed_green_signal_combinations,
        amber_time,
        red_amber_time,
        min_green_time,
    ):
        super(CustomEnv, self).__init__()
        self.maxdistance = 100
        self.groups = groups
        self.connections = connections

        self.junction = junction
        self.signal_groups = signal_groups

        self.legs_dto = []
        self.intern_legs = legs

        for leg_name, l in legs.items():
            # Populate the array for the observable state
            self.legs_dto.append(
                LegDto(name=leg_name, lanes=l["lanes"], signal_groups=l["groups"])
            )

        self.allowed_green_signal_combinations = {}

        for g in allowed_green_signal_combinations:
            self.allowed_green_signal_combinations[g["signal"][0]] = g["allowed"]

        self.allowed_green_signal_comb_dto = []

        for g in allowed_green_signal_combinations:
            self.allowed_green_signal_comb_dto.append(
                AllowedGreenSignalCombinationDto(
                    name=g["signal"][0], groups=g["allowed"]
                )
            )

        self.amber_time = amber_time
        self.red_amber_time = red_amber_time
        self.min_green_time = min_green_time

        self.group_states = {}

        for group in groups:
            self.group_states[group] = ("red", 0)

        self.next_groups = {}

        for group in groups:
            self.next_groups[group] = "red"

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8
        )

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode="human"): ...
    def close(self): ...
