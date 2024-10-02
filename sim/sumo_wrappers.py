import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from sumo_env import SumoEnv, TrafficColor
from generate_road import (
    LaneType,
    RoadDirection,
    index_to_direction,
    direction_to_index,
)
import numpy as np
import time
import json


class BinVehicles(gym.ObservationWrapper):
    """A wrapper that discretizes the vehicle observations into bins"""

    def __init__(self, env: SumoEnv, n_bins: int = 10):
        super().__init__(env)

        self.n_bins = n_bins
        self.max_distance = env.max_distance
        self.bin_distance = int(self.max_distance / n_bins)

        # TODO: Waiting times not included from api. Make some kind of tracking

        self.bin_edges = np.linspace(0, self.max_distance, n_bins + 1)
        self.bin_pairs = list(zip(self.bin_edges[:-1], self.bin_edges[1:]))

        self.vehicles_space = spaces.Box(
            low=0,
            high=200,  # It don't think there would be more than 200 vehicles in a bin
            shape=(
                4,  # Max 4 incoming roads
                n_bins,
                7,  # 7 scalar features
            ),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "vehicles": self.vehicles_space,
                "signals": env.signal_space,
                "legs": env.legs_space,
            }
        )

    def _get_bin_index(self, distance: float) -> int:
        """Get the index of the bin for the given distance"""
        bin_index = (
            np.digitize(distance, self.bin_edges) - 1
        )  # Adjust for 0-based index
        bin_index = np.clip(
            bin_index, 0, len(self.bin_edges) - 2
        )  # Ensure bin_index is within range
        return bin_index

    def _bin_vehicles(self, vehicles: dict) -> dict[str, list[list[dict]]]:
        """
        Bin the vehicles into the distance bins
        Returns a dictionary with the binned vehicles for each leg
        Ordered by the bin index
        """
        binned_vehicles = {
            leg.name: [[] for i in range(self.n_bins + 1)] for leg in self.env.legs
        }
        for leg_name, vehicles in vehicles.items():
            for vehicle in vehicles:
                distance = vehicle["distance"]
                bin_index = self._get_bin_index(distance)
                binned_vehicles[leg_name][bin_index].append(vehicle)

        return binned_vehicles

    def _get_direction_index(self, group_name: str) -> int:
        direction = group_name.split("_")[0]
        return direction_to_index(RoadDirection(direction))

    def observation(self, obs: dict) -> dict:
        vehicles = obs["vehicles"]
        assert isinstance(vehicles, dict)

        binned_vehicles = self._bin_vehicles(vehicles)

        # Calculate the features for each bin
        discrete_vehicles = np.zeros(self.vehicles_space.shape)
        for leg_name, bins in binned_vehicles.items():
            direction_index = self._get_direction_index(leg_name)
            for i, bin in enumerate(bins):

                if len(bin) == 0:
                    continue

                speeds = [vehicle["speed"] for vehicle in bin]
                distances = [vehicle["distance"] for vehicle in bin]
                amount = len(bin)
                max_speed = max(speeds)
                min_speed = min(speeds)
                avg_speed = sum(speeds) / len(speeds)
                max_distance = max(distances)
                min_distance = min(distances)
                avg_distance = sum(distances) / len(distances)
                discrete_vehicles[direction_index][i] = [
                    amount,
                    max_speed,
                    min_speed,
                    avg_speed,
                    max_distance,
                    min_distance,
                    avg_distance,
                ]

        obs["vehicles"] = discrete_vehicles

        return obs


class DiscritizeSignal(gym.ObservationWrapper):
    """A wrapper that discretizes the signal observations"""

    def __init__(self, env: SumoEnv):
        super().__init__(env)

        self.signals_space = spaces.Box(
            low=0,
            high=max(
                [self.env.amber_time, self.env.red_amber_time, self.env.min_green_time]
            ),
            shape=(
                4,  # Max 4 incoming roads
                3,  # Either left, straight, right
                len(TrafficColor),  # Red, RedAmber, Amber, Green
            ),
            dtype=np.float32,
        )
        self.vehicles_space = self.env.vehicles_space
        self.legs_space = self.env.legs_space
        self.observation_space = spaces.Dict(
            {
                "signals": self.signals_space,
                "vehicles": self.vehicles_space,
                "legs": self.legs_space,
            }
        )

    def _get_direction_index(self, group_name: str) -> int:
        """Get the index of the direction in the signal space. 0: N, 1: E, 2: S, 3: W (clockwise)"""
        direction = group_name.split("_")[0]
        return direction_to_index(RoadDirection(direction))

    def _get_lane_index(self, lane: str) -> int:
        """Get the index of the lane in the signal space. 0: Left, 1: Straight, 2: Right"""
        return ["left", "straight", "right"].index(lane)

    def _get_lane_indices(self, group_name: str) -> list[int]:
        """
        Get the indices of the lane in the signal space.
        Can be multiple in the case of all, straight right and straight left
        0: Left, 1: Straight, 2: Right
        """
        lanes = group_name.split("_")[1:]
        if lanes == ["all"]:
            return [0, 1, 2]
        else:
            return [self._get_lane_index(lane) for lane in lanes]

    def _get_color_index(self, color: TrafficColor) -> int:
        """Get the index of the color in the signal space. 0: Red, 1: RedAmber, 2: Amber, 3: Green"""
        return [
            TrafficColor.RED,
            TrafficColor.REDAMBER,
            TrafficColor.AMBER,
            TrafficColor.GREEN,
        ].index(color)

    def observation(self, obs: dict) -> dict:
        signals = obs["signals"]
        assert isinstance(signals, dict)

        signal_obs = np.zeros(self.signals_space.shape)
        for group, signal in signals.items():

            color = TrafficColor(signal["color"])
            time = signal["time"]
            direction_index = self._get_direction_index(group)
            lane_indices = self._get_lane_indices(group)
            color_index = self._get_color_index(color)
            for lane_index in lane_indices:
                signal_obs[direction_index][lane_index][color_index] = time

        obs["signals"] = signal_obs
        return obs


class DiscretizeLegs(gym.ObservationWrapper):
    """A wrapper that discretizes the legs observations"""

    def __init__(self, env: SumoEnv):
        super().__init__(env)
        self.legs_space = spaces.MultiBinary(
            [
                4,  # Max 4 incoming roads
                5,  # Max 5 lanes
                len(LaneType),
            ]
        )
        self.signals_space = env.signals_space
        self.vehicles_space = env.vehicles_space
        self.observation_space = spaces.Dict(
            {
                "signals": self.signals_space,
                "vehicles": self.vehicles_space,
                "legs": self.legs_space,
            }
        )

    def _get_direction_index(self, group_name: str) -> int:
        direction = group_name.split("_")[0]
        return direction_to_index(RoadDirection(direction))

    def observation(self, observation: dict) -> dict:
        legs = observation["legs"]
        assert isinstance(legs, dict)

        leg_obs = np.zeros(self.legs_space.shape)
        for group, lanes in legs.items():
            for i, lane_enc in enumerate(lanes):
                direction_index = self._get_direction_index(group)
                leg_obs[direction_index][i] = lane_enc

        observation["legs"] = leg_obs
        return observation


class SimpleObs(gym.ObservationWrapper):
    def __init__(self, env: SumoEnv):
        super().__init__(env)

        self.observation_space = spaces.Dict(
            {
                "signals": env.signals_space,
                "vehicles": spaces.Box(
                    low=0,
                    high=90,
                    shape=(4, 2),
                    dtype=np.float32,
                ),
            }
        )

    def _get_direction_index(self, group_name: str) -> int:
        direction = group_name.split("_")[0]
        return direction_to_index(RoadDirection(direction))

    def observation(self, obs: dict) -> dict:
        vehicles = obs["vehicles"]

        # Get the amount of cars stopped and not stopped in each leg
        legs = np.zeros((4, 2))
        for leg_name, vehicles in vehicles.items():
            direction_index = self._get_direction_index(leg_name)
            for vehicle in vehicles:
                if vehicle["speed"] < 0.5:
                    legs[direction_index][0] += 1
                else:
                    legs[direction_index][1] += 1

        return {
            "signals": obs["signals"],
            "vehicles": legs,
        }


class OnlyLegalCombinations(gym.ActionWrapper):
    def __init__(self, env: SumoEnv):
        super().__init__(env)
        self.action_space = spaces.MultiBinary([len(env.legal_combinations)])

    def action(self, action: np.ndarray) -> np.ndarray:
        legal_combinations = self.env.legal_combinations
        assert len(action) == len(legal_combinations)
        action = action.astype(bool)
        return action


if __name__ == "__main__":
    env = SumoEnv(intersection_path="intersections")
    env.visualize = True
    # env = BinVehicles(env)
    env = DiscritizeSignal(env)
    # env = DiscretizeLegs(env)
    env = SimpleObs(env)
    obs, _ = env.reset()
    done = False
    while not done:
        action = np.ones(env.action_space.shape)
        obs, reward, done, _, _ = env.step(action)
        # print(obs)
        input()
        print(f"Reward: {reward}")

    env.close()
