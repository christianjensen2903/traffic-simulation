import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from sumo_env import SumoEnv, TrafficColor, InternalLeg
from generate_road import (
    LaneType,
    RoadDirection,
    index_to_direction,
    direction_to_index,
)
import numpy as np
import time
from lane_tracker import LaneTracker, TrackedVehicle


class DiscritizeSignal(gym.ObservationWrapper):
    """A wrapper that discretizes the signal observations"""

    def __init__(self, env: SumoEnv):
        super().__init__(env)

        total_lanes = sum(len(leg.lanes) for leg in env.legs)

        self.signals_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                total_lanes,
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
        legs = obs["legs"]
        assert isinstance(signals, dict)
        assert isinstance(legs, dict)

        lights = []

        for leg_name, lanes in legs.items():
            for lane in lanes:
                signal_name = f"{leg_name[0]}_{lane}"

                # Get the signal for the lane

                for group, signal in signals.items():
                    if group == signal_name:
                        color = TrafficColor(signal["color"])
                        time = signal["time"] + 1
                        if color == TrafficColor.RED:
                            normalized_color = 1
                        elif color == TrafficColor.REDAMBER:
                            normalized_color = min(1, time / 3)
                        elif color == TrafficColor.AMBER:
                            normalized_color = min(1, time / 5)
                        elif color == TrafficColor.GREEN:
                            normalized_color = min(1, time / 7)

                        # One hot encode the color
                        color_index = self._get_color_index(color)
                        one_hot_color = np.zeros(len(TrafficColor))
                        one_hot_color[color_index] = normalized_color
                        lights.append(one_hot_color)

        # Concatenate the signals
        signal_obs = np.array(lights)

        obs["signals"] = signal_obs
        return obs


class DiscretizeLegs(gym.ObservationWrapper):
    """A wrapper that discretizes the legs observations"""

    def __init__(self, env: SumoEnv):
        super().__init__(env)

        total_lanes = sum(len(leg.lanes) for leg in env.legs)

        self.legs_space = spaces.MultiBinary(
            [
                total_lanes,
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

    def _get_lane_index(self, lane: str) -> int:
        """Get the index of the lane in the lane space"""
        return list(LaneType).index(LaneType(lane))

    def observation(self, observation: dict) -> dict:
        legs = observation["legs"]
        assert isinstance(legs, dict)

        all_lanes = []
        for lanes in legs.values():
            for lane in lanes:

                # One hot encode the lane
                lane_enc = np.zeros(len(LaneType))
                lane_index = self._get_lane_index(lane)
                lane_enc[lane_index] = 1

                all_lanes.append(lane_enc)

        leg_obs = np.array(all_lanes)

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


class TrackLanes(gym.ObservationWrapper):
    def __init__(self, env: SumoEnv, intersection: str):
        super().__init__(env)
        self.lane_tracker = LaneTracker(env, intersection)

        total_lanes = sum(len(leg.lanes) for leg in env.legs)

        self.vehicles_space = spaces.Box(
            low=0,
            high=200,  # It don't think there would be more than 200 vehicles in a bin
            shape=(
                total_lanes,
                15,  # Max 15 cars
                4,  # 4 scalar features
            ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "signals": env.signals_space,
                "vehicles": env.vehicles_space,
                "legs": env.legs_space,
            }
        )

    def get_lane_indices(
        self, lanes: list[LaneType], possible_lanes: list[LaneType]
    ) -> list[int]:
        """Get the possible indices of the lanes of the given type"""
        return [i for i, lane in enumerate(lanes) if lane in possible_lanes]

    def sort_cars(
        self, vehicles: list[TrackedVehicle], lanes: list[LaneType]
    ) -> np.ndarray:
        """Sort the vehicles by their distance"""
        sorted_cars: list[list[TrackedVehicle]] = [[] for _ in lanes]
        lane_counter: list[int] = [0] * len(
            lanes
        )  # Index for how many cars have been added
        for vehicle in vehicles:
            indices = self.get_lane_indices(lanes, vehicle.possible_lanes)
            # Add to lane with least cars
            lane_index = min(indices, key=lambda i: lane_counter[i])
            sorted_cars[lane_index].append(vehicle)
            lane_counter[lane_index] += 1

        # Sort cars in each lane by distance
        for lane in sorted_cars:
            lane.sort(key=lambda car: car.distance)

        # Convert to numpy array
        road = np.zeros((len(lanes), 15, 4))
        for i, lane in enumerate(sorted_cars):
            for j, car in enumerate(lane):

                normalized_distance = car.distance / 100
                normalized_speed = car.speed / 27.78
                normalized_waiting_time = min(1, car.waiting_time / 90)

                features = [
                    1,
                    normalized_distance,
                    normalized_speed,
                    normalized_waiting_time,
                ]
                road[i][j] = features

        return road

    def observation(self, obs: dict) -> dict:
        vehicles = obs["vehicles"]
        assert isinstance(vehicles, dict)

        roads = []

        for leg_name, vehicles in vehicles.items():
            leg = self.lane_tracker.get_leg(leg_name)
            tracked_vehicles = self.lane_tracker.update_vehicles_for_leg(leg, vehicles)
            road = self.sort_cars(tracked_vehicles, leg.lanes)
            roads.append(road)

        obs["vehicles"] = np.concatenate(roads, axis=0)

        return obs


if __name__ == "__main__":
    env = SumoEnv(intersection_path="intersections")
    env.visualize = True

    env = DiscritizeSignal(env)
    env = TrackLanes(env, "intersection_4")
    env = DiscretizeLegs(env)
    # env = SimpleObs(env)

    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = np.ones(env.action_space.shape)
        obs, reward, done, _, _ = env.step(action)

        # print(obs)
        for key, value in obs.items():
            print(key)
            print(value.shape)
            print()
        input()
        steps += 1
        print(f"Reward: {reward}")

    env.close()
