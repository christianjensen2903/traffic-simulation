from sumolib import checkBinary  # noqa
import traci  # noqa
from uuid import uuid4
import os
import sys
import yaml
import gymnasium as gym
from gymnasium import spaces
from pydantic import BaseModel
from enum import Enum
from typing import Literal
import numpy as np
from generate_road import (
    LaneType,
    Road,
    RoadDirection,
    index_to_direction,
    direction_to_index,
)
from dtos import AllowedGreenSignalCombinationDto
from generate_random_flow import generate_random_flow
import random


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)


class Connection(BaseModel):
    index: int
    groups: list[str]
    priority: bool


class TrafficColor(Enum):
    RED = "r"
    AMBER = "y"
    REDAMBER = "u"
    GREEN = "G"


class InternalLeg(BaseModel):
    name: str
    lanes: list[str]
    groups: list[str]
    segments: list[str]


class SignalState(BaseModel):
    color: TrafficColor = TrafficColor.RED
    time: int = 0


class SumoEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, intersection_path: str, max_simulation_time: int = 1000):
        super(SumoEnv, self).__init__()
        self.max_simulation_time = max_simulation_time
        self.intersection_path = intersection_path
        self.load_config(f"{intersection_path}/2")
        self.visualize = False

        # Max 4 incoming roads. Has to be flattened
        self.action_space = spaces.MultiBinary(4 * len(LaneType))

        self.vehicle_space = spaces.Dict(
            {
                "speed": spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32),
                "distance": spaces.Box(
                    low=0, high=self.max_distance, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.vehicles_space = spaces.Dict(
            {leg.name: self.vehicle_space for leg in self.legs}
        )

        self.signal_space = spaces.Dict(
            {
                "color": spaces.Text(max_length=1),
                "time": spaces.Box(
                    low=0,
                    high=max(
                        [self.amber_time, self.red_amber_time, self.min_green_time]
                    ),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
        self.signals_space = spaces.Dict(
            {group: self.signal_space for group in self.signal_groups}
        )
        self.leg_space = spaces.MultiBinary(len(LaneType))
        self.legs_space = spaces.Dict(
            {leg.name: spaces.Sequence(self.leg_space) for leg in self.legs}
        )

        self.observation_space = spaces.Dict(
            {
                "vehicles": self.vehicles_space,
                "signals": self.signals_space,
                "legs": self.legs_space,
            }
        )

        self._ticks = 0
        self._vehicle_waiting_times = {}
        self._traci_connection = None

    def load_config(self, path: str):
        with open(f"{path}/configuration.yaml", "r") as cfile:
            config = yaml.safe_load(cfile)["intersections"][0]

        self._connections = [
            Connection(
                index=conn["index"],
                groups=conn["groups"],
                priority=conn["priority"],
            )
            for conn in config["connections"]
        ]
        self.legs = [InternalLeg(**leg) for leg in config["legs"]]
        self.signal_groups: list[str] = config["groups"]
        self._signal_states = {group: SignalState() for group in self.signal_groups}
        self.junction = config["junction"]
        # allowed_green_signal_combinations = [
        #     AllowedGreenSignalCombinationDto(
        #         name=comb["signal"][0], groups=comb["allowed"]
        #     )
        #     for comb in intersection["allowed_green_signal_combinations"]
        # ]

        self.max_distance = 100
        self.random_state = True
        self.amber_time = 4
        self.red_amber_time = 2
        self.min_green_time = 6
        self.roads = [
            Road(
                direction=RoadDirection(f'{leg.name.split("_")[0]}'),
                lanes=[LaneType(lane) for lane in leg.lanes],
            )
            for leg in self.legs
        ]

        self._delay_penalty = 1.5
        self._delay_penality_threshold = 90
        self._warm_up_ticks = 10

    def _get_distance_to_stop(self, vehicle) -> float | None:
        for intersection, _, distance, _ in self._traci_connection.vehicle.getNextTLS(
            vehicle
        ):
            if intersection == self.junction:
                return distance
        return None

    def _calc_loss(self) -> float:
        losses = [
            w_time
            + max(0, w_time - self._delay_penality_threshold) ** self._delay_penalty
            for w_time in self._vehicle_waiting_times.values()
        ]
        return sum(losses)

    def _get_and_update_vehicles(self) -> list[dict]:
        """Get all vehicles and update the waiting times"""

        # Not two seperate functions for efficiency

        observed_vehicles = {}
        for leg in self.legs:
            observed_vehicles[leg.name] = []

            for segment in leg.segments:
                vehicles = list(traci.edge.getLastStepVehicleIDs(segment))
                for vehicle in vehicles:
                    distance = self._get_distance_to_stop(vehicle)
                    if distance is None or distance > self.max_distance:
                        continue
                    vehicle_speed = abs(traci.vehicle.getSpeed(vehicle))
                    observed_vehicles[leg.name].append(
                        {
                            "speed": vehicle_speed,
                            "distance": distance,
                        }
                    )

                    # self._vehicle_waiting_times[vehicle] = (
                    #     traci.vehicle.getAccumulatedWaitingTime(vehicle)
                    # )  # Not exactly same definition but allows for longer timesteps
                    if vehicle_speed < 0.5:  # Vehicle travels at less than 1.8 km/h
                        if vehicle not in self._vehicle_waiting_times:
                            self._vehicle_waiting_times[vehicle] = 0
                        self._vehicle_waiting_times[vehicle] += 1

        return observed_vehicles

    def _get_signal_states(self) -> list[dict]:
        return {
            group: {
                "color": state.color.value,
                "time": min(state.time, self.signal_space["time"].high[0]),
            }
            for group, state in self._signal_states.items()
        }

    def _get_leg_info(self) -> list[dict]:
        info = {}
        for leg in self.legs:
            lane_info = []
            for lane in leg.lanes:
                enc = np.zeros(self.leg_space.shape)
                lane_type = LaneType(lane)
                enc[list(LaneType).index(lane_type)] = 1
                lane_info.append(enc)
            info[leg.name] = lane_info
        return info

    def _get_observation(self) -> dict:
        """Gets a list of all vehicles and their states"""
        vehicles = self._get_and_update_vehicles()
        signals = self._get_signal_states()
        legs = self._get_leg_info()

        return {
            "vehicles": vehicles,
            "signals": signals,
            "legs": legs,
        }

    def _get_phase_string(self) -> str:
        phase_string = ""
        for conn in self._connections:
            phase_string += self._signal_states[conn.groups[0]].color.value
        return phase_string

    def _update_traffic_lights(
        self, action: dict[str, Literal[TrafficColor.GREEN, TrafficColor.RED]]
    ):
        """Update the traffic lights based on the action"""
        for group, state in self._signal_states.items():
            desired_color = action.get(group, TrafficColor.RED)

            if state.color == TrafficColor.RED and desired_color == TrafficColor.GREEN:
                state.color = TrafficColor.REDAMBER
                state.time = 0
            elif state.color == TrafficColor.AMBER and state.time >= self.amber_time:
                state.color = TrafficColor.RED
                state.time = 0
            elif (
                state.color == TrafficColor.REDAMBER
                and state.time >= self.red_amber_time
            ):
                state.color = TrafficColor.GREEN
                state.time = 0
            elif (
                state.color == TrafficColor.GREEN
                and desired_color == TrafficColor.RED
                and state.time >= self.min_green_time
            ):
                state.color = TrafficColor.AMBER
                state.time = 0
            else:  # Not ready to shift or not legal change
                state.time += 1

        # Set colors in SUMO
        phase_string = self._get_phase_string()
        self._traci_connection.trafficlight.setRedYellowGreenState(
            self.junction, phase_string
        )

    def _index_to_lane_type(self, index: int) -> LaneType:
        """Get the lane type from the index"""
        return list(LaneType)[index]

    def _parse_action(
        self, action: np.ndarray
    ) -> dict[str, Literal[TrafficColor.GREEN, TrafficColor.RED]]:
        """Parse the action into a dictionary"""
        action_dict = {}
        for i, a in enumerate(action):

            direction_index = i // len(LaneType)
            lane_index = i % len(LaneType)
            direction = index_to_direction(direction_index)
            lane_type = self._index_to_lane_type(lane_index)
            group = f"{direction.value}_{lane_type.value}"
            if group in self.signal_groups:
                action_dict[group] = TrafficColor.GREEN if a else TrafficColor.RED
        return action_dict

    def action_masks(self) -> np.ndarray:
        """
        Get the action masks for the environment.
        """

        mask = np.zeros(self.action_space.shape, dtype=np.float32)
        for i in range(self.action_space.shape[0]):
            for j in range(self.action_space.shape[1]):
                direction = index_to_direction(i)
                lane_type = self._index_to_lane_type(j)
                group = f"{direction}_{lane_type.value}"

                # Make 1 if the group is red
                # Make 1 if green and above min green time
                state = self._signal_states[group]
                if state.color == TrafficColor.RED:
                    mask[i, j] = 1
                elif (
                    state.color == TrafficColor.GREEN
                    and state.time >= self.min_green_time
                ):
                    mask[i, j] = 1

        return mask

    def step(self, action: np.ndarray):
        """
        Action describes if the traffic light should be turned green or not.
        The first dimension describes the direction. 0: N, 1: E, 2: S, 3: W (clockwise)
        The second dimension describes the lane type.
        """
        assert action.shape == self.action_space.shape
        assert self._traci_connection is not None

        loss_before = self._calc_loss()

        parsed_action = self._parse_action(action)
        self._update_traffic_lights(parsed_action)

        self._traci_connection.simulationStep()
        self._ticks += 1
        observation = self._get_observation()
        loss_after = self._calc_loss()
        reward = loss_before - loss_after  # - delta_loss

        done = self._ticks >= self.max_simulation_time

        return observation, reward, done, False, {}  # No extra info or truncated

    def reset(self, seed=None, options=None):

        if self.random_state:
            # choice = random.choice([3, 4])  # Either of the two val intersections
            choice = 4
            path = f"intersections/{choice}"
            generate_random_flow(self.intersection_path, roads=self.roads)
        else:
            path = f"intersections/1"

        self.load_config(path)

        sumoBinary = checkBinary("sumo-gui" if self.visualize else "sumo")

        sim_instance = uuid4().hex

        cmd = [
            sumoBinary,
            "--start",
            "--quit-on-end",
            "-c",
            f"{path}/net.sumocfg",
            "--no-warnings",
            # "--time-to-teleport",  # Disable teleporting
            # "-1",
            "--step-length",
            "1",
        ]

        if self.random_state:
            cmd.append("--random")

        traci.start(cmd, label=sim_instance)

        self._traci_connection = traci.getConnection(sim_instance)

        self._ticks = 0
        self._vehicle_waiting_times = {}
        for i in range(self._warm_up_ticks):
            self._traci_connection.simulationStep()

        return self._get_observation(), {}  # No extra info

    def toggle_visualize(self):
        self.visualize = not self.visualize

    def render(self, mode="human"):
        self.visualize = True

    def close(self):
        if self._traci_connection:
            self._traci_connection.close()
            self._traci_connection = None


if __name__ == "__main__":
    env = SumoEnv(intersection_path="intersections")

    env.visualize = True
    env.reset()
    done = False
    while not done:
        # action = np.random.randint(0, 2, size=env.action_space.shape)
        action = np.zeros(env.action_space.shape)
        obs, reward, done, _, _ = env.step(action)
        print(reward)
        # press any key to continue
        # input()
        print(f"Reward: {reward}")
    env.close()
