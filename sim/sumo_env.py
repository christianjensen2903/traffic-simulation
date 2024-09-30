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
from generate_road import LaneType
from generate_intersection_xml import generate_intersection_xml

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

    def __init__(self, config_path: str, max_simulation_time: int = 1000):
        super(SumoEnv, self).__init__()
        self.max_simulation_time = max_simulation_time
        self.path = config_path
        self.load_config(config_path)

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
        # TODO: Add lanes/legs information
        self.observation_space = spaces.Dict(
            {
                "vehicles": self.vehicles_space,
                "signals": self.signals_space,
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

        self.max_distance = 100
        self.min_lanes = 1
        self.max_lanes = 6
        self._random_state = False
        self.amber_time = 4
        self.red_amber_time = 2
        self.min_green_time = 6

        self._delay_penalty = 1.5
        self._delay_penality_threshold = 90
        self._warm_up_ticks = 10

    def _distance(self, vehicle) -> float | None:
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

    def _update_waiting_times(self) -> None:
        """Update the waiting times for all vehicles"""
        for vehicle in self._traci_connection.vehicle.getIDList():
            vehicle_speed = abs(traci.vehicle.getSpeed(vehicle))
            if vehicle_speed < 0.5:  # Vehicle travels at less than 1.8 km/h
                if vehicle not in self._vehicle_waiting_times:
                    self._vehicle_waiting_times[vehicle] = 0
                self._vehicle_waiting_times[vehicle] += 1

    def _get_vehicles(self) -> list[dict]:
        vehicles = {leg.name: [] for leg in self.legs}
        for vehicle in self._traci_connection.vehicle.getIDList():
            distance = self._distance(vehicle)
            if distance is None:
                continue
            speed = traci.vehicle.getSpeed(vehicle)
            leg = traci.vehicle.getRoadID(vehicle)
            vehicles[leg].append(
                {
                    "speed": speed,
                    "distance": distance,
                }
            )
        return vehicles

    def _get_signal_states(self) -> list[dict]:
        return {
            group: {
                "color": state.color.value,
                "time": min(state.time, self.signal_space["time"].high[0]),
            }
            for group, state in self._signal_states.items()
        }

    def _get_observation(self) -> dict:
        """Gets a list of all vehicles and their states"""
        vehicles = self._get_vehicles()
        signals = self._get_signal_states()

        return {
            "vehicles": vehicles,
            "signals": signals,
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
                state.color = TrafficColor.AMBER
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
                state.color = TrafficColor.REDAMBER
                state.time = 0
            else:  # Not ready to shift or not legal change
                state.time += 1

        # Set colors in SUMO
        phase_string = self._get_phase_string()
        self._traci_connection.trafficlight.setRedYellowGreenState(
            self.junction, phase_string
        )

    def _index_to_direction(self, index: int) -> str:
        """Get the direction from the index"""
        return ["N", "E", "S", "W"][index]

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
            direction = self._index_to_direction(direction_index)
            lane_type = self._index_to_lane_type(lane_index)
            group = f"{direction}_{lane_type.value}"
            if group in self.signal_groups:
                action_dict[group] = TrafficColor.GREEN if a else TrafficColor.RED
        return action_dict

    def step(self, action: np.ndarray):
        """
        Action describes if the traffic light should be turned green or not.
        The first dimension describes the direction. 0: N, 1: E, 2: S, 3: W (clockwise)
        The second dimension describes the lane type.
        """
        assert action.shape == self.action_space.shape
        assert self._traci_connection is not None

        parsed_action = self._parse_action(action)
        self._update_traffic_lights(parsed_action)

        self._traci_connection.simulationStep()
        self._update_waiting_times()
        self._ticks += 1
        observation = self._get_observation()
        reward = -self._calc_loss()
        done = self._ticks >= self.max_simulation_time

        return observation, reward, done, None, {}  # No extra info or truncated

    def reset(self, seed=None, options=None):
        generate_intersection_xml(
            path=self.path, min_lanes=self.min_lanes, max_lanes=self.max_lanes
        )
        self.load_config(self.path)

        sumoBinary = checkBinary("sumo-gui")

        sim_instance = uuid4().hex

        cmd = [
            sumoBinary,
            "--start",
            "--quit-on-end",
            "-c",
            f"{self.path}/net.sumocfg",
        ]

        if self._random_state:
            cmd.append("--random")

        traci.start(cmd, label=sim_instance)

        self._traci_connection = traci.getConnection(sim_instance)

        self._ticks = 0
        self._vehicle_waiting_times = {}
        for i in range(self._warm_up_ticks):
            self._traci_connection.simulationStep()

        return self._get_observation(), None  # No extra info

    def render(self, mode="human"):
        pass

    def close(self):
        if self._traci_connection:
            self._traci_connection.close()
            self._traci_connection = None


if __name__ == "__main__":
    env = SumoEnv(config_path="intersections/2")
    env.reset()
    done = False
    while not done:
        action = [True] * len(env.signal_groups)
        obs, reward, done, _, _ = env.step(action)
        print(f"Reward: {reward}")
    env.close()
