from typing import Literal
from pydantic import BaseModel
import numpy as np
from generate_road import LaneType, index_to_direction
from sumo_env import TrafficColor, SignalState, RoadDirection
from sim.dtos import VehicleDto, SignalDto, LegDto, TrafficSimulationPredictRequestDto


class RequestConverter:

    def __init__(self, legs: dict[str, list[LaneType]]):
        self.amber_time = 4
        self.red_amber_time = 2
        self.min_green_time = 6
        self.signal_states: dict[str, SignalState] = {}
        self.legs = legs

    def convert_leg_naming(self, leg_name: str) -> str:
        if leg_name == "A1":
            return "S"
        elif leg_name == "A2":
            return "N"
        elif leg_name == "B2":
            return "W"
        elif leg_name == "B1":
            return "E"

    def convert_vehicles(self, vehicles: list[VehicleDto]) -> dict[str, list[dict]]:
        converted_vehicles: dict[str, list[dict]] = {n: [] for n in ["N", "S", "W", "E"]}
        for vehicle in vehicles:
            leg_name = self.convert_leg_naming(vehicle.leg)
            if leg_name not in converted_vehicles:
                converted_vehicles[leg_name] = []

            converted_vehicles[leg_name].append(
                {"speed": vehicle.speed, "distance_to_stop": vehicle.distance_to_stop}
            )

        return converted_vehicles

    def convert_color(self, color: str) -> TrafficColor:
        if color == "red":
            return TrafficColor.RED
        elif color == "amber":
            return TrafficColor.AMBER
        elif color == "redamber":
            return TrafficColor.REDAMBER
        elif color == "green":
            return TrafficColor

    def initialize_signals(self, legs: dict[str, list[LaneType]]) -> None:
        for leg_name, lanes in legs.items():
            for lane in lanes:
                name = self.convert_leg_naming(leg_name)
                group_name = f"{name}_1_{lane}"
                self.signal_states[group_name] = SignalState()

    def convert_legs(self, leg: LegDto) -> dict[str, list[LaneType]]:
        lanes: dict[str, list[LaneType]] = {}
        leg_name = self.convert_leg_naming(leg.name)
        for lane in leg.lanes:
            if leg_name not in lanes:
                lanes[leg_name] = []
            if lane == "Left":
                lanes[leg_name].append(LaneType.LEFT)
            elif lane == "Right":
                lanes[leg_name].append(LaneType.RIGHT)
            elif lane == "Main":
                if leg.name in leg.signal_groups:
                    lanes[leg_name].append(LaneType.STRAIGHT)
                else:  # A bit hacky but works for this usecase
                    lanes[leg_name].append(LaneType.STRAIGHT_LEFT)

        return lanes

    def convert_signals(self, signals: list[SignalDto]) -> dict[str, str]:
        converted_signals: dict[str, str] = {}
        for signal in signals:
            converted_signals[signal.name] = signal.state

        return converted_signals

    def get_signals(self) -> dict[str, dict]:
        signals: dict[str, list[str]] = {}
        for group_name, signal in self.signal_states.items():
            signals[group_name] = {
                "color": signal.color.value,
                "time": min(signal.time, 6),
            }

        return signals

    def _update_traffic_lights(
        self, action: dict[str, Literal[TrafficColor.GREEN, TrafficColor.RED]]
    ):
        """Update the traffic lights based on the action"""
        for group, state in self.signal_states.items():
            desired_color = action.get(group, TrafficColor.RED)

            if (
                state.color == TrafficColor.GREEN
                and desired_color == TrafficColor.RED
                and state.time >= self.min_green_time
            ):
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
                state.color == TrafficColor.RED and desired_color == TrafficColor.GREEN
            ):
                state.color = TrafficColor.REDAMBER
                state.time = 0
            else:  # Not ready to shift or not legal change
                state.time += 1

    def _parse_action(
        self, action: np.ndarray
    ) -> dict[str, Literal[TrafficColor.GREEN, TrafficColor.RED]]:
        """Parse the action into a dictionary"""
        action_dict = {}

        action = action.reshape((4,6))
        action = action.flatten()


        for i, a in enumerate(action):

            direction_index = i // len(LaneType)
            lane_index = i % len(LaneType)
            direction = index_to_direction(direction_index)
            lane_type = list(LaneType)[lane_index]
            if lane_type not in self.legs[direction.value]:
                continue
            group = f"{direction.value}_{lane_type.value}"
            action_dict[group] = TrafficColor.GREEN if a else TrafficColor.RED

        # print(action_dict)
        return action_dict

    #def update_signals(self, action: np.ndarray) -> None:
    #    parsed_action = self._parse_action(action)
    #    self._update_traffic_lights(parsed_action)

    def reset(self, legs: dict[str, list[LaneType]]) -> None:
        self.signal_states = {}
        self.legs = legs

    def convert_request(self, request: TrafficSimulationPredictRequestDto) -> dict:
        vehicles = self.convert_vehicles(request.vehicles)
        legs = {leg.name: self.convert_legs(leg) for leg in request.legs}

        if self.signal_states == {}:
            self.initialize_signals(legs)

        #signals = self.convert_signals(request.signals)

        return {
            "vehicles": vehicles,
            #"signals": signals,
            "legs": legs,
        }

    def convert_signal_back(self, signal: str) -> str:
        direction = RoadDirection(signal.split("_")[0])
        lane_type = LaneType(signal.split("_")[1])

        if direction == RoadDirection.N:
            direction_string = "A2"
        elif direction == RoadDirection.S:
            direction_string = "A1"
        elif direction == RoadDirection.W:
            direction_string = "B2"
        elif direction == RoadDirection.E:
            direction_string = "B1"

        if lane_type == LaneType.LEFT:
            lane_string = "LeftTurn"
        elif lane_type == LaneType.RIGHT:
            lane_string = "RightTurn"
        else:
            lane_string = ""

        return f"{direction_string}{lane_string}"

    def convert_color_back(self, color: TrafficColor) -> str:
        if color == TrafficColor.RED:
            return "red"
        elif color == TrafficColor.AMBER:
            return "amber"
        elif color == TrafficColor.REDAMBER:
            return "redamber"
        elif color == TrafficColor.GREEN:
            return "green"

    def convert_action(self, action: np.ndarray) -> dict:
        new_action = []
        parsed_action = self._parse_action(action)
        for group, color in parsed_action.items():
            new_action.append(
                SignalDto(
                    name=self.convert_signal_back(group),
                    state=self.convert_color_back(color),
                )
            )

        return new_action
