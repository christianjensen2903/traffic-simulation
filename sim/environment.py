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
import json
from pydantic import BaseModel


class InternalLeg(BaseModel):
    name: str
    lanes: list[str]
    groups: list[str]
    radar: str
    segments: list[str]


def load_configuration(
    configuration_file: str, start_time: int, test_duration_seconds: int
):
    """Load environment based on the configuration file"""

    model_folder = pathlib.Path(configuration_file).parent

    with open(configuration_file, "r") as cfile:
        configuration = yaml.safe_load(cfile)

    for intersection in configuration["intersections"]:
        connections = []

        legs = [InternalLeg(**leg) for leg in intersection["legs"]]

        for connection in intersection["connections"]:
            connections.append(
                Connection(
                    connection["index"], connection["groups"], connection["priority"]
                )
            )

        allowed_green_signal_combinations = [
            AllowedGreenSignalCombinationDto(
                name=comb["signal"][0], groups=comb["allowed"]
            )
            for comb in intersection["allowed_green_signal_combinations"]
        ]
        env = TrafficSimulationEnvHandler(
            start_time=start_time,
            test_duration_seconds=test_duration_seconds,
            model_folder=model_folder,
            groups=intersection["groups"],
            connections=connections,
            junction=intersection["junction"],
            signal_groups=intersection["groups"],
            legs=legs,
            allowed_green_signal_combinations=allowed_green_signal_combinations,
            amber_time=4,
            red_amber_time=2,
            min_green_time=6,
        )

        return env

    return None


def load_and_run_simulation(
    configuration_file: str,
    start_time: int,
    test_duration_seconds: int,
    random_state: bool,
    input_q: Queue,
    output_q: Queue,
    error_q: Queue,
):
    env = load_configuration(configuration_file, start_time, test_duration_seconds)
    env.set_queues(input_q, output_q, error_q)

    env.set_random_state(random_state)
    env.run_simulation()
    return env.get_observable_state()


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

lock = threading.Lock()


class Connection:
    def __init__(self, number: int, groups: list[str], priority: bool):
        self.number = number
        self.groups = groups
        self.priority = priority


class TrafficSimulationEnvHandler:
    def __init__(
        self,
        start_time: int,
        test_duration_seconds: int,
        model_folder: pathlib.Path,
        groups: list[str],
        connections: list[Connection],
        junction: str,
        signal_groups: list[str],
        legs: list[InternalLeg],
        allowed_green_signal_combinations: list[AllowedGreenSignalCombinationDto],
        amber_time: int,
        red_amber_time: int,
        min_green_time: int,
    ) -> None:

        # Initialize simulation
        self._game_ticks = 0
        self._total_score = 0
        self._start_time = start_time
        self._test_duration_seconds = test_duration_seconds
        self._model_folder = model_folder
        self._random_state = False

        self.maxdistance = 100
        self.groups = groups
        self.connections = connections

        self.junction = junction
        self.signal_groups = signal_groups

        self.legs_dto = [
            LegDto(name=leg.name, lanes=leg.lanes, signal_groups=leg.groups)
            for leg in legs
        ]
        self._internal_legs = legs

        self.allowed_green_signal_combinations_lookup = {}

        for g in allowed_green_signal_combinations:
            self.allowed_green_signal_combinations_lookup[g.name] = g.groups

        self.allowed_green_signal_comb_dto = allowed_green_signal_combinations

        self.amber_time = amber_time
        self.red_amber_time = red_amber_time
        self.min_green_time = min_green_time

        self.group_states: dict[str, tuple[str, int]] = {}

        for group in groups:
            self.group_states[group] = ("red", 0)

        self.next_groups: dict[str, str] = {}

        for group in groups:
            self.next_groups[group] = "red"

        self.vehicle_waiting_time: dict[str, int] = {}
        self.observable_state = TrafficSimulationPredictRequestDto(
            vehicles=[],
            total_score=0,
            simulation_ticks=0,
            signals=[],
            signal_groups=self.signal_groups,
            legs=self.legs_dto,
            allowed_green_signal_combinations=self.allowed_green_signal_comb_dto,
            is_terminated=False,
        )

        self._is_initialized = False
        self._simulation_is_running = False
        self.simulation_ticks = 0

        self.delay_penalty_coefficient = 1.5
        self.delay_penalty_start_seconds = 90
        self.warm_up_ticks = 10

        self.errors = []

        self._traci_connection = None

        self._input_queue = None
        self._output_queue = None
        self._error_queue = None

    def set_queues(self, input_q, output_q, error_q):
        self._input_queue = input_q
        self._output_queue = output_q
        self._error_queue = error_q

    def distance_to_stop(self, vehicle):
        for intersection, _, distance, _ in self._traci_connection.vehicle.getNextTLS(
            vehicle
        ):
            if intersection == self.junction:
                return distance
        return None

    def set_random_state(self, random):
        self._random_state = random

    def _calculate_score(self):
        score = 0.0

        for vehicle, waiting_time in self.vehicle_waiting_time.items():
            score += waiting_time

            if waiting_time > self.delay_penalty_start_seconds:
                score += (
                    self.vehicle_waiting_time[vehicle]
                    - self.delay_penalty_start_seconds
                ) ** self.delay_penalty_coefficient

        return score

    def get_simulation_is_running(self):
        if self._is_initialized == False:
            return True
        else:
            return self._simulation_is_running

    def get_simulation_ticks(self):
        return self.simulation_ticks

    def get_observable_state(self):
        return self.observable_state

    def _validate_next_signals(self, next_groups):
        logic_errors = []
        desired_next_groups = {}  # desired color for each group

        # Set default desired color to "red" for all groups
        for group in self.groups:
            desired_next_groups[group] = "red"

        # Update desired colors with next_groups
        for group, color in next_groups.items():
            if group not in self.groups:
                logic_errors.append(
                    f"Invalid signal group {group} at time step {self.get_simulation_ticks()}"
                )
                continue
            desired_next_groups[group] = color.lower()

        # Collect green lights for conflict checking
        green_lights = [
            group for group, color in desired_next_groups.items() if color == "green"
        ]

        # Check for conflicts
        for group in green_lights:
            for other_group in green_lights:
                if group == other_group:
                    continue
                if (
                    other_group
                    not in self.allowed_green_signal_combinations_lookup.get(group, [])
                ):
                    logic_errors.append(
                        f"Invalid green light combination at time step {self.get_simulation_ticks()}: {group} and {other_group}."
                    )
                    # Optionally, you can remove one of the conflicting groups
                    # desired_next_groups[group] = 'red'
                    # break

        # Update self.next_groups with desired colors
        for group, color in desired_next_groups.items():
            self.next_groups[group] = color

        if len(logic_errors) == 0:
            return None

        logger.info(f"logic_errors: {logic_errors}")
        return ";".join(logic_errors)

    def set_next_signals(self, next_groups):

        errors = self._validate_next_signals(next_groups)

        return errors

    def _update_group_states(self):
        for group in self.groups:
            desired_color = self.next_groups[group]
            current_color, time_in_current_color = self.group_states[group]

            if current_color == "red":
                if desired_color == "green":
                    # Start transition to green: red -> redamber
                    self.group_states[group] = ("redamber", 1)
                else:
                    # Stay in red
                    self.group_states[group] = ("red", time_in_current_color + 1)
            elif current_color == "redamber":
                if time_in_current_color >= self.red_amber_time:
                    # Transition to green
                    self.group_states[group] = ("green", 1)
                else:
                    # Continue in redamber
                    self.group_states[group] = ("redamber", time_in_current_color + 1)
            elif current_color == "green":
                if desired_color == "red":
                    if time_in_current_color >= self.min_green_time:
                        # Start transition to red: green -> amber
                        self.group_states[group] = ("amber", 1)
                    else:
                        # Stay in green
                        self.group_states[group] = ("green", time_in_current_color + 1)
                else:
                    # Stay in green
                    self.group_states[group] = ("green", time_in_current_color + 1)
            elif current_color == "amber":
                if time_in_current_color >= self.amber_time:
                    # Transition to red
                    self.group_states[group] = ("red", 1)
                else:
                    # Continue in amber
                    self.group_states[group] = ("amber", time_in_current_color + 1)
            else:
                # Should not reach here
                raise Exception(
                    f"Invalid state {current_color} at tick {self.simulation_ticks} for group {group}"
                )

    def _color_to_letter(self, color):
        if color == "red":
            return "r"
        elif color == "amber" or color == "redamber":
            return "y"
        elif color == "green":
            return "g"
        else:
            raise ValueError(f"Got unknown color {color}")

    def _get_phase_string(self):
        res = ""
        for connection in self.connections:
            to_set = "r"
            for g in connection.groups:
                (color, _time) = self.group_states[g]
                color = self._color_to_letter(color)
                if to_set == "r":
                    to_set = color
                elif to_set == "y" and color == "g":
                    to_set == "g"
                elif to_set == "g" or to_set == "y":
                    pass
                else:
                    raise ValueError("Invalid state reached in get_phase_string")
            if to_set == "g" and connection.priority:
                to_set = "G"
            res += to_set
        assert len(res) == len(self.connections)

        return res

    def _set_signal_state(self):
        phase_string = self._get_phase_string()

        self._traci_connection.trafficlight.setRedYellowGreenState(
            self.junction, phase_string
        )

    def _update_vehicles(self):

        observed_vehicles = []

        for leg in self._internal_legs:
            segments = leg.segments

            for segment in segments:
                vehicles = list(traci.edge.getLastStepVehicleIDs(segment))
                for vehicle in vehicles:
                    distance = self.distance_to_stop(vehicle)
                    if distance == None or distance > self.maxdistance:
                        continue
                    vehicle_speed = abs(traci.vehicle.getSpeed(vehicle))

                    observed_vehicles.append(
                        VehicleDto(
                            speed=round(vehicle_speed, 1),
                            distance_to_stop=round(distance, 1),
                            leg=leg.name,
                        )
                    )

                    if vehicle_speed < 0.5:  # Vehicle travels at less than 1.8 km/h
                        if vehicle not in self.vehicle_waiting_time:
                            self.vehicle_waiting_time[vehicle] = 0
                        self.vehicle_waiting_time[vehicle] += 1

        return observed_vehicles

    def _run_one_tick(self, terminates_now=False):
        self._traci_connection.simulationStep()
        self.simulation_ticks += 1

        observed_vehicles = []
        signals = []

        # Get the current state of the simulation
        observed_vehicles = self._update_vehicles()

        # Update the score
        self._total_score = self._calculate_score()

        # Get the next phase from the request and set it
        with lock:
            try:
                if self._input_queue and not self._input_queue.empty():
                    next_groups = self._input_queue.get_nowait()

                    signal_logic_errors = self.set_next_signals(next_groups)
                    if self._error_queue and signal_logic_errors:

                        self._error_queue.put(signal_logic_errors)

                self._update_group_states()
            except Exception as e:
                self.errors.append(e)

        self._set_signal_state()

        for group, state in self.group_states.items():
            signals.append(SignalDto(name=group, state=state[0]))

        # Update the observable state
        self.observable_state = TrafficSimulationPredictRequestDto(
            vehicles=observed_vehicles,
            total_score=self._total_score,
            simulation_ticks=self.simulation_ticks,
            signals=signals,
            legs=self.legs_dto,
            signal_groups=self.signal_groups,
            allowed_green_signal_combinations=self.allowed_green_signal_comb_dto,
            is_terminated=terminates_now,
        )

        if self._output_queue:
            self._output_queue.put(self.observable_state)

    def run_simulation(self):

        self._simulation_is_running = True

        logger.info("Traffic simulation - starting sumo....")
        sumoBinary = checkBinary("sumo-gui")

        sim_instance = uuid4().hex

        cmd = [
            sumoBinary,
            "--start",
            "--quit-on-end",
            "-c",
            (self._model_folder / "net.sumocfg").as_posix(),
        ]

        if self._random_state:
            cmd.append("--random")

        traci.start(cmd, label=sim_instance)

        self._traci_connection = traci.getConnection(sim_instance)
        self._is_initialized = True

        self.simulation_ticks = 0
        for i in range(self.warm_up_ticks):
            self._run_one_tick()

        while True:
            logger.info(f"Traffic simulation - tick {self.simulation_ticks}....")

            if self.simulation_ticks < (
                self._test_duration_seconds + self.warm_up_ticks
            ):
                self._run_one_tick()
                sleep(1)
            else:
                self._run_one_tick(terminates_now=True)
                break

        self._traci_connection.close()
        self._simulation_is_running = False
