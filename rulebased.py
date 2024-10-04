from collections import defaultdict
from enum import Enum
import sumo_env
import numpy as np
import lane_tracker
import sys
import numpy as np
import time as timelib
from lane_tracker import LaneTracker

# turn on
NORTH = 0
EAST_ = 1
SOUTH = 2
WEST_ = 3

LEGS = [NORTH, EAST_, SOUTH, WEST_]

NESW_MAP = {
    NORTH: "N_1",
    EAST_: "E_1",
    SOUTH: "S_1",
    WEST_: "W_1",
}

NESW_MAP_REVERSE = {
    "N_1": NORTH,
    "E_1": EAST_,
    "S_1": SOUTH,
    "W_1": WEST_,
}


class LaneType(Enum):
    ALL = "ALL"
    STRAIGHT = "STRAIGHT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT_RIGHT = "STRAIGHT_RIGHT"
    STRAIGHT_LEFT = "STRAIGHT_LEFT"


int_to_lanetype = {
    0: LaneType.ALL,
    1: LaneType.STRAIGHT,
    2: LaneType.LEFT,
    3: LaneType.RIGHT,
    4: LaneType.STRAIGHT_RIGHT,
    5: LaneType.STRAIGHT_LEFT,
}

ALL = 0
STRAIGHT = 1
LEFT = 2
RIGHT = 3
STRAIGHT_RIGHT = 4
STRAIGHT_LEFT = 5

N_LEG_LANES = {NORTH: 3, EAST_: 2, SOUTH: 3, WEST_: 1}

VALID_LANES = [
    (NORTH, RIGHT),
    (NORTH, STRAIGHT),
    (NORTH, LEFT),
    (SOUTH, RIGHT),
    (SOUTH, STRAIGHT),
    (SOUTH, LEFT),
    (EAST_, RIGHT),
    (EAST_, STRAIGHT_LEFT),
    (WEST_, ALL),
]


CAR_EXIT_RATE = 6.0 / 14.0


class LearningPhase(Enum):
    LEG_RATE_ESTIMATION = 0
    """turn all lights red and count incoming cars. This gives a rate for each leg."""
    LANE_RATE_ESTIMATION = 1
    """turn on only a single lane in each leg. Use previous lane calculation to estimate rates of specific lanes."""
    EXPLOITATION = 2
    """Using these estimates per lane, devise a strategy to maximize throughput."""


LANE_RATE_ESTIMATION_CYCLE = [
    [(NORTH, STRAIGHT), (SOUTH, STRAIGHT)],
    [
        (NORTH, RIGHT),
        (SOUTH, RIGHT),
        (EAST_, STRAIGHT_LEFT),
    ],
]
"""The combinations of lanes to turn on during the lane car per second rate estimation phase"""

LEG_CPS_PHASE_TICKS = 0
"""The amount of ticks to wait at the beginning to estimate traffic on each leg"""
LANE_CPS_PHASE_CYCLE_TIME = 15
"""Amount of ticks in each cycle during the lane car per second rate estimation phase"""
LANE_CPS_PHASE_TICKS = len(LANE_RATE_ESTIMATION_CYCLE) * LANE_CPS_PHASE_CYCLE_TIME
"""The amount of ticks to wait at the beginning to estimate traffic on each lane"""


CYCLE_TIME = 10
"""Amount of ticks before switching in the exploitation phase"""
EXPLOIT_CYCLE_COMPLEX = [
    [
        (NORTH, STRAIGHT),
        (NORTH, RIGHT),
        (SOUTH, STRAIGHT),
        (SOUTH, RIGHT),
        (EAST_, RIGHT),
    ],
    [(NORTH, LEFT), (SOUTH, LEFT), (EAST_, RIGHT), (NORTH, RIGHT), (SOUTH, RIGHT)],
    [(EAST_, STRAIGHT_LEFT), (EAST_, RIGHT), (SOUTH, RIGHT), (NORTH, RIGHT)],
    [(WEST_, ALL), (NORTH, RIGHT), (SOUTH, RIGHT), (EAST_, RIGHT), (NORTH, RIGHT)],
    [(SOUTH, RIGHT), (NORTH, RIGHT), (EAST_, RIGHT)],
    [(NORTH, LEFT), (SOUTH, LEFT), (EAST_, RIGHT), (NORTH, RIGHT), (SOUTH, RIGHT)],
]

EXPLOIT_CYCLE_SIMPLE_AF = [
    [(NORTH, LEFT)],
    [(NORTH, STRAIGHT)],
    [(NORTH, RIGHT)],
    [(SOUTH, LEFT)],
    [(SOUTH, STRAIGHT)],
    [(EAST_, STRAIGHT_LEFT)],
    [(EAST_, RIGHT)],
    [(WEST_, ALL)],
]
EXPLOIT_CYCLE = EXPLOIT_CYCLE_COMPLEX
WARMUP_TICKS = 10


class RulebasedModel:

    def __init__(self, tracker: LaneTracker, legs: dict[str, list[LaneType]]):
        self.reset(tracker, legs)

    def reset(self, tracker: LaneTracker, legs: dict[str, list[LaneType]]):
        self.tracker = tracker
        self.legs = legs
        self.time = 1
        self.count_in_leg = defaultdict(int)
        self.leg_entry_rates = defaultdict(float)
        """The amount of cars entering each leg per tick (=per second)"""
        self.max_dist_per_leg = defaultdict(lambda: 100000)
        self.cars_entered = defaultdict(int)
        self.new_action = np.zeros((4, 6))
        self.action = np.zeros((4, 6))
        self.estimated_in_lane = defaultdict(float)
        self.cost_minimizing_choice = []

    def get_action(self, obs: dict):

        ###### Time starts at 1. Set all the lights to the first combination.
        if self.time == 1:
            for leg, light in EXPLOIT_CYCLE[0]:
                self.action[leg, light] = 1
                self.new_action[leg, light] = 1

        ###### At time 2, we know how many cars are in each leg (count_in_leg). We can now keep track of "cars entered", which is used to estimate the rate of cars entering each leg.
        if self.time == 2:
            for leg in LEGS:
                self.cars_entered[leg] += self.count_in_leg[NESW_MAP[leg]]

        #### Counts the cars that enter each leg
        if (
            obs
        ):  # if the observation is not None - it will be None in the very first tick

            for leg, light in cost_minimizing_choice:
                old_guess = self.estimated_in_lane[(leg, light)]
                self.estimated_in_lane[(leg, light)] = max(old_guess - 6.0 / 14.0, 0)

            for leg in list(self.legs.keys()):
                self.count_in_leg[leg] = len(obs["vehicles"][leg])

            # now, the estimated_in_lane counts must sum to the amount in the entire leg
            # distribute the difference onto all other
            estimated_count_in_leg = defaultdict(float)
            for leg, light in VALID_LANES:
                estimated_count_in_leg[leg] += self.estimated_in_lane[(leg, light)]

            for leg in LEGS:
                car_increment = estimated_count_in_leg[leg] - self.count_in_leg[leg]
                for leg2, light in VALID_LANES:
                    if leg != leg2:  # we don't need to distribute to other legs
                        continue
                    if (
                        leg2,
                        light,
                    ) in cost_minimizing_choice:  # we don't need to distribute to lanes where we trust the count
                        continue
                    if N_LEG_LANES[leg] == 1:
                        self.estimated_in_lane[(leg2, light)] = self.count_in_leg[leg]
                        continue
                    to_add = car_increment / (N_LEG_LANES[leg] - 1)
                    self.estimated_in_lane[(leg2, light)] -= to_add
                    self.estimated_in_lane[(leg2, light)] = max(
                        self.estimated_in_lane[(leg2, light)], 0
                    )

            for leg in [l.name for l in env.legs]:
                leg_id = NESW_MAP_REVERSE[leg]

                self.count_in_leg[leg_id] = len(obs["vehicles"][leg])

                threshold = self.max_dist_per_leg[leg_id]
                entered = 0
                highest_dist = 0
                set_highest = False
                for v in obs["vehicles"][leg]:
                    dist = v["distance"]
                    speed = v["speed"]
                    last_pos = dist + speed
                    if last_pos > threshold:
                        entered += 1
                    if speed > 0:
                        highest_dist = max(highest_dist, dist)
                        set_highest = True

                self.cars_entered[leg_id] += entered
                # print (cars_entered[leg_id], "cars entered in leg", leg, "highest dist", highest_dist)
                self.max_dist_per_leg[leg_id] = highest_dist if set_highest else 100000

        # turn on new lights
        if self.time % CYCLE_TIME == 0:

            costs = []
            for combination in EXPLOIT_CYCLE:

                cost_increment = 0
                for leg, light in combination:
                    vehicles = tracker.tracked_vehicles[NESW_MAP[leg]]
                    lane_type = int_to_lanetype[light]
                    cars_controlled_by_light = [
                        (v, 1.0 / len(v.possible_lanes))
                        for v in vehicles
                        if any(lane_type.name == l.name for l in v.possible_lanes)
                    ]
                    if self.count_in_leg[leg] > 0:
                        leg_cost = 0
                        for v in vehicles:
                            current_cost = (
                                v.waiting_time + max(0, v.waiting_time - 90) ** 1.5
                            )
                            leg_cost += current_cost

                        lane_weight = (
                            self.estimated_in_lane[(leg, light)]
                            / self.count_in_leg[leg]
                        )
                        cost_increment += lane_weight * leg_cost
                        for v, prob_of_car_in_lane in cars_controlled_by_light:
                            current_cost = (
                                v.waiting_time + max(0, v.waiting_time - 90) ** 1.5
                            )
                            cost_increment += prob_of_car_in_lane * current_cost

                costs.append(cost_increment)

            idx = np.argmax(costs)

            cost_minimizing_choice = EXPLOIT_CYCLE[idx]
            # reset our choice of new action, but
            self.new_action *= 0
            for leg, light in cost_minimizing_choice:
                self.new_action[leg, light] = 1
                self.action[leg, light] = 1

        # turn off old lights a bit laters (3 ticks)
        if (self.time - 3) % CYCLE_TIME == 0:
            self.action *= 0
            self.action = self.new_action.copy()

        self.time += 1

        return self.action.flatten()


if __name__ == "__main__":
    env = sumo_env.SumoEnv(intersection_path="intersections")

    env.visualize = True
    env.reset()
    tracker = lane_tracker.LaneTracker(env, intersection_name="intersection_4")

    step_gen = env_step_generator(env)
    loss = 0
    while True:
        # input()
        obs, reward, done = next(step_gen)
        loss += reward
        # timelib.sleep(0.3)
        if done:
            break

    print("Total reward:", loss)

    env.close()
