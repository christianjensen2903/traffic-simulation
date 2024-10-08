from collections import defaultdict
from enum import Enum
import numpy as np
import numpy as np
from lane_tracker import LaneTracker

# turn on
NORTH = 0
EAST_ = 1
SOUTH = 2
WEST_ = 3

LEGS = [NORTH, EAST_, SOUTH, WEST_]

NESW_MAP = {
    NORTH: "N",
    EAST_: "E",
    SOUTH: "S",
    WEST_: "W",
}

NESW_MAP_REVERSE = {
    "N": NORTH,
    "E": EAST_,
    "S": SOUTH,
    "W": WEST_,
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



CAR_EXIT_RATE = 6.0 / 14.0

CYCLE_TIME = 10

WARMUP_TICKS = 10

class RulebasedModel:
    def __init__(self, tracker: LaneTracker, legs: dict[str, list[LaneType]], is_first : bool):
        
        self.EXPLOIT_CYCLE_INTERSECTION_FIRST = [
            [(NORTH, STRAIGHT), (SOUTH, STRAIGHT)],
            [(EAST_, STRAIGHT), (WEST_, STRAIGHT)],
            [(EAST_, STRAIGHT), (EAST_, LEFT)],
            [(WEST_, STRAIGHT), (WEST_, LEFT)],
            [(NORTH, STRAIGHT), (NORTH, LEFT)],
            [(SOUTH, STRAIGHT), (SOUTH, LEFT)],
            [(NORTH, LEFT), (SOUTH, LEFT)],
            [(EAST_, LEFT), (WEST_, LEFT)],
        ]
        
        self.EXPLOIT_CYCLE_INTERSECTION_SECOND = [
            [
                (NORTH, STRAIGHT),
                (NORTH, RIGHT),
                (SOUTH, STRAIGHT),
                (SOUTH, RIGHT),
                (EAST_, RIGHT),
            ],
            [
                (NORTH, LEFT),
                (SOUTH, LEFT),
                (EAST_, RIGHT),
                (NORTH, RIGHT),
                (SOUTH, RIGHT),
            ],
            [(EAST_, STRAIGHT_LEFT), (EAST_, RIGHT), (SOUTH, RIGHT), (NORTH, RIGHT)],
            [
                (WEST_, ALL),
                (NORTH, RIGHT),
                (SOUTH, RIGHT),
                (EAST_, RIGHT),
                (NORTH, RIGHT),
            ],
            [(SOUTH, RIGHT), (NORTH, RIGHT), (EAST_, RIGHT)],
            [
                (NORTH, LEFT),
                (SOUTH, LEFT),
                (EAST_, RIGHT),
                (NORTH, RIGHT),
                (SOUTH, RIGHT),
            ],
        ]

        if is_first:
            self.EXPLOIT_CYCLE = self.EXPLOIT_CYCLE_INTERSECTION_FIRST
        else:
            self.EXPLOIT_CYCLE = self.EXPLOIT_CYCLE_INTERSECTION_SECOND

        self.reset(tracker, legs, is_first=is_first)

    def reset(self, tracker: LaneTracker, legs: dict[str, list[LaneType]], is_first : bool):
        
        if is_first:
            self.EXPLOIT_CYCLE = self.EXPLOIT_CYCLE_INTERSECTION_FIRST
            self.light_sizes = {
                (NORTH, STRAIGHT): 2,
                (NORTH, LEFT): 1,
                (WEST_, STRAIGHT): 2,
                (WEST_, LEFT): 3,
                (EAST_, STRAIGHT): 2,
                (EAST_, LEFT): 1,
                (SOUTH, LEFT): 1,
                (SOUTH, STRAIGHT): 2,
            }
            self.STAGGER_TIME = 2
            self.VALID_LANES = [
                (NORTH, STRAIGHT),
                (NORTH, LEFT),
                (SOUTH, STRAIGHT),
                (SOUTH, LEFT),
                (EAST_, STRAIGHT),
                (EAST_, LEFT),
                (WEST_, STRAIGHT),
                (WEST_, LEFT),
            ]
            self.N_LEG_LANES = {NORTH: 3, EAST_: 3, SOUTH: 3, WEST_: 5}
        else:
            self.light_sizes = defaultdict(lambda : 1)
            self.EXPLOIT_CYCLE = self.EXPLOIT_CYCLE_INTERSECTION_SECOND
            self.STAGGER_TIME = 3
            self.VALID_LANES = [
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
            self.N_LEG_LANES = {NORTH: 3, EAST_: 2, SOUTH: 3, WEST_: 1}

        self.is_first = is_first
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
        self.leg_lane_last_activated = defaultdict(int)

    def get_action(self, obs: dict):
        ###### Time starts at 1. Set all the lights to the first combination.
        if self.time == 1:
            for leg, light in self.EXPLOIT_CYCLE[0]:
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
            for leg, light in self.cost_minimizing_choice:
                old_guess = self.estimated_in_lane[(leg, light)]
                self.estimated_in_lane[(leg, light)] = max(old_guess - (6.0 / 14.0) * self.light_sizes[(leg, light)], 0)

            for leg in list(self.legs.keys()):
                try:
                    self.count_in_leg[leg] = len(obs["vehicles"][leg])
                except KeyError:
                    self.count_in_leg[leg] = 0

            # now, the estimated_in_lane counts must sum to the amount in the entire leg
            # distribute the difference onto all other
            self.estimated_count_in_leg = defaultdict(float)
            for leg, light in self.VALID_LANES:
                self.estimated_count_in_leg[leg] += self.estimated_in_lane[(leg, light)]

            for leg in LEGS:
                car_increment = (
                    self.estimated_count_in_leg[leg] - self.count_in_leg[leg]
                )
                for leg2, light in self.VALID_LANES:
                    if leg != leg2:  # we don't need to distribute to other legs
                        continue
                    if (
                        (
                            leg2,
                            light,
                        )
                        in self.cost_minimizing_choice
                    ):  # we don't need to distribute to lanes where we trust the count
                        continue
                    if self.N_LEG_LANES[leg] == 1:
                        self.estimated_in_lane[(leg2, light)] = self.count_in_leg[leg]
                        continue
                    to_add = car_increment / (self.N_LEG_LANES[leg] - 1)
                    self.estimated_in_lane[(leg2, light)] -= to_add
                    self.estimated_in_lane[(leg2, light)] = max(
                        self.estimated_in_lane[(leg2, light)], 0
                    )

            for leg in [l for l in self.legs.keys()]:
                leg_id = NESW_MAP_REVERSE[leg]
                try:
                    self.count_in_leg[leg_id] = len(obs["vehicles"][leg])
                except KeyError:
                    self.count_in_leg[leg_id] = 0

                threshold = self.max_dist_per_leg[leg_id]
                entered = 0
                highest_dist = 0
                set_highest = False
                for v in obs["vehicles"][leg]:
                    dist = v["distance_to_stop"]
                    speed = v["speed"]
                    last_pos = dist + speed
                    if last_pos > threshold:
                        entered += 1
                    if speed > 0:
                        highest_dist = max(highest_dist, dist)
                        set_highest = True

                self.cars_entered[leg_id] += entered
                self.max_dist_per_leg[leg_id] = highest_dist if set_highest else 100000

        # turn on new lights
        if self.time % CYCLE_TIME == 0:

            if self.is_first:
                
                print("estimated_in_lane", self.estimated_in_lane)
                
                costs = []
                for combination in self.EXPLOIT_CYCLE:
                    
                    cost_increment = 0
                    for leg, light in combination:
                        
                        time_unattended = self.time - self.leg_lane_last_activated[(leg, light)]
                        est_cars = self.estimated_in_lane[(leg, light)]
                        cost_increment += est_cars * (time_unattended + max(0, time_unattended) ** 1.5)

                    costs.append(cost_increment)
                
            else:
                costs = []
                for combination in self.EXPLOIT_CYCLE:
                    cost_increment = 0
                    
                    for leg, light in combination:

                        vehicles = self.tracker.tracked_vehicles[NESW_MAP[leg]]
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
                                    v.waiting_time + max(0, v.waiting_time - 79) ** 1.5
                                )
                                leg_cost += current_cost

                            lane_weight = (
                                self.estimated_in_lane[(leg, light)]
                                / self.count_in_leg[leg]
                            )
                            cost_increment += lane_weight * leg_cost
                            for v, prob_of_car_in_lane in cars_controlled_by_light:
                                current_cost = (
                                    v.waiting_time + max(0, v.waiting_time - 79) ** 1.5
                                )
                                cost_increment += prob_of_car_in_lane * current_cost

                    costs.append(cost_increment)

            idx = np.argmax(costs)
            
            for (leg, lane) in self.EXPLOIT_CYCLE[idx]:
                self.leg_lane_last_activated[(leg, lane)] = self.time

            self.cost_minimizing_choice = self.EXPLOIT_CYCLE[idx]
            # reset our choice of new action, but
            self.new_action *= 0
            for leg, light in self.cost_minimizing_choice:
                self.new_action[leg, light] = 1
                self.action[leg, light] = 1

        # turn off old lights a bit laters (3 ticks)
        if (self.time - self.STAGGER_TIME) % CYCLE_TIME == 0:
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
