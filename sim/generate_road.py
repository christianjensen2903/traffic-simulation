import random
from pydantic import BaseModel
from enum import Enum


class RoadType(Enum):
    HIGHWAY_PRIMARY = "highway.primary"
    HIGHWAY_SECONDARY = "highway.secondary"
    HIGHWAY_TERTIARY = "highway.tertiary"


class RoadDirection(Enum):
    N = "N"
    S = "S"
    E = "E"
    W = "W"


class LaneType(Enum):
    MAIN = "main"
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"


class Road(BaseModel):
    lanes: list[LaneType]
    road_type: RoadType
    direction: RoadDirection


def get_lane_type_probabilities(
    position: int, total_lanes: int
) -> dict[LaneType, float]:
    """Get the lane type probabilities for the given position and total lanes"""
    if total_lanes == 1:
        return {
            LaneType.MAIN: 0.4,
            LaneType.STRAIGHT: 0.4,
            LaneType.LEFT: 0.1,
            LaneType.RIGHT: 0.1,
        }
    else:
        # Weights for each lane type
        w_left = 1.0  # Weight for left-turn lanes
        w_right = 0.5  # Weight for right-turn lanes
        w_straight = (
            3.0  # Weight for straight lanes (higher to make straight more common)
        )

        # Calculate scores based on position
        left_score = (total_lanes - position - 1) * w_left
        right_score = position * w_right
        straight_score = w_straight

        total_score = left_score + straight_score + right_score

        # Calculate probabilities
        p_left = left_score / total_score
        p_straight = straight_score / total_score
        p_right = right_score / total_score

        return {
            LaneType.MAIN: 0,
            LaneType.LEFT: p_left,
            LaneType.STRAIGHT: p_straight,
            LaneType.RIGHT: p_right,
        }


def generate_random_road(direction: RoadDirection) -> Road:
    """Generate a random road for the given direction"""
    road_type = random.choice(list(RoadType))

    num_lanes = random.randint(1, 5)

    lanes = []

    for i in range(num_lanes):
        probabilities = get_lane_type_probabilities(i, num_lanes)
        lane_types = list(probabilities.keys())

        # Avoid lanes that would collide

        if LaneType.STRAIGHT in lanes:
            probabilities[LaneType.LEFT] = 0
        if LaneType.RIGHT in lanes:
            probabilities[LaneType.LEFT] = 0
            probabilities[LaneType.STRAIGHT] = 0

        # Renormalize weights
        total_weight = sum(probabilities.values())
        for lane_type in lane_types:
            probabilities[lane_type] /= total_weight

        weights = list(probabilities.values())

        lane = random.choices(lane_types, weights=weights, k=1)[0]

        lanes.append(lane)

    return Road(lanes=lanes, road_type=road_type, direction=direction)
