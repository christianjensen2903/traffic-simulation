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


def index_to_direction(index: int) -> RoadDirection:
    return [RoadDirection.N, RoadDirection.E, RoadDirection.S, RoadDirection.W][index]


def direction_to_index(direction: RoadDirection) -> int:
    return [RoadDirection.N, RoadDirection.E, RoadDirection.S, RoadDirection.W].index(
        direction
    )


class LaneType(Enum):
    ALL = "all"
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT_RIGHT = "straight_right"
    STRAIGHT_LEFT = "straight_left"


class Road(BaseModel):
    lanes: list[LaneType]
    road_type: RoadType = RoadType.HIGHWAY_PRIMARY
    direction: RoadDirection


def get_lane_type_probabilities(
    position: int, total_lanes: int
) -> dict[LaneType, float]:
    """Get the lane type probabilities for the given position and total lanes"""
    if total_lanes == 1:
        return {
            LaneType.ALL: 0.4,
            LaneType.STRAIGHT: 0.4,
            LaneType.LEFT: 0.1,
            LaneType.RIGHT: 0.1,
            LaneType.STRAIGHT_LEFT: 0,
            LaneType.STRAIGHT_RIGHT: 0,
        }
    else:
        # Weights for each lane type
        w_left = 0.5  # Weight for left-turn lanes
        w_right = 1.0  # Weight for right-turn lanes
        w_straight_right = 1
        w_straight_left = 0.5
        w_straight = (
            3.0  # Weight for straight lanes (higher to make straight more common)
        )

        # Calculate scores based on position
        left_score = position * w_left
        right_score = (total_lanes - position - 1) * w_right
        straight_right_score = (total_lanes - position - 1) * w_straight_right
        straight_left_score = position * w_straight_left
        straight_score = w_straight

        total_score = (
            left_score
            + straight_score
            + right_score
            + straight_right_score
            + straight_left_score
        )

        # Calculate probabilities
        p_left = left_score / total_score
        p_straight = straight_score / total_score
        p_right = right_score / total_score
        p_straight_right = straight_right_score / total_score
        p_straight_left = straight_left_score / total_score

        return {
            LaneType.ALL: 0,
            LaneType.LEFT: p_left,
            LaneType.STRAIGHT: p_straight,
            LaneType.RIGHT: p_right,
            LaneType.STRAIGHT_RIGHT: p_straight_right,
            LaneType.STRAIGHT_LEFT: p_straight_left,
        }


def generate_random_road(
    direction: RoadDirection, min_lanes: int = 1, max_lanes: int = 6
) -> Road:
    """Generate a random road for the given direction"""
    assert min_lanes <= max_lanes
    road_type = random.choice(list(RoadType))

    num_lanes = random.randint(min_lanes, max_lanes)

    lanes = []

    for i in range(num_lanes):
        probabilities = get_lane_type_probabilities(i, num_lanes)
        lane_types = list(probabilities.keys())

        # Avoid lanes that would collide

        if LaneType.RIGHT in lanes:
            probabilities[LaneType.STRAIGHT_RIGHT] = 0
        if LaneType.STRAIGHT in lanes or LaneType.STRAIGHT_RIGHT in lanes:
            probabilities[LaneType.RIGHT] = 0
            probabilities[LaneType.STRAIGHT_RIGHT] = 0
        if LaneType.STRAIGHT_LEFT in lanes:
            break
        if LaneType.LEFT in lanes:
            probabilities[LaneType.RIGHT] = 0
            probabilities[LaneType.STRAIGHT] = 0
            probabilities[LaneType.STRAIGHT_RIGHT] = 0
            probabilities[LaneType.STRAIGHT_LEFT] = 0

        # Renormalize weights
        total_weight = sum(probabilities.values())
        for lane_type in lane_types:
            probabilities[lane_type] /= total_weight

        weights = list(probabilities.values())

        lane = random.choices(lane_types, weights=weights, k=1)[0]

        lanes.append(lane)

    return Road(lanes=lanes, road_type=road_type, direction=direction)


if __name__ == "__main__":
    print(generate_random_road(RoadDirection.N).model_dump_json(indent=4))
