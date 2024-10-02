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
    ALL = "ALL"
    STRAIGHT = "STRAIGHT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    STRAIGHT_RIGHT = "STRAIGHT_RIGHT"
    STRAIGHT_LEFT = "STRAIGHT_LEFT"


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
        w_LEFT = 0.5  # Weight for left-turn lanes
        w_RIGHT = 1.0  # Weight for right-turn lanes
        w_STRAIGHT_RIGHT = 1
        w_STRAIGHT_LEFT = 0.5
        w_STRAIGHT = (
            3.0  # Weight for straight lanes (higher to make straight more common)
        )

        # Calculate scores based on position
        left_score = position * w_LEFT
        right_score = (total_lanes - position - 1) * w_RIGHT
        straight_RIGHT_score = (total_lanes - position - 1) * w_STRAIGHT_RIGHT
        straight_LEFT_score = position * w_STRAIGHT_LEFT
        straight_score = w_STRAIGHT

        total_score = (
            left_score
            + straight_score
            + right_score
            + straight_RIGHT_score
            + straight_LEFT_score
        )

        # Calculate probabilities
        p_LEFT = left_score / total_score
        p_STRAIGHT = straight_score / total_score
        p_RIGHT = right_score / total_score
        p_STRAIGHT_RIGHT = straight_RIGHT_score / total_score
        p_STRAIGHT_LEFT = straight_LEFT_score / total_score

        return {
            LaneType.ALL: 0,
            LaneType.LEFT: p_LEFT,
            LaneType.STRAIGHT: p_STRAIGHT,
            LaneType.RIGHT: p_RIGHT,
            LaneType.STRAIGHT_RIGHT: p_STRAIGHT_RIGHT,
            LaneType.STRAIGHT_LEFT: p_STRAIGHT_LEFT,
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
