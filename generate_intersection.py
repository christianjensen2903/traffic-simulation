import random
from generate_road import Road, RoadDirection, RoadType, LaneType, generate_random_road


def get_opposite_direction(direction: RoadDirection) -> RoadDirection:
    mapping = {
        RoadDirection.N: RoadDirection.S,
        RoadDirection.S: RoadDirection.N,
        RoadDirection.E: RoadDirection.W,
        RoadDirection.W: RoadDirection.E,
    }
    return mapping[direction]


def get_lane_target_directions(
    from_dir: RoadDirection, lane_type: LaneType
) -> list[RoadDirection]:
    mapping = {
        RoadDirection.N: {
            LaneType.LEFT: [RoadDirection.W],
            LaneType.STRAIGHT: [RoadDirection.N],
            LaneType.RIGHT: [RoadDirection.E],
            LaneType.STRAIGHT_LEFT: [RoadDirection.W, RoadDirection.N],
            LaneType.STRAIGHT_RIGHT: [RoadDirection.N, RoadDirection.E],
            LaneType.ALL: [RoadDirection.W, RoadDirection.N, RoadDirection.E],
        },
        RoadDirection.S: {
            LaneType.LEFT: [RoadDirection.E],
            LaneType.STRAIGHT: [RoadDirection.S],
            LaneType.RIGHT: [RoadDirection.W],
            LaneType.STRAIGHT_LEFT: [RoadDirection.E, RoadDirection.S],
            LaneType.STRAIGHT_RIGHT: [RoadDirection.S, RoadDirection.W],
            LaneType.ALL: [RoadDirection.E, RoadDirection.S, RoadDirection.W],
        },
        RoadDirection.E: {
            LaneType.LEFT: [RoadDirection.N],
            LaneType.STRAIGHT: [RoadDirection.E],
            LaneType.RIGHT: [RoadDirection.S],
            LaneType.STRAIGHT_LEFT: [RoadDirection.N, RoadDirection.E],
            LaneType.STRAIGHT_RIGHT: [RoadDirection.E, RoadDirection.S],
            LaneType.ALL: [RoadDirection.N, RoadDirection.E, RoadDirection.S],
        },
        RoadDirection.W: {
            LaneType.LEFT: [RoadDirection.S],
            LaneType.STRAIGHT: [RoadDirection.W],
            LaneType.RIGHT: [RoadDirection.N],
            LaneType.STRAIGHT_LEFT: [RoadDirection.S, RoadDirection.W],
            LaneType.STRAIGHT_RIGHT: [RoadDirection.W, RoadDirection.N],
            LaneType.ALL: [RoadDirection.S, RoadDirection.W, RoadDirection.N],
        },
    }
    return mapping[from_dir][lane_type]


def adjust_roads_after_leg_removal(
    roads: list[Road], removed_direction: RoadDirection
) -> None:
    for road in roads:
        new_lanes = []
        for lane in road.lanes:
            if lane == LaneType.ALL:
                if (
                    get_lane_target_directions(road.direction, LaneType.LEFT)[0]
                    == removed_direction
                ):
                    new_lanes.append(LaneType.STRAIGHT_RIGHT)
                elif (
                    get_lane_target_directions(road.direction, LaneType.RIGHT)[0]
                    == removed_direction
                ):
                    new_lanes.append(LaneType.STRAIGHT_LEFT)
                else:  # If opposite is removed
                    new_lanes.append(LaneType.LEFT)
                    new_lanes.append(LaneType.STRAIGHT)
            elif lane == LaneType.STRAIGHT_RIGHT:
                if (
                    get_lane_target_directions(road.direction, LaneType.RIGHT)[0]
                    == removed_direction
                ):
                    new_lanes.append(LaneType.STRAIGHT)
                elif (
                    get_lane_target_directions(road.direction, LaneType.STRAIGHT)[0]
                    == removed_direction
                ):
                    new_lanes.append(LaneType.RIGHT)
                else:
                    new_lanes.append(lane)
            elif lane == LaneType.STRAIGHT_LEFT:
                if (
                    get_lane_target_directions(road.direction, LaneType.LEFT)[0]
                    == removed_direction
                ):
                    new_lanes.append(LaneType.STRAIGHT)
                elif (
                    get_lane_target_directions(road.direction, LaneType.STRAIGHT)[0]
                    == removed_direction
                ):
                    new_lanes.append(LaneType.LEFT)
                else:
                    new_lanes.append(lane)
            else:
                target_dir = get_lane_target_directions(road.direction, lane)[0]
                if target_dir == removed_direction:
                    if lane == LaneType.LEFT or lane == LaneType.RIGHT:
                        new_lanes.append(LaneType.STRAIGHT)
                else:
                    new_lanes.append(lane)

        # Ensure at least one valid lane remains
        if not new_lanes:
            possible_lane_types = [LaneType.LEFT, LaneType.STRAIGHT, LaneType.RIGHT]
            for lt in possible_lane_types:
                target_dir = get_lane_target_directions(road.direction, lt)[0]
                if target_dir != removed_direction:
                    new_lanes.append(lt)

        road.lanes = new_lanes


def generate_intersection(min_lanes: int, max_lanes: int = 6) -> list[Road]:
    roads: list[Road] = []
    for direction in [
        RoadDirection.N,
        RoadDirection.S,
        RoadDirection.E,
        RoadDirection.W,
    ]:
        roads.append(generate_random_road(direction, min_lanes, max_lanes))

    # With 50% probability remove one of the roads
    if random.random() < 0.5:
        removed_road = roads.pop(random.randint(0, len(roads) - 1))
        removed_direction = get_opposite_direction(removed_road.direction)

        adjust_roads_after_leg_removal(roads, removed_direction)

    return roads


if __name__ == "__main__":
    intersection = generate_intersection()
    for road in intersection:
        print(road.model_dump_json(indent=4))
