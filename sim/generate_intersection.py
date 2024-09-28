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


def get_lane_target_direction(
    from_dir: RoadDirection, lane_type: LaneType
) -> RoadDirection:
    mapping = {
        RoadDirection.N: {
            LaneType.LEFT: RoadDirection.W,
            LaneType.STRAIGHT: RoadDirection.N,
            LaneType.RIGHT: RoadDirection.E,
        },
        RoadDirection.S: {
            LaneType.LEFT: RoadDirection.E,
            LaneType.STRAIGHT: RoadDirection.S,
            LaneType.RIGHT: RoadDirection.W,
        },
        RoadDirection.E: {
            LaneType.LEFT: RoadDirection.N,
            LaneType.STRAIGHT: RoadDirection.E,
            LaneType.RIGHT: RoadDirection.S,
        },
        RoadDirection.W: {
            LaneType.LEFT: RoadDirection.S,
            LaneType.STRAIGHT: RoadDirection.W,
            LaneType.RIGHT: RoadDirection.N,
        },
    }
    return mapping[from_dir].get(lane_type, None)


def adjust_roads_after_leg_removal(
    roads: list[Road], removed_direction: RoadDirection
) -> None:

    for road in roads:
        new_lanes = []
        for lane in road.lanes:
            if lane == LaneType.MAIN:
                # For MAIN lane, determine valid turns
                possible_lane_types = [LaneType.LEFT, LaneType.STRAIGHT, LaneType.RIGHT]
                for lt in possible_lane_types:
                    target_dir = get_lane_target_direction(road.direction, lt)
                    if target_dir != removed_direction:
                        new_lanes.append(lt)
            else:
                target_dir = get_lane_target_direction(road.direction, lane)
                if target_dir == removed_direction:
                    if lane == LaneType.LEFT or lane == LaneType.RIGHT:
                        new_lanes.append(LaneType.STRAIGHT)
                else:
                    new_lanes.append(lane)

        # Ensure at least one valid lane remains
        if not new_lanes:
            possible_lane_types = [LaneType.LEFT, LaneType.STRAIGHT, LaneType.RIGHT]
            for lt in possible_lane_types:
                target_dir = get_lane_target_direction(road.direction, lt)
                if target_dir != removed_direction:
                    new_lanes.append(lt)

        road.lanes = new_lanes


def generate_intersection() -> list[Road]:
    roads: list[Road] = []
    for direction in [
        RoadDirection.N,
        RoadDirection.S,
        RoadDirection.E,
        RoadDirection.W,
    ]:
        roads.append(generate_random_road(direction))

    # With 50% probability remove one of the roads
    if random.random() < 0.5:
        removed_road = roads.pop(random.randint(0, len(roads) - 1))
        removed_direction = get_opposite_direction(removed_road.direction)

        adjust_roads_after_leg_removal(roads, removed_direction)

    return roads
