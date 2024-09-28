from generate_intersection import Road, RoadDirection, LaneType, RoadType


class Lane:
    """Class to model the movement into a specific direction"""

    def __init__(self, from_dir: RoadDirection, lane_type: LaneType):
        self.from_dir = from_dir
        self.lane_type = lane_type

    def __eq__(self, other):
        return self.from_dir == other.from_dir and self.lane_type == other.lane_type

    def __hash__(self):
        return hash((self.from_dir, self.lane_type))

    def __repr__(self):
        return f"{self.from_dir.value}_{self.lane_type.value}".upper()


def generate_lanes(roads: list[Road]) -> list[Lane]:
    return [
        Lane(road.direction, lane_type) for road in roads for lane_type in road.lanes
    ]


def is_conflicting(l1: Lane, l2: Lane) -> bool:
    """Check if two lanes are conflicting"""

    if l1 == l2:
        return True

    # Define opposite and adjacent directions
    opposite = {
        RoadDirection.N: RoadDirection.S,
        RoadDirection.S: RoadDirection.N,
        RoadDirection.E: RoadDirection.W,
        RoadDirection.W: RoadDirection.E,
    }

    left_adjacent = {
        RoadDirection.N: RoadDirection.E,
        RoadDirection.S: RoadDirection.W,
        RoadDirection.E: RoadDirection.S,
        RoadDirection.W: RoadDirection.N,
    }

    right_adjacent = {
        RoadDirection.N: RoadDirection.W,
        RoadDirection.S: RoadDirection.E,
        RoadDirection.E: RoadDirection.N,
        RoadDirection.W: RoadDirection.S,
    }

    # Conflict rules
    # Left turn conflicts
    if l1.lane_type == LaneType.LEFT:

        if l2.lane_type == LaneType.MAIN:
            return True

        # Conflicts with opposing straight and right turns
        if l2.from_dir == opposite[l1.from_dir]:
            if l2.lane_type in [LaneType.STRAIGHT, LaneType.RIGHT]:
                return True
        # Conflicts with adjacent left turns
        if l2.from_dir in [left_adjacent[l1.from_dir], right_adjacent[l1.from_dir]]:
            return True

    # Straight movement conflicts
    if l1.lane_type == LaneType.STRAIGHT:
        if l2.lane_type == LaneType.MAIN:
            return True

        # Conflicts with opposing left turns
        if l2.from_dir == opposite[l1.from_dir]:
            if l2.lane_type == LaneType.LEFT:
                return True

        # Conflicts with adjacent left turns
        if l2.from_dir in [left_adjacent[l1.from_dir], right_adjacent[l1.from_dir]]:
            # TODO: Check if left adjacant can turn right
            # In theory possible
            return True

    # Right turn conflicts
    if l1.lane_type == LaneType.RIGHT:
        if l2.lane_type == LaneType.MAIN:
            return True

        # Conflicts with opposing left turns
        if l2.from_dir == opposite[l1.from_dir]:
            if l2.lane_type == LaneType.LEFT:
                return True

        # Conflicts with adjacent turns
        if l2.from_dir in [left_adjacent[l1.from_dir], right_adjacent[l1.from_dir]]:
            # TODO: Same here with check left adjacent
            return True

    if l1.lane_type == LaneType.MAIN:
        return True

    return False


def generate_allowed_combinations(roads: list[Road]) -> dict[Lane, list[Lane]]:
    movements = generate_lanes(roads)
    allowed_combinations: dict[Lane, set[Lane]] = {}
    for m1 in movements:
        allowed_combinations[m1] = set()
        for m2 in movements:
            if not is_conflicting(m1, m2):
                allowed_combinations[m1].add(m2)

    return {m1: list(allowed) for m1, allowed in allowed_combinations.items()}


if __name__ == "__main__":
    # Example usage:

    roads = [
        Road(
            lanes=[LaneType.LEFT, LaneType.LEFT, LaneType.STRAIGHT, LaneType.RIGHT],
            direction=RoadDirection.N,
            road_type=RoadType.HIGHWAY_PRIMARY,
        ),
        Road(
            lanes=[LaneType.LEFT, LaneType.LEFT, LaneType.STRAIGHT, LaneType.RIGHT],
            direction=RoadDirection.S,
            road_type=RoadType.HIGHWAY_PRIMARY,
        ),
        Road(
            lanes=[LaneType.LEFT, LaneType.LEFT, LaneType.STRAIGHT, LaneType.RIGHT],
            direction=RoadDirection.E,
            road_type=RoadType.HIGHWAY_PRIMARY,
        ),
        Road(
            lanes=[LaneType.LEFT, LaneType.LEFT, LaneType.STRAIGHT, LaneType.RIGHT],
            direction=RoadDirection.W,
            road_type=RoadType.HIGHWAY_PRIMARY,
        ),
    ]

    allowed_combinations = generate_allowed_combinations(roads)

    # Pretty print
    for movement, allowed in allowed_combinations.items():
        print(f"{movement}: {allowed}")
