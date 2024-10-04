import random
from generate_intersection import (
    Road,
    RoadDirection,
    LaneType,
    RoadType,
    get_lane_target_directions,
)


def generate_random_flow(path: str, roads: list[Road]) -> None:
    rou_xml = """<?xml version="1.0" encoding="UTF-8"?>

    <!-- generated on 2024-07-04 11:40:22 by Eclipse SUMO netedit Version 1.15.0
    -->

    <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">"""

    options = {0: 0.02, 150: 0.05, 300: 0.2, 600: 0.22, 900: 0.2, 1800: 0.2, 3600: 0.01}

    id_counter = 0
    for road in roads:
        for lane_type in list(LaneType):
            number_of_lanes = road.lanes.count(lane_type)
            if number_of_lanes == 0:
                continue
            # Pick a random number of vehicles per hour based on the options
            vehicles_per_hour = min(
                sum(
                    random.choices(
                        list(options.keys()),
                        weights=list(options.values()),
                        k=number_of_lanes,
                    )
                ),
                5000,
            )
            if vehicles_per_hour == 0:
                continue
            target_directions = get_lane_target_directions(road.direction, lane_type)
            for target_direction in target_directions:
                rou_xml += f"""
                <flow id="f_{id_counter}" begin="0.00" from="{road.direction.value}_1" to="{target_direction.value}_2" end="3600.00" vehsPerHour="{vehicles_per_hour}"/>"""
                id_counter += 1
    rou_xml += """
    </routes>"""

    with open(f"sim/{path}/intersection.rou.xml", "w") as f:
        f.write(rou_xml)


if __name__ == "__main__":
    roads = [
        Road(
            lanes=[LaneType.RIGHT, LaneType.STRAIGHT, LaneType.LEFT],
            road_type=RoadType.HIGHWAY_PRIMARY,
            direction=RoadDirection.N,
        ),
        Road(
            lanes=[LaneType.RIGHT, LaneType.STRAIGHT, LaneType.LEFT],
            road_type=RoadType.HIGHWAY_PRIMARY,
            direction=RoadDirection.S,
        ),
        Road(
            lanes=[LaneType.RIGHT, LaneType.STRAIGHT_LEFT],
            road_type=RoadType.HIGHWAY_PRIMARY,
            direction=RoadDirection.E,
        ),
        Road(
            lanes=[LaneType.ALL],
            road_type=RoadType.HIGHWAY_PRIMARY,
            direction=RoadDirection.W,
        ),
    ]
    generate_random_flow(path=f"intersections/{2}", roads=roads)
