import random
from generate_intersection import (
    Road,
    RoadDirection,
    LaneType,
    RoadType,
    generate_intersection,
    get_opposite_direction,
    get_lane_target_directions,
)
import math
from sim.environment import InternalLeg, Connection
from allowed_signal_calculator import generate_allowed_combinations
from generate_random_flow import generate_random_flow
import yaml
import os
import subprocess


INTERSECTION_OFFSET = 15
ROAD_LENGTH = 150  # Done to ensure road are long enough
LANE_WIDTH = 3.25
DISALLOW = "tram rail_urban rail rail_electric rail_fast ship cable_car subway"

speed_limits = {
    RoadType.HIGHWAY_PRIMARY: 27.78,
    RoadType.HIGHWAY_SECONDARY: 27.78,
    RoadType.HIGHWAY_TERTIARY: 22.22,
}


def shape_to_string(shape: list[tuple[float, float]]) -> str:
    return " ".join([f"{x:.2f},{y:.2f}" for x, y in shape])


def rotate_point_clockwise(point, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x, y = point
    x_new = x * math.cos(angle_radians) + y * math.sin(angle_radians)
    y_new = -x * math.sin(angle_radians) + y * math.cos(angle_radians)
    return (x_new, y_new)


def rotate_shape(shape, angle_degrees):
    return [rotate_point_clockwise(point, angle_degrees) for point in shape]


def generate_edge_xml(
    edge_id: str,
    from_node: str,
    to_node: str,
    shape: list[tuple[int, int]],
    road: Road,
    degrees: int = 0,
) -> str:
    xml = f"""
    <edge id="{edge_id}" from="{from_node}" to="{to_node}" name="{edge_id}" priority="11" type="{road.road_type.value}" spreadType="center" shape="{shape_to_string(shape)}">"""
    for i, lane in enumerate(road.lanes):
        lane_shape = [
            (shape[0][0] + i * LANE_WIDTH, shape[0][1]),
            (shape[1][0] + i * LANE_WIDTH, shape[1][1]),
        ]
        lane_shape = rotate_shape(lane_shape, degrees)
        xml += f"""
        <lane id="{edge_id}_{i}" index="{i}" disallow="{DISALLOW}" speed="{speed_limits[road.road_type]}" length="{ROAD_LENGTH}" shape="{shape_to_string(lane_shape)}"/>"""
    xml += """
    </edge>
    """
    return xml


def generate_dead_end_xml(
    junction_id: str,
    x: int,
    y: int,
    inc_lanes: list[str],
    shape: list[tuple[int, int]],
) -> str:
    return f"""
    <junction id="{junction_id}" type="dead_end" x="{x:.2f}" y="{y:.2f}" incLanes="{' '.join(inc_lanes)}" intLanes="" shape="{shape_to_string(shape)}"/>"""


def generate_connection_xml(
    from_node: str,
    to_node: str,
    from_lane: int,
    to_lane: int,
    lane_type: LaneType,
    state: str,
    via: str | None = None,
    tl: str | None = None,
    linkIndex: int | None = None,
) -> str:
    if lane_type == LaneType.STRAIGHT:
        dir = "s"
    elif lane_type == LaneType.LEFT:
        dir = "l"
    else:
        dir = "r"

    conn = f"""
    <connection from="{from_node}" to="{to_node}" fromLane="{from_lane}" toLane="{to_lane}" dir="{dir}" state="{state}" """
    if via is not None:
        conn += f"""via="{via}" """
    if tl is not None:
        conn += f"""tl="{tl}" """
    if linkIndex is not None:
        conn += f"""linkIndex="{linkIndex}" """
    conn += "/>"
    return conn


def generate_intersection_xml(
    path: str, min_lanes: int = 1, max_lanes: int = 6, roads: list[Road] = None
) -> None:
    # Load the SUMO network
    net_xml = """<?xml version="1.0" encoding="UTF-8"?>

    <net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">

        <location netOffset="-700171.91,-4526092.49" convBoundary="184.77,407.98,762.04,1291.78" origBoundary="29.375301,40.861337,29.386818,40.872898" projParameter="+proj=utm +zone=35 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"/>

        <type id="highway.primary" priority="12" numLanes="2" speed="27.78" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
        <type id="highway.secondary" priority="11" numLanes="1" speed="27.78" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>
        <type id="highway.tertiary" priority="10" numLanes="1" speed="22.22" disallow="tram rail_urban rail rail_electric rail_fast ship" oneway="0"/>

        <tlLogic id="intersection" type="static" programID="0" offset="0">
            <phase duration="15" state="GGgGgGgGrrrrrrrrrrrr" name="Factory"/>
            <phase duration="6"  state="yyyyyyyyrrrrrrrrrrrr"/>
            <phase duration="17" state="rrrrrrrGGGrrrrrrGGrr" name="Main_road"/>
            <phase duration="6"  state="rrrrrrryyyrrrrrryyrr"/>
            <phase duration="17" state="rrrrrrrrrrrrGGGGrrrr" name="Restaurant"/>
            <phase duration="6"  state="rrrrrrrrrrrryyyyrrrr"/>
            <phase duration="17" state="rrrrrrrrrrGGrrrrrrGG" name="Main_LEFT"/>
            <phase duration="6"  state="rrrrrrrrrryyrrrrrryy"/>
            <phase duration="17" state="rrrrrrrrrrrrrrrrGGGG" name="Hotel"/>
            <phase duration="2"  state="rrrrrrrrrrrrrrrryyyy"/>
            <phase duration="17" state="rrrrrrrGGGGGrrrrrrrr" name="Cafe"/>
            <phase duration="2"  state="rrrrrrryyyyyrrrrrrrr"/>
        </tlLogic>
    """

    if roads is None:
        roads = generate_intersection(min_lanes, max_lanes)

    # Mapping directions to angles
    direction_angles = {
        RoadDirection.S: 0,
        RoadDirection.E: 270,
        RoadDirection.N: 180,
        RoadDirection.W: 90,
    }

    linkIndexCounter = 0
    intLanes: list[str] = []
    lane_connection_strings: list[str] = []
    junction_strings: list[str] = []
    connections: list[Connection] = []

    for road in roads:
        degrees = direction_angles[road.direction]

        # Rotate shape resembling S1
        shape_1 = [(-5, ROAD_LENGTH + INTERSECTION_OFFSET), (-5, INTERSECTION_OFFSET)]

        shape_1 = rotate_shape(shape_1, degrees)

        # Add road to the XML
        net_xml += generate_edge_xml(
            edge_id=f"{road.direction.value}_1",
            from_node=f"{road.direction.value}_0",
            to_node="intersection",
            shape=shape_1,
            road=road,
            degrees=degrees,
        )

        junction_strings.append(
            generate_dead_end_xml(
                junction_id=f"{road.direction.value}_0",
                x=shape_1[0][0],
                y=shape_1[0][1],
                inc_lanes=[],
                shape=[(10, 10), (10, -10)],
            )
        )

        opposite_direction = get_opposite_direction(road.direction)

        # Get max number of incoming lanes
        max_incoming_lanes = min(
            2,
            max(
                [
                    len(
                        [
                            get_lane_target_directions(r.direction, l)
                            for l in r.lanes
                            if opposite_direction
                            in get_lane_target_directions(r.direction, l)
                        ]
                    )
                    for r in roads
                ]
            ),
        )  # 2 is by looking at validation set

        degrees = direction_angles[opposite_direction]

        # Rotate shape resembling S2

        shape_2 = [
            (-5, -INTERSECTION_OFFSET),
            (-5, -(ROAD_LENGTH + INTERSECTION_OFFSET)),
        ]
        # If above 4 lanes shift the shape to the left
        # A quick fix to avoid overlapping lanes
        if len(road.lanes) > 4:
            shape_2 = [
                (-7, -INTERSECTION_OFFSET),
                (-7, -(ROAD_LENGTH + INTERSECTION_OFFSET)),
            ]

        shape_2 = rotate_shape(
            shape_2,
            degrees,
        )

        net_xml += generate_edge_xml(
            edge_id=f"{opposite_direction.value}_2",
            from_node="intersection",
            to_node=f"{opposite_direction.value}_3",
            shape=shape_2,
            road=Road(
                lanes=[LaneType.STRAIGHT for _ in range(max_incoming_lanes)],
                road_type=road.road_type,
                direction=opposite_direction,
            ),
            degrees=degrees,
        )

        junction_strings.append(
            generate_dead_end_xml(
                junction_id=f"{opposite_direction.value}_3",
                x=shape_2[1][0],
                y=shape_2[1][1],
                inc_lanes=[
                    f"{opposite_direction.value}_2_{i}"
                    for i in range(max_incoming_lanes)
                ],
                shape=[(10, 10), (10, -10)],
            )
        )

        net_xml += "\n"

        outgoing_lane_counter: dict[RoadDirection, int] = {}

        for lane_number, lane in enumerate(road.lanes):
            target_directions = get_lane_target_directions(road.direction, lane)

            for target_direction in target_directions:
                if target_direction not in outgoing_lane_counter:
                    outgoing_lane_counter[target_direction] = 0

                lane_connection_strings.append(
                    generate_connection_xml(
                        from_node=f"{road.direction.value}_1",
                        to_node=f"{target_direction.value}_2",
                        from_lane=lane_number,
                        to_lane=outgoing_lane_counter[target_direction],
                        lane_type=lane,
                        state="o",
                        tl="intersection",
                        linkIndex=linkIndexCounter,
                    )
                )
                outgoing_lane_counter[target_direction] += 1

            connections.append(
                Connection(
                    priority=True,
                    index=linkIndexCounter,
                    groups=[f"{road.direction.value}_{lane.value}"],
                )
            )
            linkIndexCounter += 1

        net_xml += "\n"

    net_xml += "".join(junction_strings)

    incLanes: list[str] = []
    for road in roads:
        for lane_number, lane in enumerate(road.lanes):
            incLanes.append(f"{road.direction.value}_1_{lane_number}")
    net_xml += f"""
        <junction id="intersection" type="traffic_light" x="0" y="0" incLanes="{' '.join(incLanes)}" intLanes="{' '.join(intLanes)}" fringe="inner" shape="{shape_to_string([(INTERSECTION_OFFSET, INTERSECTION_OFFSET), (INTERSECTION_OFFSET, -INTERSECTION_OFFSET), (-INTERSECTION_OFFSET, -INTERSECTION_OFFSET), (-INTERSECTION_OFFSET, INTERSECTION_OFFSET)])}"/>"""

    net_xml += "\n"

    net_xml += "".join(lane_connection_strings)

    net_xml += """
    </net>
    """

    # Generate the rou.xml file
    generate_random_flow(path, roads)

    legs = [
        InternalLeg(
            name=f"{road.direction.value}_1",
            lanes=[lane.value for lane in road.lanes],
            groups=list(
                set([f"{road.direction.value}_{lane.value}" for lane in road.lanes])
            ),
            radar=None,
            segments=[f"{road.direction.value}_1"],
        )
        for road in roads
    ]

    groups = [
        f"{road.direction.value}_{lane.value}" for road in roads for lane in road.lanes
    ]
    groups = list(set(groups))

    allowed_combinations = generate_allowed_combinations(roads)

    # Make configuration.yaml
    config = {
        "intersections": [
            {
                "legs": [leg.model_dump(mode="json") for leg in legs],
                "groups": groups,
                "allowed_green_signal_combinations": [
                    {
                        "signal": str(signal),
                        "allowed": [str(lane) for lane in allowed],
                    }
                    for signal, allowed in allowed_combinations.items()
                ],
                "junction": "intersection",
                "amber_time": 3,
                "red_amber_time": 0,
                "connections": [
                    connection.model_dump(mode="json") for connection in connections
                ],
            }
        ]
    }

    # Write net.sumocfg
    sumo_cfg = f"""
    <configuration>
        <net-file value="intersection.net.xml"/>
        <route-files value="intersection.rou.xml"/>
    </configuration>
    """

    # Make folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/configuration.yaml", "w") as f:
        yaml.dump(config, f)

    with open(f"{path}/intersection.net.xml", "w") as f:
        f.write(net_xml)

    cmd = [
        "netconvert",
        "-s" f"{path}/intersection.net.xml",
        "-o",
        f"{path}/intersection.net.xml",
        "--no-internal-links",
        "false",
    ]

    subprocess.run(cmd)

    with open(f"{path}/net.sumocfg", "w") as f:
        f.write(sumo_cfg)


if __name__ == "__main__":
    # roads = [
    #     Road(
    #         lanes=[LaneType.RIGHT, LaneType.STRAIGHT, LaneType.LEFT],
    #         road_type=RoadType.HIGHWAY_PRIMARY,
    #         direction=RoadDirection.N,
    #     ),
    #     Road(
    #         lanes=[LaneType.RIGHT, LaneType.STRAIGHT, LaneType.LEFT],
    #         road_type=RoadType.HIGHWAY_PRIMARY,
    #         direction=RoadDirection.S,
    #     ),
    #     Road(
    #         lanes=[LaneType.RIGHT, LaneType.STRAIGHT_LEFT],
    #         road_type=RoadType.HIGHWAY_PRIMARY,
    #         direction=RoadDirection.E,
    #     ),
    #     Road(
    #         lanes=[LaneType.ALL],
    #         road_type=RoadType.HIGHWAY_TERTIARY,
    #         direction=RoadDirection.W,
    #     ),
    # ]

    roads = [
        Road(
            lanes=[LaneType.STRAIGHT, LaneType.STRAIGHT, LaneType.LEFT],
            road_type=RoadType.HIGHWAY_PRIMARY,
            direction=RoadDirection.N,
        ),
        Road(
            lanes=[LaneType.STRAIGHT, LaneType.STRAIGHT, LaneType.LEFT],
            road_type=RoadType.HIGHWAY_SECONDARY,
            direction=RoadDirection.S,
        ),
        Road(
            lanes=[LaneType.STRAIGHT, LaneType.STRAIGHT, LaneType.LEFT],
            road_type=RoadType.HIGHWAY_TERTIARY,
            direction=RoadDirection.E,
        ),
        Road(
            lanes=[
                LaneType.STRAIGHT,
                LaneType.STRAIGHT,
                LaneType.LEFT,
                LaneType.LEFT,
                LaneType.LEFT,
            ],
            road_type=RoadType.HIGHWAY_PRIMARY,
            direction=RoadDirection.W,
        ),
    ]

    generate_intersection_xml(path=f"intersections/3", roads=roads)
