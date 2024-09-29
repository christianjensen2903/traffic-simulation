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
from environment import InternalLeg, Connection
from allowed_signal_calculator import generate_allowed_combinations
import yaml
import os
import subprocess


speed_limits = {
    RoadType.HIGHWAY_PRIMARY: 27.78,
    RoadType.HIGHWAY_SECONDARY: 27.78,
    RoadType.HIGHWAY_TERTIARY: 22.22,
}


def generate_intersection_xml(path: str):
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
            <phase duration="17" state="rrrrrrrrrrGGrrrrrrGG" name="Main_left"/>
            <phase duration="6"  state="rrrrrrrrrryyrrrrrryy"/>
            <phase duration="17" state="rrrrrrrrrrrrrrrrGGGG" name="Hotel"/>
            <phase duration="2"  state="rrrrrrrrrrrrrrrryyyy"/>
            <phase duration="17" state="rrrrrrrGGGGGrrrrrrrr" name="Cafe"/>
            <phase duration="2"  state="rrrrrrryyyyyrrrrrrrr"/>
        </tlLogic>
    """

    INTERSECTION_OFFSET = 15
    ROAD_LENGTH = 100
    LANE_WIDTH = 3.2
    DISALLOW = "tram rail_urban rail rail_electric rail_fast ship cable_car subway"

    roads = generate_intersection()

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
        shape_1 = rotate_shape(
            [(-5, ROAD_LENGTH + INTERSECTION_OFFSET), (-5, INTERSECTION_OFFSET)],
            degrees,
        )

        # Add road to the XML
        net_xml += generate_edge_xml(
            edge_id=f"{road.direction.value}_1",
            from_node=f"{road.direction.value}_0",
            to_node="intersection",
            shape=shape_1,
            road=road,
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

        # Add a single lane road in the opposite direction making up the leg
        # 1 lane only necessary since we don't care about collisions
        opposite_direction = get_opposite_direction(road.direction)
        degrees = direction_angles[opposite_direction]

        # Rotate shape resembling S2
        shape_2 = rotate_shape(
            [(-5, -INTERSECTION_OFFSET), (-5, -(ROAD_LENGTH + INTERSECTION_OFFSET))],
            degrees,
        )
        net_xml += generate_edge_xml(
            edge_id=f"{opposite_direction.value}_2",
            from_node="intersection",
            to_node=f"{opposite_direction.value}_3",
            shape=shape_2,
            road=Road(
                lanes=[LaneType.STRAIGHT],
                road_type=road.road_type,
                direction=opposite_direction,
            ),
        )

        junction_strings.append(
            generate_dead_end_xml(
                junction_id=f"{opposite_direction.value}_3",
                x=shape_2[1][0],
                y=shape_2[1][1],
                inc_lanes=[f"{opposite_direction.value}_2_0"],  # Only one lane
                shape=[(10, 10), (10, -10)],
            )
        )

        net_xml += "\n"

        for lane_number, lane in enumerate(road.lanes):
            target_directions = get_lane_target_directions(road.direction, lane)
            for target_direction in target_directions:
                lane_connection_strings.append(
                    generate_connection_xml(
                        from_node=f"{road.direction.value}_1",
                        to_node=f"{target_direction.value}_2",
                        from_lane=lane_number,
                        to_lane=0,
                        lane_type=lane,
                        state="o",
                        tl="intersection",
                        linkIndex=linkIndexCounter,
                    )
                )

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

    rou_xml = """<?xml version="1.0" encoding="UTF-8"?>

    <!-- generated on 2024-07-04 11:40:22 by Eclipse SUMO netedit Version 1.15.0
    -->

    <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">"""

    options = {150: 0.5, 300: 0.2, 600: 0.3, 900: 0.3, 1800: 0.04, 3600: 0.01}

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
                3600,
            )
            target_directions = get_lane_target_directions(road.direction, lane_type)
            for target_direction in target_directions:
                rou_xml += f"""
                <flow id="f_{id_counter}" begin="0.00" from="{road.direction.value}_1" to="{target_direction.value}_2" end="3600.00" vehsPerHour="{vehicles_per_hour}"/>"""
                id_counter += 1
    rou_xml += """
    </routes>"""

    legs = [
        InternalLeg(
            name=road.direction.value,
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
    ]

    subprocess.run(cmd)

    with open(f"{path}/intersection.rou.xml", "w") as f:
        f.write(rou_xml)

    with open(f"{path}/net.sumocfg", "w") as f:
        f.write(sumo_cfg)


if __name__ == "__main__":
    generate_intersection_xml(path=f"models/{2}")
