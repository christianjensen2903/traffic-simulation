import random
from generate_intersection import (
    Road,
    RoadDirection,
    LaneType,
    generate_intersection,
    get_opposite_direction,
    get_lane_target_direction,
)
import math

# Load the SUMO network
xml = """<?xml version="1.0" encoding="UTF-8"?>

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


INTERSECTION_OFFSET = 10
ROAD_LENGTH = 100
LANE_WIDTH = 3.2
DISALLOW = "tram rail_urban rail rail_electric rail_fast ship cable_car subway"
SPEED = 25

roads = generate_intersection()


def shape_to_string(shape: list[tuple[float, float]]) -> str:
    return " ".join([f"{x:.2f},{y:.2f}" for x, y in shape])


def flip_around_axis(
    shape: list[tuple[int, int]], around_y: bool, around_x: bool
) -> list[tuple[int, int]]:
    return [(x * -1 if around_y else x, y * -1 if around_x else y) for x, y in shape]


def rotate_point_clockwise(point, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    x, y = point
    x_new = x * math.cos(angle_radians) + y * math.sin(angle_radians)
    y_new = -x * math.sin(angle_radians) + y * math.cos(angle_radians)
    return (x_new, y_new)


def rotate_shape(shape, angle_degrees):
    return [rotate_point_clockwise(point, angle_degrees) for point in shape]


def generate_edge_xml(
    edge_id: str, from_node: str, to_node: str, shape: list[tuple[int, int]], road: Road
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
        <lane id="{edge_id}_{i}" index="{i}" disallow="{DISALLOW}" speed="{SPEED}" length="{ROAD_LENGTH}" shape="{shape_to_string(lane_shape)}"/>"""
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
    RoadDirection.E: 90,
    RoadDirection.N: 180,
    RoadDirection.W: 270,
}

linkIndexCounter = 0
intLanes: list[str] = []
lane_connection_strings: list[str] = []
intersection_connection_strings: list[str] = []
junction_strings: list[str] = []

for road in roads:
    xml += f"""
    <edge id=":intersection_{road.direction.value}" function="internal">"""
    for lane_number, lane in enumerate(road.lanes):
        int_id = f":intersection_{road.direction.value}_{lane_number}"
        xml += f"""
        <lane id="{int_id}" index="{lane_number}" disallow="{DISALLOW}" speed="{SPEED}" length="30" shape="313.74,480.14 306.42,472.85 301.49,466.60 297.40,461.12 292.60,456.08"/>"""
        intLanes.append(int_id)
    xml += """
    </edge>
    """

    degrees = direction_angles[road.direction]

    # Rotate shape resembling S1
    shape_1 = rotate_shape(
        [(-5, ROAD_LENGTH + INTERSECTION_OFFSET), (-5, INTERSECTION_OFFSET)], degrees
    )

    # Add road to the XML
    xml += generate_edge_xml(
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
    xml += generate_edge_xml(
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
            inc_lanes=[f"{opposite_direction.value}_2_1"],  # Only one lane
            shape=[(10, 10), (10, -10)],
        )
    )

    xml += "\n"

    for lane_number, lane in enumerate(road.lanes):
        print(road.direction, lane)
        target_direction = get_lane_target_direction(road.direction, lane)
        lane_connection_strings.append(
            generate_connection_xml(
                from_node=f"{road.direction.value}_1",
                to_node=f"{target_direction.value}_2",
                from_lane=lane_number,
                to_lane=0,
                lane_type=lane,
                state="o",
                via=f":intersection_{road.direction.value}_{lane_number}",
                tl="intersection",
                linkIndex=linkIndexCounter,
            )
        )
        linkIndexCounter += 1
    xml += "\n"

    for lane_number, lane in enumerate(road.lanes):
        target_direction = get_lane_target_direction(road.direction, lane)
        intersection_connection_strings.append(
            generate_connection_xml(
                from_node=f":intersection_{road.direction.value}",
                to_node=f"{target_direction.value}_2",
                from_lane=lane_number,
                to_lane=0,
                lane_type=target_direction,
                state="M",
            )
        )
    xml += "\n"

xml += "".join(junction_strings)

incLanes: list[str] = []
for road in roads:
    for lane_number, lane in enumerate(road.lanes):
        incLanes.append(f"{road.direction.value}_1_{lane_number}")
xml += f"""
    <junction id="intersection" type="traffic_light" x="0" y="0" incLanes="{' '.join(incLanes)}" intLanes="{' '.join(intLanes)}" fringe="inner" shape="312.73,481.39 329.28,460.72 327.46,458.20 327.22,456.67 327.42,454.97 328.06,453.09 329.14,451.02 310.93,440.02 309.42,441.66 308.56,442.00 307.65,442.02 306.66,441.72 305.61,441.10 291.55,457.29 293.48,460.48 293.59,462.48 293.14,464.76 292.13,467.30 290.55,470.12 307.43,482.02 309.04,480.50 309.90,480.23 310.81,480.29 311.75,480.68"/>"""
xml += "\n"

xml += "".join(lane_connection_strings)
xml += "\n"
xml += "".join(intersection_connection_strings)
xml += "\n"

xml += """
</net>
"""

# Save to file
with open("intersection.net.xml", "w") as f:
    f.write(xml)
