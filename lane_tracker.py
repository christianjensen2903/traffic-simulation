from pydantic import BaseModel
from generate_road import LaneType, RoadDirection, RoadType
from sumo_env import SumoEnv, InternalLeg


class TrackedVehicle(BaseModel):
    waiting_time: float = 0
    speed: float = 0
    distance: float = 0
    possible_lanes: list[LaneType] = []


# Used to determine when a vehicle has crossed a segment
segment_threshold_lookup = {
    "intersection_1": {
        "W": [23.5],
        "E": [16.1, 66.4],
        "N": [16.1, 75.2],
        "S": [],
    },
    "intersection_2": {"W": [], "E": [21], "N": [], "S": []},
    "intersection_3": {"W": [], "E": [], "N": [], "S": []},
    "intersection_4": {"W": [], "E": [21], "N": [], "S": []},
    "test_intersection": {"W": [], "E": [], "N": [], "S": []},
}


class LaneTracker:
    """
    The ordering of the cars follows
    Segment -> Lane (Right to left in driving direction) -> Car (Further to closer)
    Make algorithm that tracks the cars in the lanes.
    There can maximum be 15 cars in a lane
    """

    def __init__(self, intersection_name: str, legs: dict[str, list[LaneType]]):
        self.reset(intersection_name, legs)

    def reset(self, intersection_name: str, legs: dict[str, list[LaneType]]):
        self.segment_thresholds = segment_threshold_lookup[intersection_name]
        self.tracked_vehicles: dict[str, list[TrackedVehicle]] = {
            leg_name: [] for leg_name in legs.keys()
        }
        self.legs = legs

    def get_lanes(self, leg_name: str) -> list[LaneType]:
        for leg_name in self.legs.keys():
            if leg_name == leg_name:
                return self.legs[leg_name]

    def add_vehicle(self, leg_name: str, vehicle: TrackedVehicle):
        self.tracked_vehicles[leg_name].vehicles.append(vehicle)
        return vehicle

    def find_and_update_vehicle(
        self,
        leg_name: str,
        distance: float,
        possible_lanes: list[LaneType],
        speed: float,
    ):  # -> TrackedVehicle | None:
        """Finds the vehicles in the leg and updates the vehicle if found"""
        
        try:
            the_list = self.tracked_vehicles[leg_name]
        except KeyError:
            the_list = self.tracked_vehicles[leg_name[0]]
        
        for vehicle in the_list:
            old_distance = distance + speed

            if vehicle.distance == old_distance:
                for lane in possible_lanes:
                    if lane in vehicle.possible_lanes:
                        vehicle.distance = distance
                        vehicle.speed = speed
                        if speed < 0.5:
                            vehicle.waiting_time += 1
                        return vehicle

        try:
            tracked_vehicle = TrackedVehicle(
                distance=distance,
                speed=speed,
                possible_lanes=[lane.value for lane in possible_lanes],
            )
        except Exception as e:
            print(possible_lanes)
            raise e
        return tracked_vehicle  # No vehicle found

    def _get_unique_lane_types(self, lanes: list[LaneType]) -> list[LaneType]:
        new_lanes = []
        for lane in lanes:
            if lane not in new_lanes:
                new_lanes.append(lane)
        return new_lanes

    def is_index_greater_than_last_occurrence(
        self, lane_list: list[LaneType], lane_type: LaneType, lane_index: int
    ) -> bool:
        """Checks if the provided index is greater than the last occurrence of the lane_type in the lane_list"""
        try:
            last_index = self.get_max_index(lane_list, lane_type)
            # Check if the provided index is greater than the last occurrence
            return lane_index > last_index
        except ValueError:
            # If the direction_type is not found in the list
            return False

    def is_index_less_than_first_occurrence(
        self, lane_list: list[LaneType], lane_type: LaneType, lane_index: int
    ) -> bool:
        """Checks if the provided index is less than the first occurrence of the lane_type in the lane_list"""
        try:
            first_index = self.get_min_index(lane_list, lane_type)
            # Check if the provided index is less than the first occurrence
            return lane_index < first_index
        except ValueError:
            # If the direction_type is not found in the list
            return False

    def get_initial_possible_lanes(self, leg_name: str) -> list[LaneType]:
        """Gets the possible lane types for a leg"""
        lanes = self.get_lanes(leg_name)
        return self._get_unique_lane_types(lanes)

    def get_segment_threshold_forward(self, leg_name: str, distance: float) -> float:
        """Get the threshold for the leg and distance"""
        try:
            thresholds = self.segment_thresholds[leg_name]
        except KeyError:
            thresholds = self.segment_thresholds[leg_name[0]]
        return min([t for t in thresholds if distance < t], default=float("inf"))

    def get_segment_threshold_backward(self, leg_name: str, distance: float) -> float:
        """Get the threshold for the leg and distance"""
        try:
            thresholds = self.segment_thresholds[leg_name]
        except KeyError:
            thresholds = self.segment_thresholds[leg_name[0]]
        return max([t for t in thresholds if distance > t], default=float("-inf"))

    def update_lane_info_forward(
        self,
        leg_name: str,
        lanes: list[LaneType],
        distance: float,
        last_vehicle: TrackedVehicle,
        lane_index: int,
        possible_lanes: list[LaneType],
    ) -> tuple[int, list[LaneType]]:
        """Increment the lane index if necessary and update the possible lanes if reset"""
        threshold = self.get_segment_threshold_forward(leg_name, last_vehicle.distance)

        # Check if it has skipped to the next segment
        if distance > threshold:
            lane_index = 0
            possible_lanes = self.get_initial_possible_lanes(leg_name)
            return lane_index, possible_lanes
        
        if distance > last_vehicle.distance:
            lane_index += 1
        try:
            if lane_index >= len(lanes):
                possible_lanes = self.get_initial_possible_lanes(leg_name)
                lane_index = 0
            elif self.is_index_greater_than_last_occurrence(
                lanes, possible_lanes[0], lane_index
            ):
                possible_lanes.pop(0)
        except IndexError:
            possible_lanes = self.get_initial_possible_lanes(leg_name)
            lane_index = 0
        return lane_index, possible_lanes

    def update_lane_info_backward(
        self,
        leg_name: str,
        lanes: list[LaneType],
        distance: float,
        last_vehicle: TrackedVehicle,
        lane_index: int,
        possible_lanes: list[LaneType],
    ) -> tuple[int, list[LaneType]]:
        """Increment the lane index if necessary and update the possible lanes if reset"""
        threshold = self.get_segment_threshold_backward(leg_name, last_vehicle.distance)

        # Check if it has skipped to the next segment
        if distance < threshold:
            lane_index = len(lanes) - 1
            possible_lanes = self.get_initial_possible_lanes(leg_name)
            return lane_index, possible_lanes

        if distance < last_vehicle.distance:
            lane_index -= 1
        try:
            if lane_index < 0:
                possible_lanes = self.get_initial_possible_lanes(leg_name)
                lane_index = len(lanes) - 1
            elif self.is_index_less_than_first_occurrence(
                lanes, possible_lanes[-1], lane_index
            ):
                possible_lanes.pop(-1)
        except IndexError:
            possible_lanes = self.get_initial_possible_lanes(leg_name)
            lane_index = 0
        return lane_index, possible_lanes

    def get_min_index(self, lane_list: list[LaneType], lane_type: LaneType) -> int:
        """Get the minimum index of the lane_type in the lane_list"""
        return lane_list.index(lane_type)

    def get_max_index(self, lane_list: list[LaneType], lane_type: LaneType) -> int:
        """Get the maximum index of the lane_type in the lane_list"""
        return len(lane_list) - 1 - lane_list[::-1].index(lane_type)

    def backward_pass(
        self, seen_vehicles: list[TrackedVehicle], leg_name: str, lanes: list[LaneType]
    ):
        """
        Go through the vehicles in reverse and remove lanes that are not possible.
        This time it is by the possible lanes ordered left to right
        """

        # Go through the vehicles in reverse excluding the last vehicle
        try:
            last_vehicle = seen_vehicles[-1]
        except IndexError:
            return
        lane_index = len(lanes) - 1
        possible_lanes = self.get_initial_possible_lanes(leg_name)
        for vehicle in seen_vehicles[-2::-1]:
            distance = vehicle.distance

            # Update the lane index and possible lanes
            lane_index, possible_lanes = self.update_lane_info_backward(
                leg_name, lanes, distance, last_vehicle, lane_index, possible_lanes
            )

            # Remove lanes from the lanes we know the vehicle is not in
            for lane in vehicle.possible_lanes[::-1]:
                if lane in possible_lanes:
                    break
                vehicle.possible_lanes.remove(lane)

            for lane in possible_lanes[::-1]:
                if lane in vehicle.possible_lanes:
                    break
                possible_lanes.remove(lane)

            last_vehicle = vehicle

    def forward_pass(
        self, leg_name: str, lanes: list[LaneType], vehicles: dict
    ) -> list[TrackedVehicle]:
        possible_lanes = self.get_initial_possible_lanes(leg_name)
        lane_index: int = 0
        last_vehicle: TrackedVehicle | None = None
        seen_vehicles = []
        for vehicle in vehicles:
            try:
                distance = vehicle["distance"]
            except:
                distance = vehicle["distance_to_stop"]

            speed = vehicle["speed"]

            if last_vehicle:
                lane_index, possible_lanes = self.update_lane_info_forward(
                    leg_name, lanes, distance, last_vehicle, lane_index, possible_lanes
                )

            tracked_vehicle = self.find_and_update_vehicle(
                leg_name, distance, possible_lanes, speed
            )

            if not tracked_vehicle:
                continue

            # Remove lanes from the lanes we know the vehicle is not in
            # Only up to the first possible lane of the vehice
            for lane in possible_lanes:
                if lane in tracked_vehicle.possible_lanes:
                    break
                possible_lanes.remove(lane)

            last_vehicle = tracked_vehicle
            seen_vehicles.append(tracked_vehicle)
        return seen_vehicles

    def update_vehicles_for_leg(
        self, leg_name: str, lanes: list[LaneType], vehicles: list[TrackedVehicle]
    ) -> list[TrackedVehicle]:
        seen_vehicles = self.forward_pass(leg_name, lanes, vehicles)
        self.backward_pass(seen_vehicles, leg_name, lanes)
        self.tracked_vehicles[leg_name] = seen_vehicles
        return seen_vehicles

    def update_vehicles(self, vehicles: dict) -> dict[str, list[TrackedVehicle]]:
        for leg_name, v in vehicles.items():
            leg = self.get_lanes(leg_name)
            self.update_vehicles_for_leg(leg_name, leg, v)
        return self.tracked_vehicles


# Should be able to say the last is left

# names = ["N_1"]
# # Mock data for testing
# lane_types = [
#     LaneType.STRAIGHT,
#     LaneType.STRAIGHT,
#     LaneType.LEFT,
# ]
# legs = [InternalLeg(name=n, lanes=lane_types, groups=[], segments=[]) for n in names]


# # Mock environment
# class MockSumoEnv(SumoEnv):
#     def __init__(self):
#         self.legs = legs


# # Initialize LaneTracker with one leg
# env = MockSumoEnv()
# lane_tracker = LaneTracker(env, intersection_name="test_intersection")

# lane_tracker.tracked_vehicles = {n: [] for n in names}

# # Test input
# vehicle_data = {
#     n: [
#         {
#             "waiting_time": 0.0,
#             "speed": 17.51432202709839,
#             "distance": 57.1979066106025,
#             "possible_lanes": ["STRAIGHT"],
#         },
#         {
#             "waiting_time": 0.0,
#             "speed": 4.910187832276778,
#             "distance": 19.801399759173023,
#             "possible_lanes": ["STRAIGHT"],
#         },
#         {
#             "waiting_time": 0.0,
#             "speed": 0.3990962447020697,
#             "distance": 10.625561190391124,
#             "possible_lanes": ["STRAIGHT"],
#         },
#         {
#             "waiting_time": 0.0,
#             "speed": 0.37460644356474504,
#             "distance": 2.738467667314268,
#             "possible_lanes": ["STRAIGHT"],
#         },
#         {
#             "waiting_time": 0.0,
#             "speed": 1.752336993228365,
#             "distance": 12.350705342717589,
#             "possible_lanes": ["STRAIGHT"],
#         },
#         {
#             "waiting_time": 0.0,
#             "speed": 1.9255959058372487,
#             "distance": 2.6521700728480653,
#             "possible_lanes": ["STRAIGHT"],
#         },
#         {
#             "waiting_time": 0.0,
#             "speed": 10.011907320282761,
#             "distance": 53.11803491896451,
#             "possible_lanes": ["LEFT"],
#         },
#         {
#             "waiting_time": 1.0,
#             "speed": 0.2447285619662552,
#             "distance": 37.635628050776546,
#             "possible_lanes": ["LEFT"],
#         },
#         {
#             "waiting_time": 0.0,
#             "speed": 0.0004318723134917394,
#             "distance": 30.10465347567967,
#             "possible_lanes": ["LEFT"],
#         },
#         {
#             "waiting_time": 6.0,
#             "speed": 0.0,
#             "distance": 22.60337154524315,
#             "possible_lanes": ["LEFT"],
#         },
#         {
#             "waiting_time": 11.0,
#             "speed": 0.0,
#             "distance": 15.102348996698566,
#             "possible_lanes": ["LEFT"],
#         },
#         {
#             "waiting_time": 17.0,
#             "speed": 0.0,
#             "distance": 7.6013372044879475,
#             "possible_lanes": ["LEFT"],
#         },
#         {
#             "waiting_time": 20.0,
#             "speed": 0.0,
#             "distance": 0.10027895434842549,
#             "possible_lanes": ["STRAIGHT", "LEFT"],
#         },
#     ]
#     for n in names
# }

# # Add vehicle and update tracker
# lane_tracker.update_vehicles(vehicle_data)

# # Check the tracked vehicle
# for vehicle in lane_tracker.tracked_vehicles["N_1"]:
#     print(vehicle.model_dump_json(indent=4))
