import uvicorn
from fastapi import FastAPI
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import (
    TrafficSimulationPredictResponseDto,
    TrafficSimulationPredictRequestDto,
    SignalDto,
)
import json

HOST = "0.0.0.0"
PORT = 9051

app = FastAPI()
start_time = time.time()


@app.get("/api")
def hello():
    return {
        "service": "traffic-simulation-usecase",
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def index():
    return "Your endpoint is running!"


steps_taken = 0


ALL = 0
STRAIGHT = 1
LEFT = 2
RIGHT = 3
STRAIGHT_RIGHT = 4
STRAIGHT_LEFT = 5

N = 0
E = 1
S = 2
W = 3


@app.post("/predict", response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(request: TrafficSimulationPredictRequestDto):
    # Decode request
    data = request
    vehicles = data.vehicles
    total_score = data.total_score
    simulation_ticks = data.simulation_ticks
    signals = data.signals
    signal_groups = data.signal_groups
    legs = data.legs
    allowed_green_signal_combinations = data.allowed_green_signal_combinations
    is_terminated = data.is_terminated

    if steps_taken >= 300:
        intersectionID = 1
    else:
        intersectionID = 0

    obs = {
        "vehicles": [
            {"distance": v.distance_to_stop, "speed": v.speed} for v in vehicles
        ]
    }

    action, algo_state = baseline_algorithm(intersectionID, obs, algo_state)

    # Convert the algorithm's action format to the expected response format
    response = TrafficSimulationPredictResponseDto()

    if intersectionID == 0:
        # ['A1', 'A1LeftTurn', 'A2', 'A2LeftTurn', 'B1', 'B1LeftTurn', 'B2', 'B2LeftTurn']
        # N =  A1: Right (Straight StraightRight) lanes=['Left', 'Main', 'Main']
        # E = B2: Right StraightLeft lanes=['Left', 'Left', 'Left', 'Main', 'Main']
        # S = A2: Right Straight Left lanes=['Left', 'Main', 'Main']
        # W = B1: All lanes=['Left', 'Main', 'Main']

        signals = [
            SignalDto(
                name="A1RightTurn", state="green" if action[N, RIGHT] == 1 else "red"
            ),
            SignalDto(name="A1", state="green" if action[N, STRAIGHT] == 1 else "red"),
            SignalDto(
                name="A1LeftTurn", state="green" if action[N, LEFT] == 1 else "red"
            ),
            SignalDto(
                name="B2RightTurn", state="green" if action[E, RIGHT] == 1 else "red"
            ),
            SignalDto(
                name="B2", state="green" if action[E, STRAIGHT_LEFT] == 1 else "red"
            ),
            SignalDto(
                name="A2RightTurn", state="green" if action[S, RIGHT] == 1 else "red"
            ),
            SignalDto(name="A2", state="green" if action[S, STRAIGHT] == 1 else "red"),
            SignalDto(
                name="A2LeftTurn", state="green" if action[S, LEFT] == 1 else "red"
            ),
            SignalDto(name="B2", state="green" if action[W, ALL] == 1 else "red"),
        ]

    elif intersectionID == 1:
        # ['A1', 'A1RightTurn', 'A1LeftTurn', 'A2', 'A2RightTurn', 'A2LeftTurn', 'B1', 'B1RightTurn', 'B2']
        # N =  A1: Right Straight Left
        # E = B2: Right StraightLeft
        # S = A2: Right Straight Left
        # W = B1: All

        signals = [
            SignalDto(
                name="A1RightTurn", state="green" if action[N, RIGHT] == 1 else "red"
            ),
            SignalDto(name="A1", state="green" if action[N, STRAIGHT] == 1 else "red"),
            SignalDto(
                name="A1LeftTurn", state="green" if action[N, LEFT] == 1 else "red"
            ),
            SignalDto(
                name="B2RightTurn", state="green" if action[E, RIGHT] == 1 else "red"
            ),
            SignalDto(
                name="B2", state="green" if action[E, STRAIGHT_LEFT] == 1 else "red"
            ),
            SignalDto(
                name="A2RightTurn", state="green" if action[S, RIGHT] == 1 else "red"
            ),
            SignalDto(name="A2", state="green" if action[S, STRAIGHT] == 1 else "red"),
            SignalDto(
                name="A2LeftTurn", state="green" if action[S, LEFT] == 1 else "red"
            ),
            SignalDto(name="B2", state="green" if action[W, ALL] == 1 else "red"),
        ]

    response = TrafficSimulationPredictResponseDto(signals=signals)

    steps_taken += 1
    return response


def convert_ambolt_state_to_gym_state(request: TrafficSimulationPredictRequestDto):
    vehicles = [
        {"distance": v.distance_to_stop, "speed": v.speed} for v in request.vehicles
    ]

    return {
        "vehicles": vehicles,
        "signals": signals,
        "legs": legs,
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
