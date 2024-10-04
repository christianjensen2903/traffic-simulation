import uvicorn
from fastapi import FastAPI, Depends
import datetime
import time
from loguru import logger
from pydantic import BaseModel
from sim.dtos import (
    TrafficSimulationPredictResponseDto,
    TrafficSimulationPredictRequestDto,
    SignalDto,
)
from typing import Annotated
import json
from api_env import RequestConverter
from lane_tracker import LaneTracker
from generate_road import LaneType
from rulebased import RulebasedModel

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




initial_legs = {
    "N": [LaneType.STRAIGHT, LaneType.STRAIGHT, LaneType.LEFT],
    "S": [LaneType.STRAIGHT, LaneType.STRAIGHT, LaneType.LEFT],
    "W": [
        LaneType.STRAIGHT,
        LaneType.STRAIGHT,
        LaneType.LEFT,
        LaneType.LEFT,
        LaneType.LEFT,
    ],
    "E": [LaneType.STRAIGHT, LaneType.STRAIGHT, LaneType.LEFT],
}

second_legs = {
    "N": [LaneType.RIGHT, LaneType.STRAIGHT, LaneType.LEFT],
    "S": [LaneType.RIGHT, LaneType.STRAIGHT, LaneType.LEFT],
    "W": [LaneType.ALL],
    "E": [LaneType.RIGHT, LaneType.STRAIGHT_LEFT],
}

converter = RequestConverter(initial_legs)

lane_tracker = LaneTracker(intersection_name="intersection_1", legs=initial_legs)

model = RulebasedModel(lane_tracker, initial_legs)

is_first = True
@app.post("/predict", response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(
    request: TrafficSimulationPredictRequestDto,
):

    obs = converter.convert_request(request)
    if request.simulation_ticks == 1:
        converter.reset(second_legs)
        lane_tracker.reset("intersection_2", second_legs)
        model.reset(lane_tracker, second_legs)

    lane_tracker.update_vehicles(obs["vehicles"])
    action = model.get_action(obs)
    signals = converter.convert_action(action)
    response = TrafficSimulationPredictResponseDto(signals=signals)
    print(response)

    return response


if __name__ == "__main__":

    uvicorn.run("api:app", host=HOST, port=PORT)
