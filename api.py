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


converter = RequestConverter()

lane_tracker = LaneTracker()

model = RulebasedModel()


@app.post("/predict", response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(
    request: TrafficSimulationPredictRequestDto,
):

    obs = converter.convert_vehicles(request.vehicles)
    if request.simulation_ticks == 1:
        converter.reset()
        lane_tracker.reset("intersection_2", obs["legs"])
        model.reset(lane_tracker, obs["legs"])

    lane_tracker.update_vehicles(obs["vehicles"])
    action = model.get_action(obs)
    signals = converter.convert_signals(action)
    response = TrafficSimulationPredictResponseDto(signals=signals)

    return response


if __name__ == "__main__":

    uvicorn.run("api:app", host=HOST, port=PORT)
