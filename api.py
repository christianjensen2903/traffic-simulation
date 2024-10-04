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


def get_converter() -> RequestConverter:
    return converter


@app.post("/predict", response_model=TrafficSimulationPredictResponseDto)
def predict_endpoint(
    request: TrafficSimulationPredictRequestDto,
    # converter: Annotated[RequestConverter, Depends(get_converter)],
):

    obs = converter.convert_vehicles(request.vehicles)
    # print(obs)/
    if request.simulation_ticks < 20:
        print(request.simulation_ticks)

    # Return the encoded image to the validation/evalution service
    response = TrafficSimulationPredictResponseDto(signals=[])

    return response


if __name__ == "__main__":

    uvicorn.run("api:app", host=HOST, port=PORT)
