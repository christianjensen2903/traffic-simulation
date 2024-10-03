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

    if simulation_ticks == 30:
        with open(f"vehicles_per_hour.json", "r+") as f:
            vehicles_per_hour_data = json.load(f)
            sim2 = "A1RightTurn" in signal_groups
            grp = "2" if sim2 else "1"
            for leg in legs:
                vehicle_count = len(
                    [vehicle for vehicle in vehicles if vehicle.leg == leg.name]
                )
                if leg.name not in vehicles_per_hour_data[grp]:
                    vehicles_per_hour_data[grp][leg.name] = []
                vehicles_per_hour_data[grp][leg.name].append(vehicle_count / 30 * 3600)
            f.seek(0)
            f.truncate()
            json.dump(vehicles_per_hour_data, f)

    # # Read the vehicles
    # with open(f"threshold.json", "r") as f:
    #     threshold = json.load(f)

    # threshold[simulation_ticks] = [
    #     vehicle.model_dump(mode="json") for vehicle in vehicles
    # ]

    # with open(f"threshold.json", "w") as f:
    #     json.dump(threshold, f)

    logger.info(f"Number of vehicles at tick {simulation_ticks}: {len(vehicles)}")

    # Select a signal group to go green
    # green_signal_group = signal_groups[0]

    # if simulation_ticks > 100 and simulation_ticks < 120:
    #     signals = [
    #         SignalDto(name="A1", state="green"),
    #         SignalDto(name="A2", state="green"),
    #     ]
    # elif simulation_ticks > 120 and simulation_ticks < 150:
    #     signals = [SignalDto(name="A1", state="red"), SignalDto(name="A2", state="red")]
    # elif simulation_ticks > 150 and simulation_ticks < 170:
    #     signals = [SignalDto(name="B1", state="green")]
    # elif simulation_ticks > 170 and simulation_ticks < 190:
    #     signals = [SignalDto(name="B1", state="red")]
    # elif simulation_ticks > 190 and simulation_ticks < 210:
    #     signals = [SignalDto(name="B2", state="green")]

    # Return the encoded image to the validation/evalution service
    response = TrafficSimulationPredictResponseDto(signals=signals)

    return response


if __name__ == "__main__":

    uvicorn.run("api:app", host=HOST, port=PORT)
