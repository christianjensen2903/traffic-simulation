from multiprocessing import Process, Queue
from time import sleep, time
import random

from environment import load_and_run_simulation


def run_game():

    test_duration_seconds = 600
    is_random = True
    configuration_file = "intersections/2/configuration.yaml"
    start_time = time()

    input_queue = Queue()
    output_queue = Queue()
    error_queue = Queue()
    errors = []

    p = Process(
        target=load_and_run_simulation,
        args=(
            configuration_file,
            start_time,
            test_duration_seconds,
            is_random,
            input_queue,
            output_queue,
            error_queue,
        ),
    )

    p.start()

    # Wait for the simulation to start
    sleep(0.2)

    # For logging
    actions = {}

    while True:

        state = output_queue.get()

        if state.is_terminated:
            p.join()
            break

        # Insert your own logic here to parse the state and
        # select the next action to take

        # print(f"Vehicles: {state.vehicles}")
        # print(f"Signals: {state.signals}")

        signal_logic_errors = None
        prediction = {}
        prediction["signals"] = []

        # Update the desired phase of the traffic lights
        next_signals = {signal.name: "green" for signal in state.signals}
        # Pick a random signal and set it to green
        # random_signal = random.choice(state.signals)
        # print(f"Changing {random_signal.name} to green")
        # next_signals[random_signal.name] = "green"

        # next_signals = {}
        current_tick = state.simulation_ticks

        for signal in prediction["signals"]:
            actions[current_tick] = (signal["name"], signal["state"])
            next_signals[signal["name"]] = signal["state"]

        signal_logic_errors = input_queue.put(next_signals)

        if signal_logic_errors:
            print(f"Signal logic errors: {signal_logic_errors}")
            errors.append(signal_logic_errors)

        # Wait 12 seconds
        sleep(20)

    # End of simulation, return the score

    # Transform the score to the range [0, 1]
    if state.total_score == 0:
        state.total_score = 1e9

    inverted_score = 1.0 / state.total_score

    return inverted_score


if __name__ == "__main__":
    run_game()
