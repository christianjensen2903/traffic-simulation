import requests
import time
import json
from tqdm import tqdm

# Constants
TOKEN = "bb0edcfeea314b2580cee51219c82cf0"
BASE_URL = "https://cases.dmiai.dk/api/v1/usecases/traffic-simulation/validate/queue"
SERVICE_URL = "https://2ac8-147-78-30-67.ngrok-free.app" + "/predict"


def queue_validation_attempt() -> str:
    headers = {"x-token": TOKEN}
    data = {"url": SERVICE_URL}
    response = requests.post(BASE_URL, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        queued_attempt_uuid = result.get("queued_attempt_uuid")
        print(f"Validation attempt queued with UUID: {queued_attempt_uuid}")
        return queued_attempt_uuid
    else:
        raise (
            f"Error queuing validation attempt: {response.status_code} {response.text}"
        )


def check_attempt_status(queued_attempt_uuid: str) -> float:
    headers = {"x-token": TOKEN}
    status_url = f"{BASE_URL}/{queued_attempt_uuid}"
    while True:
        response = requests.get(status_url, headers=headers)

        if response.status_code != 200:
            raise (
                f"Error checking attempt status: {response.status_code} {response.text}"
            )

        result = response.json()
        status = result.get("status")
        print(f"Attempt status: {status}")
        if status == "done":
            print("Validation attempt completed.")

            return
        else:
            # Wait for a few seconds before checking again
            time.sleep(250)


def main():
    while True:
        queued_attempt_uuid = queue_validation_attempt()
        if not queued_attempt_uuid:
            return
        check_attempt_status(queued_attempt_uuid)


if __name__ == "__main__":
    main()
