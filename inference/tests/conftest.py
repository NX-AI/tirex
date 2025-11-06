# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
import subprocess
import time

import pytest
import requests
import torch

start_server = int(os.getenv("TEST_START_SERVER", "1")) == 1
base_host = os.getenv("TEST_HOST", "127.0.0.1")
base_port = int(os.getenv("TEST_PORT", "8002"))
mqtt_host = os.getenv("TEST_MQTT_BROKER_HOST", "broker.emqx.io")
mqtt_port = int(os.getenv("TEST_MQTT_BROKER_PORT", "1883"))


def wait_for_api(healthcheck_url, timeout=30):
    print("Start for model load")
    for i in range(timeout):
        try:
            response = requests.get(healthcheck_url)
            if response.status_code == 200:
                print(f"Connected in {i} seconds!")
                return
        except:
            pass
        time.sleep(1)

    raise TimeoutError(f"Can't connect to {healthcheck_url} in {timeout} seconds!")


@pytest.fixture(scope="session")
def api_server():
    base_url = f"http://{base_host}:{base_port}"
    server_command = f"HTTP_HOST={base_host} HTTP_PORT={base_port} MQTT_ENABLED=1 MQTT_BROKER_HOST={mqtt_host} MQTT_BROKER_PORT={mqtt_port} python -m app.main > test-server.log"

    try:
        process = None
        if start_server:
            # Can't redirect stdout to subprocess.PIPE, so we write it into the test-server.log.
            process = subprocess.Popen(server_command, stdout=None, shell=True)
        wait_for_api(f"{base_url}/health", timeout=30)

        yield base_url
    finally:
        if process is not None:
            process.kill()
            process.wait()


def get_default_context():
    prediction_length = 2
    context = [[0.0, 1.0, 2.0, 3.0]]
    return context, prediction_length


def assert_default_prediction_correct(forecast: list[float]):
    data = torch.tensor(forecast, dtype=torch.float32)
    data_ref = torch.tensor([[3.751096248, 4.562105178]], dtype=torch.float32)
    # bfloat16 tolerances to allow for small differences between CPU and CUDA
    torch.testing.assert_close(data, data_ref, rtol=1.6e-2, atol=1e-5)
