# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
import time
from queue import Queue
from uuid import uuid4

import paho.mqtt.client as mqtt
import pytest
from conftest import assert_default_prediction_correct, get_default_context, mqtt_host, mqtt_port
from paho.mqtt.client import MQTTMessage

connect_timeout = 30
test_timeout = 120


@pytest.fixture(scope="module")
def mqtt_client():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(mqtt_host, mqtt_port, 60)
        client.loop_start()

        for _ in range(connect_timeout):
            if client.is_connected():
                break
            time.sleep(1)

        if client.is_connected() is False:
            raise ConnectionRefusedError(f"Failed to connect to MQTT broker at {mqtt_host}:{mqtt_port}")

    except Exception as e:
        pytest.fail(f"MQTT Broker connection failed: {e}")

    yield client

    client.loop_stop()
    client.disconnect()


@pytest.fixture
def message_listener(mqtt_client):
    # Fixture that saves received mqtt messages to a queue
    message_queue = Queue()

    def on_message(client, userdata, msg):
        message_queue.put(msg)

    mqtt_client.on_message = on_message

    yield mqtt_client, message_queue

    mqtt_client.on_message = None


def test_mqtt(message_listener, api_server):
    client, message_queue = message_listener
    client.subscribe("tirex/forecast/result")

    id = str(uuid4())
    context, prediction_length = get_default_context()
    msg = {"id": id, "context": context, "prediction_length": prediction_length}

    client.publish("tirex/forecast/request", json.dumps(msg))

    msg: MQTTMessage = message_queue.get(timeout=test_timeout)

    assert msg.topic == "tirex/forecast/result"

    payload = json.loads(msg.payload.decode())
    assert payload["id"] == id

    assert_default_prediction_correct(payload["mean"])
