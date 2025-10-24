# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json

import paho.mqtt.client as mqtt
import requests

from app.config import Settings


class TirexMQTTClient:
    def __init__(self, config: Settings):
        self.config: Settings = config

        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        if config.mqtt_broker_username is not None:
            self.client.username_pw_set(username=config.mqtt_broker_username, password=config.mqtt_broker_password)

        self.mqtt_topics_to_subscribe = [config.mqtt_topic_forecast]

        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            id, context, prediction_length = payload["id"], payload["context"], payload["prediction_length"]
            quantiles, mean = self.predict(context, prediction_length)

            message = {"id": id, "mean": mean, "quantiles": quantiles}
            self.client.publish(self.config.mqtt_topic_forecast_result, json.dumps(message))
        except Exception as e:
            # TODO: give an appropiate message when the http backend is offline!
            print(f"Error processing message: {e}")
            message = {"id": id, "error": str(e)}
            self.client.publish(self.config.mqtt_topic_forecast_error, json.dumps(message))

    def predict(self, context, prediction_length):
        response = requests.post(
            f"http://{self.config.http_host}:{self.config.http_port}/forecast/quantiles",
            json={"context": context, "prediction_length": prediction_length},
        )

        quantiles = response.json()

        mean_quantile_index = 4  # index of the 0.5 quantile out of the 9 quantiles
        mean = [[q[mean_quantile_index] for q in ts] for ts in quantiles]

        return quantiles, mean

    def connect(self, keepalive=60):
        try:
            print(f"Connecting to MQTT broker at {self.config.mqtt_broker_host}:{self.config.mqtt_broker_port}")
            self.client.connect(self.config.mqtt_broker_host, self.config.mqtt_broker_port, keepalive)
            self.client.loop_forever()
        finally:
            self.disconnect()

    def disconnect(self):
        self.client.disconnect()
        print("MQTT client disconnected")

    def on_connect(self, client, userdata, connect_flags, reason_code, properties):
        if reason_code == 0:
            print("Connected to MQTT broker")
            for topic in self.mqtt_topics_to_subscribe:
                client.subscribe(topic)
        else:
            print(f"Failed to connect to MQTT broker with code: {reason_code}")

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        if reason_code != 0:
            print(f"Unexpected disconnection from MQTT broker with code: {reason_code}")
