# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_path: str = "NX-AI/TiRex"
    model_device: str = "cpu"
    model_compile: int = 0

    http_port: int = 8000
    http_host: str = "0.0.0.0"

    mqtt_enabled: int = 0
    mqtt_broker_host: str | None = None
    mqtt_broker_port: int | None = None
    mqtt_broker_username: str | None = None
    mqtt_broker_password: str | None = None
    mqtt_topic_forecast: str = "tirex/forecast/request"
    mqtt_topic_forecast_result: str = "tirex/forecast/result"
    mqtt_topic_forecast_error: str = "tirex/forecast/error"
