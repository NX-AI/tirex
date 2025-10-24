# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import Settings

from tirex import ForecastModel, load_model

settings = Settings()


model: ForecastModel = load_model(
    settings.model_path, device=settings.model_device, backend="torch", compile=settings.model_compile
)

if settings.model_compile:
    print("Compile the model. That might take over 2 minutes...")
    _, __ = model.forecast(context=[list(range(2048))], prediction_length=32)
    print("Compilation done.")

app = FastAPI(title="Tirex API")


@app.exception_handler(Exception)
async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error_code": 500, "error_message": exc.__str__()})


class Forecast(BaseModel):
    context: list[list[float]] = [[0, 1, 2, 3]]
    prediction_length: int = 32


@app.post("/forecast/mean")
async def predict(forecast: Forecast) -> list[list[float]]:
    context = torch.tensor(forecast.context, dtype=torch.float32)
    _, mean = model.forecast(context=context, prediction_length=forecast.prediction_length)
    return mean.tolist()


@app.post("/forecast/quantiles")
async def predict(forecast: Forecast) -> list[list[list[float]]]:
    context = torch.tensor(forecast.context, dtype=torch.float32)
    quantiles, _ = model.forecast(context=context, prediction_length=forecast.prediction_length)
    return quantiles.tolist()


@app.get("/health")
def health():
    return {"message": "OK"}
