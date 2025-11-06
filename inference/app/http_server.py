# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from pydantic import BaseModel

from app.config import Settings
from app.model import TirexModel

settings = Settings()
model = TirexModel(settings)
model.warmup()


app = FastAPI(title="Tirex API")
print(f"\033[32mINFO\033[0m:     Access Swagger documentation at http://{settings.http_host}:{settings.http_port}/docs")


@app.exception_handler(Exception)
async def api_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error_code": 500, "error_message": exc.__str__()})


class Forecast(BaseModel):
    context: list[list[float]] = [[0, 1, 2, 3]]
    prediction_length: int = 32


@app.post("/forecast/mean")
async def forecast_mean(forecast: Forecast) -> list[list[float]]:
    context = torch.tensor(forecast.context, dtype=torch.float32)
    _, mean = model.predict(context=context, prediction_length=forecast.prediction_length)
    return mean.tolist()


@app.post("/forecast/quantiles")
async def forecast_quantiles(forecast: Forecast) -> list[list[list[float]]]:
    context = torch.tensor(forecast.context, dtype=torch.float32)
    quantiles, _ = model.predict(context=context, prediction_length=forecast.prediction_length)
    return quantiles.tolist()


@app.get("/health")
def health():
    return {"message": "OK"}


mcp = FastMCP("TiRex MCP")

disclaimer = (
    "Disclaimer: NXAI is not responsible for any incorrect interpretations of the "
    "forecasted values by LLMs. Check the TiRex license for more details: "
    "https://github.com/NX-AI/tirex\n\n"
)


@mcp.tool()
async def tirex_model(context: list[float], prediction_length: int) -> str:
    """Use the TiRex model to forecast time series data."""
    # MCP is the only API that isn't batched!

    context = torch.tensor([context], dtype=torch.float32)
    _, mean = model.predict(context=context, prediction_length=prediction_length)
    mean = mean[0].tolist()

    return (
        "TiRex Forecast Results:\n"
        f"Input data length: {len(context)}\n"
        f"Prediction length: {prediction_length}\n\n"
        f"Forecasted values: {mean}\n\n"
        f"{disclaimer}"
    )


mcp_app = mcp.http_app(path="/")
app.router.lifespan_context = mcp_app.router.lifespan_context
app.mount("/mcp", mcp_app)
