# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import requests
import torch
from conftest import assert_default_prediction_correct, get_default_context


def test_http_mean(api_server):
    prediction_length = 2
    context = [[0.0, 1.0, 2.0, 3.0]]
    context, prediction_length = get_default_context()
    response = requests.post(
        f"{api_server}/forecast/mean",
        json={"context": context, "prediction_length": prediction_length},
    )

    assert response.status_code == 200
    data = response.json()
    assert_default_prediction_correct(data)


def test_http_quantiles(api_server):
    prediction_length = 2
    context = [[0.0, 1.0, 2.0, 3.0]]
    response = requests.post(
        f"{api_server}/forecast/quantiles",
        json={"context": context, "prediction_length": prediction_length},
    )

    assert response.status_code == 200
    data = response.json()

    data = torch.tensor(data, dtype=torch.float32)
    assert data.shape == (1, 2, 9)
