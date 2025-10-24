# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import requests
import torch


def test_http_mean(api_server):
    prediction_length = 2
    context = [[0.0, 1.0, 2.0, 3.0]]
    response = requests.post(
        f"{api_server}/forecast/mean",
        json={"context": context, "prediction_length": prediction_length},
    )

    assert response.status_code == 200
    data = response.json()

    data = torch.tensor(data, dtype=torch.float32)
    data_ref = torch.tensor([[3.751096248, 4.562105178]], dtype=torch.float32)
    # bfloat16 tolerances to allow for small differences between CPU and CUDA
    torch.testing.assert_close(data, data_ref, rtol=1.6e-2, atol=1e-5)


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
