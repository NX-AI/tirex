# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import time

import torch

from tirex import load_model


def measure_model_execution_time(model):
    context = torch.randn((1, 128))

    _, __ = model.forecast(context, prediction_length=32)  # warmup

    start = time.time()
    _, __ = model.forecast(context, prediction_length=32)
    end = time.time()

    return end - start


def test_compileable():
    model = load_model("NX-AI/TiRex", backend="torch", compile=True)

    context = torch.randn((1, 128))
    _, mean = model.forecast(context, prediction_length=32)

    assert mean.shape == (1, 32)


def test_compiled_faster():
    model_uncompiled = load_model("NX-AI/TiRex", backend="torch", compile=False)
    model_compiled = load_model("NX-AI/TiRex", backend="torch", compile=True)

    time_uncompiled = measure_model_execution_time(model_uncompiled)
    time_compiled = measure_model_execution_time(model_compiled)

    assert time_compiled < time_uncompiled, "Compiled model has to be faster than uncompiled one!"
