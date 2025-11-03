# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import math
import os
import time

import datasets
import fev
import pytest

from tirex import ForecastModel, load_model
from tirex.util import select_quantile_subset


def geometric_mean(s):
    return math.prod(s) ** (1 / len(s))


def eval_task(model, task):
    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows(trust_remote_code=True):
        past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        past_data = past_data.with_format("torch").cast_column("target", datasets.Sequence(datasets.Value("float32")))
        loaded_targets = [t for t in past_data["target"]]

        start_time = time.monotonic()
        quantiles, means = model.forecast(loaded_targets, prediction_length=task.horizon)
        inference_time += time.monotonic() - start_time

        predictions_dict = {"predictions": means}
        quantiles_subset = select_quantile_subset(quantiles, task.quantile_levels)
        for idx, level in enumerate(task.quantile_levels):
            predictions_dict[str(level)] = quantiles_subset[:, :, idx]

        predictions_per_window.append(
            fev.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
            )
        )

    return predictions_per_window, inference_time


@pytest.fixture
def tirex_model() -> ForecastModel:
    return load_model("NX-AI/TiRex")


@pytest.fixture
def benchmark():
    url = "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/tasks.yaml"
    return fev.Benchmark.from_yaml(url)


def test_chronos_single(tirex_model, benchmark):
    task_name = "monash_australian_electricity"
    task = [task for task in benchmark.tasks if task.dataset_config == task_name][0]
    predictions, inference_time = eval_task(tirex_model, task)
    evaluation_summary = task.evaluation_summary(
        predictions,
        model_name="TiRex",
        inference_time_s=inference_time,
    )

    assert evaluation_summary["WQL"] < 0.055, "WQL on the electricity task needs to be less than 0.055"
    assert evaluation_summary["MASE"] < 0.99, "MASE on the electricity task needs to be less than 0.99"


@pytest.mark.skipif(
    os.getenv("CI") is not None and os.getenv("CI_RUN_BENCHMARKS") is None,
    reason="Skip Chronos benchmarks in CI",
)
def test_chronos_all(tirex_model, benchmark):
    tasks_wql = []
    tasks_mase = []
    for task in benchmark.tasks:
        predictions, inference_time = eval_task(tirex_model, task)
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name="TiRex",
            inference_time_s=inference_time,
        )
        tasks_wql.append(evaluation_summary["WQL"])
        tasks_mase.append(evaluation_summary["MASE"])

    # Calculated from the geometric mean of the WQL and MASE data of the seasonal_naive model
    # https://github.com/autogluon/fev/blob/main/benchmarks/chronos_zeroshot/results/seasonal_naive.csv
    agg_wql_baseline = 0.1460642781226389
    agg_mase_baseline = 1.6708210897174531

    agg_wql = geometric_mean(tasks_wql)
    agg_mase = geometric_mean(tasks_mase)

    print(f"WQL: {agg_wql / agg_wql_baseline:.3f}")
    print(f"MASE: {agg_mase / agg_mase_baseline:.3f}")

    tolerance = 0.01

    # Values from Tirex paper: https://arxiv.org/pdf/2505.23719
    assert agg_wql / agg_wql_baseline < 0.59 + tolerance, "WQL on chromos needs to be less than 0.60"
    assert agg_mase / agg_mase_baseline < 0.78 + tolerance, "MASE on chromos needs to be less than 0.79"
