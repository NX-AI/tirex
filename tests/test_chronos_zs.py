from tirex import ForecastModel, load_model
import time
import datasets
import fev
import pytest
import os
import math

def geometric_mean(s):
    return math.prod(s) ** (1/len(s))

def eval_task(model, task):
    past_data, _ = task.get_input_data(trust_remote_code=True)
    quantile_levels = task.quantile_levels
    past_data = past_data.with_format("torch").cast_column(
        task.target_column, datasets.Sequence(datasets.Value("float32"))
    )[task.target_column]
    loaded_data = [t for t in past_data]

    start_time = time.monotonic()
    quantiles, means = model.forecast(loaded_data, quantile_levels=quantile_levels, prediction_length=task.horizon)
    inference_time = time.monotonic() - start_time
    predictions_dict = {"predictions": means}
    for idx, level in enumerate(quantile_levels):
        predictions_dict[str(level)] = quantiles[:, :, idx]  # [num_items, horizon]

    predictions = datasets.Dataset.from_dict(predictions_dict)
    return predictions, inference_time


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


@pytest.mark.skipif(os.getenv("CI"), reason="Skip full chromos benchmarking in the CI")
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
