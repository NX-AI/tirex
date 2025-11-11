import argparse
import copy
import gc
import os
import time
from datetime import date

import pandas as pd
import torch

from tirex import ForecastModel, load_model


def measure_time_wrapper(fn, name, repeats=1, is_cuda=False):
    total_time = 0
    for i in range(repeats):
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        # Some time for gc.collect() and cuda.empty_cache()
        time.sleep(0.1)
        start = time.time()
        out = fn()
        if is_cuda:
            torch.cuda.synchronize()
        end = time.time()
        total_time += end - start
    runtime = total_time / repeats
    print(f"{name} time: {runtime:.6f}")
    return out, runtime


def log_result(out_path, data):
    file_exists = os.path.isfile(out_path)
    pd.DataFrame([data]).to_csv(out_path, mode="a" if file_exists else "w", header=not file_exists, index=False)


def benchmark(args):
    print(args)
    torch.manual_seed(args.seed)
    model: ForecastModel = load_model(
        args.model, backend=args.backend, device=args.device, compile=args.compile == "True"
    )

    today = f"{date.today().year}-{date.today().month}-{date.today().day}"
    model_name = args.model.replace("NX-AI/", "")
    name = f"{model_name}-{args.backend}-{args.device}{'' if args.compile == 'True' else '-uncompiled'}-{args.hardware}-{today}".lower()
    out_path = f"{args.result_base_dir}/{name}.csv"

    if os.path.exists(out_path):
        os.remove(out_path)

    for batch_size in args.batch_sizes:
        for context_length in args.context_lengths:
            for prediction_length in args.prediction_lengths:
                context = torch.randn((batch_size, context_length), device=args.device)

                _, _ = model.forecast(context, prediction_length=prediction_length)  # warmup

                run_name = f"{batch_size}-{context_length}-{prediction_length}"
                _, time = measure_time_wrapper(
                    lambda: model.forecast(context, prediction_length=prediction_length),
                    name=run_name,
                    repeats=args.num_repeats,
                    is_cuda=args.device == "cuda",
                )
                throughput = batch_size / time

                log_result(
                    out_path,
                    {
                        "date": today,
                        "model": args.model,
                        "backend": args.backend,
                        "device": args.device,
                        "compiled": args.compile,
                        "hardware": args.hardware,
                        "batch_size": batch_size,
                        "context_length": context_length,
                        "prediction_length": prediction_length,
                        "time": round(time, 3),
                        "throughput": round(throughput, 3),
                    },
                )


def benchmark_all(args):
    def override_args(backend, device, compile):
        argx = copy.copy(args)
        argx.backend = backend
        argx.device = device
        argx.compile = compile
        return argx

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")

    for device in devices:
        for compile in ["True", "False"]:
            try:
                benchmark(override_args(backend="torch", device=device, compile=compile))
            except Exception as e:
                print(f"An unexpected error occurred for {device} compile={compile}: {e}")

    if torch.cuda.is_available():
        benchmark(override_args(backend="cuda", device="cuda", compile="False"))


def main():
    parser = argparse.ArgumentParser(prog="Tirex benchmarker")
    parser.add_argument("--all", action="store_true", help="Run all tests for a system with defualt config")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model", default="NX-AI/TiRex")
    parser.add_argument("--hardware", required=True, type=str)
    parser.add_argument("--backend", default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", default="True")
    parser.add_argument("--num_repeats", default=2, type=int)
    parser.add_argument("--batch_sizes", default=[1, 16, 256], type=int, nargs="+")
    parser.add_argument("--prediction_lengths", default=[32, 64, 128], type=int, nargs="+")
    parser.add_argument("--context_lengths", default=[2048], type=int, nargs="+")
    parser.add_argument("--result_base_dir", default="./result/")

    args = parser.parse_args()
    if not args.all:
        benchmark(args)
    else:
        benchmark_all(args)


if __name__ == "__main__":
    main()
