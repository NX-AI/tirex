# TiRex: Zero-Shot Forecasting across Long and Short Horizons

[Paper](https://arxiv.org/abs/2505.23719) | [TiRex Huggingface Model Card](https://huggingface.co/NX-AI/TiRex)


This repository provides the pre-trained forecasting model TiRex introduced in the paper
[TiRex: Zero-Shot Forecasting across Long and Short Horizons with Enhanced In-Context Learning](https://arxiv.org/abs/2505.23719).


## TiRex Model

TiRex is a 35M parameter pre-trained time series forecasting model based on [xLSTM](https://github.com/NX-AI/xlstm).

### Key Facts:

- **Zero-Shot Forecasting**:
  TiRex is a pre-trained model that performs time series forecasting without requiring any training on your data. Simply download and use it.

- **Quantile Predictions**:
  TiRex provides both point estimates and quantile estimates.

- **State-of-the-art Performance over Long and Short Horizons**:
  TiRex achieves top scores in various time series forecasting benchmarks, see [GiftEval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [ChronosZS](https://huggingface.co/spaces/autogluon/fev-leaderboard).
  These benchmark show that TiRex provides great performance for both long and short-term forecasting.

## Installation

```sh
pip install tirex-ts
```

Install with additional input/output adapter:
```sh
pip install 'tirex-ts[gluonts,hfdataset]'
```

TiRex is currently only tested on Linux and MacOS.

## Quick Start

```python
import torch
from tirex import load_model, ForecastModel

model: ForecastModel = load_model("NX-AI/TiRex")
data = torch.rand((5, 128))  # Sample Data (5 time series with length 128)
quantiles, mean = model.forecast(context=data, prediction_length=64)
```

We provide an extended quick start example in [examples/quick_start_tirex.ipynb](./examples/quick_start_tirex.ipynb).
This notebook also shows how to use the different input and output types of you time series data.
You can also run it in [Google Colab](https://colab.research.google.com/github/NX-AI/tirex/blob/main/examples/quick_start_tirex.ipynb).

We provide notebooks to run the benchmarks: [GiftEval](./examples/gifteval/gifteval.ipynb) and [Chronos-ZS](./examples/chronos_zs/chronos_zs.ipynb).

## Finetuning TiRex
TiRex already provide state-of-the-art performance for zero-shot prediction. Hence, you can use it without training on your own data.

If you are interested in models fine-tuned on your data or with different pretraining, please contact us at [contact@nx-ai.com](mailto:contact@nx-ai.com)


## CUDA Kernels

Tirex can use custom CUDA kernels for the sLSTM cells.
These CUDA kernels are compiled when the model is loaded the first time.
The CUDA kernels require GPU hardware that support CUDA compute capability 8.0 or later.
We also highly suggest to use the provided [conda environment spec](./requirements_py26.yaml).
The CUDA kernels are automatically used when the xlstm package is installed.

Create a conda environment:
```sh
conda create --file requirements_gpu.yaml
conda activate tirex
```

To install TiRex with the CUDA kernels run:
```sh
pip install 'tirex-ts[cuda,gluonts,hfdataset]'
```

Explicitly set the custom CUDA backend:
```python
model = load_model("NX-AI/TiRex", backend="cuda")
```

## Installation (Docker)
If you prefer running TiRex in containers, we provide ready-to-use Docker services for both GPU and CPU.

Prerequisites:
- Docker and Docker Compose installed
- For GPU: NVIDIA driver and NVIDIA Container Toolkit (to enable `runtime: nvidia`)

Volumes and ports:
- Local `./examples` and `./data` are mounted into the container at `/app/examples` and `/app/data`.
- GPU service publishes Jupyter on port `8888`; CPU service publishes on `8889`.

Start the CPU container and launch Jupyter Notebook:
```bash
docker build -t tirex:cpu -f Dockerfile.cpu . && docker run --rm -d --name tirex-cpu -p 8889:8888 -v "$(pwd)/examples":/app/examples -v "$(pwd)/data":/app/data tirex:cpu bash -lc "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
```

Start the GPU container and launch Jupyter Notebook (check your gpu capability number with this [link](https://developer.nvidia.com/cuda-gpus)):
```bash
docker build -t tirex:gpu -f Dockerfile.gpu . && docker run -e TORCH_CUDA_ARCH_LIST="9.0" --gpus all --rm -d --name tirex-gpu -p 8888:8888 -v "$(pwd)/examples":/app/examples -v "$(pwd)/data":/app/data tirex:gpu bash -lc "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''"
```

Access in your browser:
- GPU: http://localhost:8888
- CPU: http://localhost:8889


### Troubleshooting CUDA

**This information is taken from the
[xLSTM repository](https://github.com/NX-AI/xlstm) - See this for further details**:

For the CUDA version of sLSTM, you need to specify Compute Capability, see [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). Or just specify a range as in the example below:
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
```

For all kinds of custom setups with torch and CUDA, keep in mind that versions have to match. Also, to make sure the correct CUDA libraries are included you can use the `XLSTM_EXTRA_INCLUDE_PATHS` environment variable now to inject different include paths, for example:

```bash
export XLSTM_EXTRA_INCLUDE_PATHS='/usr/local/include/cuda/:/usr/include/cuda/'
```

or within python:

```python
import os
os.environ['XLSTM_EXTRA_INCLUDE_PATHS']='/usr/local/include/cuda/:/usr/include/cuda/'
```


## Cite

If you use TiRex in your research, please cite our work:

```bibtex
@article{auerTiRexZeroShotForecasting2025,
  title = {{{TiRex}}: {{Zero-Shot Forecasting Across Long}} and {{Short Horizons}} with {{Enhanced In-Context Learning}}},
  author = {Auer, Andreas and Podest, Patrick and Klotz, Daniel and B{\"o}ck, Sebastian and Klambauer, G{\"u}nter and Hochreiter, Sepp},
  journal = {ArXiv},
  volume = {2505.23719},
  year = {2025}
}
```


## License

TiRex is licensed under the [NXAI community license](./LICENSE).
