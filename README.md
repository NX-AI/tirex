<div align="center">

# TiRex: Zero-Shot Forecasting across Long and Short Horizons

[![Paper](https://img.shields.io/static/v1?label=Paper&message=2505.23719&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2505.23719)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-TiRex-yellow?logo=huggingface)](https://huggingface.co/NX-AI/TiRex)
[![PyPI](https://img.shields.io/pypi/v/tirex-ts?color=blue)](https://pypi.org/project/tirex-ts/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tirex-ts?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/tirex-ts)
[![License](https://img.shields.io/badge/license-nxai_community-green)](https://github.com/NX-AI/tirex-internal/blob/main/LICENSE)
[![docs](https://img.shields.io/website-up-down-green-red/http/nx-ai.github.io/tirex//.svg?label=docs)](https://nx-ai.github.io/tirex/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![formatter: docformatter](https://img.shields.io/badge/formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)

</div>


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

## Documentation
A detailed documentation of TiRex can be found here: https://nx-ai.github.io/tirex/

## Installation

```sh
pip install tirex-ts
```

Install with additional input/output adapter:
```sh
pip install 'tirex-ts[gluonts,hfdataset]'
```

You can also install TiRex with all extras at once using:
```sh
pip install 'tirex-ts[all]'
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

## TiRex Docker image

For detailed instructions on building and running TiRex in a Docker container, see the [Docker README](./inference/README.md).

## Finetuning TiRex
TiRex already provide state-of-the-art performance for zero-shot prediction. Hence, you can use it without training on your own data.

If you are interested in models fine-tuned on your data or with different pretraining, please contact us at [contact@nx-ai.com](mailto:contact@nx-ai.com)


## CUDA Kernels

Tirex can use custom CUDA kernels for the sLSTM cells.
These CUDA kernels are compiled when the model is loaded the first time.
The CUDA kernels require GPU hardware that support CUDA compute capability 8.0 or later.
We also highly suggest to use the provided [conda environment spec](./requirements_py26.yaml).
The CUDA kernels are automatically used when the xlstm package is installed.

To install TiRex with the CUDA kernels run:
```sh
pip install 'tirex-ts[cuda,gluonts,hfdataset]'
```

Explicitly set the custom CUDA backend:
```python
model = load_model("NX-AI/TiRex", backend="cuda")
```

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
@inproceedings{auer:25tirex,
  title = {{{TiRex}}: {{Zero-Shot Forecasting Across Long}} and {{Short Horizons}} with {{Enhanced In-Context Learning}}},
  author = {Andreas Auer and Patrick Podest and Daniel Klotz and Sebastian B{\"o}ck and G{\"u}nter Klambauer and Sepp Hochreiter},
  booktitle = {The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
  year = {2025}
  url = {https://arxiv.org/abs/2505.23719},
}
```

## License

TiRex is licensed under the [NXAI community license](./LICENSE).
