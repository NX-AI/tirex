# TiRex: Zero-Shot Forecasting across Long and Short Horizons

[Paper](https://arxiv.org/abs/2505.23719) | [TiRex Huggingface Model Card](https://huggingface.co/NX-AI/TiRex)


This repository provides the pre-trained forecasting model TiRex introduced in the paper
[TiRex: Zero-Shot Forecasting across Long and Short Horizons with Enhanced In-Context Learning](https://arxiv.org/abs/2505.23719).


## TiRex Model

TiRex is a 35M parameter pre-trained time series forecasting model based on [xLSTM](https://github.com/NX-AI/xlstm).

### Key Facts:

- **Zero-Shot Forecasting**:
  TiRex performs forecasting without any training on your data. Just download and forecast.

- **Quantile Predictions**:
  TiRex not only provides point estimates but provides quantile estimates.

- **State-of-the-art Performance over Long and Short Horizons**:
  TiRex achieves top scores in various time series forecasting benchmarks, see [GiftEval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [ChronosZS](https://huggingface.co/spaces/autogluon/fev-leaderboard).
  These benchmark show that TiRex provides great performance for both long and short-term forecasting.

## Installation
TiRex is currently only tested on *Linux systems* and Nvidia GPUs with compute capability >= 8.0.
If you want to use different systems, please check the [FAQ](#faq--troubleshooting).
It's best to install TiRex in the specified conda environment.
The respective conda dependency file is [requirements_py26.yaml](./requirements_py26.yaml).

```sh
# 1) Setup and activate conda env from ./requirements_py26.yaml
git clone github.com/NX-AI/tirex
conda env create --file ./tirex/requirements_py26.yaml
conda activate tirex

# 2) [Mandatory] Install Tirex

## 2a) Install from source
git clone github.com/NX-AI/tirex  # if not already cloned before
cd tirex
pip install -e .

# 2b) Install from PyPi (will be available soon)

# 2) Optional: Install also optional dependencies
pip install .[gluonts]      # enable gluonTS in/output API
pip install .[hfdataset]    # enable HuggingFace datasets in/output API
pip install .[notebooks]    # To run the example notebooks
```


## Quick Start

```python
import torch
from tirex import load_model, ForecastModel

model: ForecastModel = load_model("NX-AI/TiRex")
data = torch.rand((5, 128))  # Sample Data (5 time series with length 128)
forecast = model.forecast(context=data, prediction_length=64)
```

We provide an extended quick start example in [examples/quick_start_tirex.ipynb](./examples/quick_start_tirex.ipynb).
This notebook also shows how to use the different input and output types of you time series data.
If you dont have suitable hardware you can run the the extended quick start example example also in Google Colab:

<a target="_blank" href="https://colab.research.google.com/github/NX-AI/tirex/blob/main/examples/quick_start_tirex.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open Quick Start In Colab"/>
</a>

###  Example Notebooks

We provide notebooks to run the benchmarks: [GiftEval](./examples/gifteval/gifteval.ipynb) and [Chronos-ZS](./examples/chronos_zs/chronos_zs.ipynb).



## FAQ / Troubleshooting:

- **Can I run TiRex on CPU?**:
  > At the moment CPU support **is experimental**.
  Running on CPU will slow down the model considerable and might likely forecast results.
  To enable TiRex on CPU you need to disable the CUDA kernels (see section [CUDA Kernels](#cuda-kernels)).
  You can also test Tirex with [Google Colab](https://colab.research.google.com/github/NX-AI/tirex/blob/main/examples/quick_start_tirex.ipynb).
  If you are interested in running TiRex on more resource constrained or embedded devices get in touch with us.

- **Can I run TiRex on Windows?**:
  > We don't support Windows at the moment.
  You might still be able to run TiRex on Windows.
  In this case you can skip the conda environment installation.
  For troubleshooting on windows you can find relevant discussion on the [xLSTM GitHub repository](https://github.com/NX-AI/xlstm/issues?q=is%3Aissue%20state%3Aopen%20windows).
  You can also test Tirex with [Google Colab](https://colab.research.google.com/github/NX-AI/tirex/blob/main/examples/quick_start_tirex.ipynb).


- **Can I run TiRex on macOS?**:
  > macOS is not officially supported yet, but TiRex can run on CPU (see above) and hence on macOS.
 MPS has the same limitations as CPU and is also experimental.
  You can also test Tirex with [Google Colab](https://colab.research.google.com/github/NX-AI/tirex/blob/main/examples/quick_start_tirex.ipynb).


- **Can I run TiRex on Nvidia GPU with CUDA compute capability < 8.0?**:
  > The custom CUDA kernels require a GPU with CUDA compute capability >= 8.0.
  You can deactivate the custom CUDA kernels (see section [CUDA Kernels](#cuda-kernels) for more details).
  However, at the moment this **is experimental** and can result in NaNs or degraded forecasts!
  If you are interested in running TiRex on more resource constrained or embedded devices get in touch with us.

- **Can I train / finetune TiRex for my own data?**:
  > TiRex already provide state-of-the-art performance for zero-shot prediction. Hence, you can use it without training on your own data.
  However, we plan to provide fine-tuning support in the future.
  If you are interested in models fine-tuned on your data, get in touch with us.

- **Error during the installation of the conda environment**:
  > If you encounter errors during installing of the conda environment your system most likely does not support CUDA.
  Please skip the conda environment installation and install TiRex directly in a Python environment.

- **When loading TiRex I get error messages regarding sLSTM or CUDA**:
  > Please check the section on [CUDA kernels](#cuda-kernels) in the Readme.
  In the case you can not fix problem you can use a fallback implementation in pure Pytorch.
  However, this can slow down TiRex considerably and might degrade results!



## CUDA Kernels

Tirex uses custom CUDA kernels for the sLSTM cells.
These CUDA kernels are compiled when the model is loaded the first time.
The CUDA kernels require GPU hardware that support CUDA compute capability 8.0 or later.
We also highly suggest to use the provided [conda environment spec](./requirements_py26.yaml).
If you don't have such a device, or you have unresolvable problems with the kernels you can use a fallback implementation in pure Pytorch.
**However, this is at the moment **EXPERIMENTAL**, **slows** down TiRex considerably and likely **degrade forecasting** results!**
To disable the CUDA kernels set the environment variable
```bash
export TIREX_NO_CUDA=1
```
or within python:

```python
import os
os.environ['TIREX_NO_CUDA'] = '1'
```

### Troubleshooting CUDA

**This information is taken from the
[xLSTM repository](https://github.com/NX-AI/xlstm) - See this for further details**:

For the CUDA version of sLSTM, you need Compute Capability >= 8.0, see [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus). If you have problems with the compilation, please try:
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
