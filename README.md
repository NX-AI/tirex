# TiRex: Zero-Shot Forecasting across Long and Short Horizions

[Paper - TBA]() | [TiRex Hugginface Model Card](https://huggingface.co/NX-AI/TiRex)


This repository provides the pre-trained forecasting model TiRex introduced in the paper [TiRex: Zero-Shot Forecasting across Long and Short Horizions with Enhanced In-Context Learning]().


## TiRex Model

TiRex is a 35M parameter pre-trained time series forecasting model bases on [xLSTM](https://github.com/NX-AI/xlstm).

### Key Facts:

- **Zero-Shot Forecasting**: TiRex performs forecasting without any training on your data. Just download and forecast.

- **Quantile Predicitons**: TiRex not only provides point estimates but provides quantile estimates.

- **State-of-the-art Performance over Long and Short Horizions**: TiRex achieves top scores the divers time series forecasting benchmarks [GiftEval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) and [ChronosZS](https://huggingface.co/spaces/autogluon/fev-leaderboard). These benchmark show that TiRex provides great performance for both -- long and short-term forecasting.


TODO add figure of results


## Installation
Its best to install TiRex in an own conda environment. The respective conda dependency file is [requirements_py26.yaml](./requirements_py26.yaml).

```sh
# 1) [Suggested if you run with Nvidia GPU] Setup and activate conda env from ./requirements_py26.yaml
git clone github.com/NX-AI/tirex
conda env create --file ./tirex/requirements_py26.yaml
conda activate tirex

# 2) [Mandatory] Install Tirex

## 2a) Install from Github
git clone github.com/NX-AI/tirex # if not already cloned before
cd tirex
pip install .

# 2b) Instal from PyPi (Not yet supported)

# 2) Optional: Install also optional dependicies
pip install .[gluonts]      # enable gluonTS in/output API
pip install .[hfdataset]    # enable HuggingFace datasets in/output API
pip install .[notebooks]    # To run the example notebooks
```


## Quick Start

```python
import torch
from tirex import load_model, ForecastModel

model: ForecastModel = load_model("NX-AI/TiRex")
data = torch.rand((5, 128)) # Sample Data (5 time series with length 128)
forecast = model.forecast(context=data, prediction_length=64)
```

We provide an extended quick start example in [examples/quick_start_tirex.ipynb](./examples/quick_start_tirex.ipynb).
This notebook also shows how to use the different input and output types of you time series data.

###  Example Notebooks

We provide notebooks to run the benchmarks: [GiftEval](./examples/gifteval/gifteval.ipynb) and [Chronos-ZS](./examples/chronos_zs/chronos_zs.ipynb).



## FAQ / Troubleshooting:

- **Can I use TiRex on CPU**:
> In general you can use TiRex with CPU. However this will slow down the model considerable and might impact forecast quality.
To enable TiRex on CPU you need to disable the CUDA kernels (see section [CUDA Kernels](#cuda-kernels)).
If you are interested in running TiRex on more resource constrainted or embedded devices get in touch with NxAI.

- **Can I train / finetune TiRex for my own data**
> TiRex already provide state-of-the-art performance for zero-shot prediction, i.e. you can use it without training on your own data.
 However, we plan to provide fine-tuning support in the future. If you are interested you can also get in touch with NxAI.

- **Error during the installation of the conda enviornment**:
> If you encouter errors when installing the conda enviornment your system likley does not support CUDA. This is for example the case with *Mac hardware*. Please skip the conda envionrment installation and install tirex directly in an new python enviornment.
Note that in this case you also need to run TiRex on CPU (see *Can I use TiRex on CPU*).

- **When loading TiRex I get error messages regarding sLSTM or CUDA**:
> Please check the section on [CUDA kernels](#cuda-kernels) in the Readme. In the case you can not fix your problem you can use a fallback implementation in pure Pytorch. However this can slow down TiRex considerably and might degrade results!



## CUDA Kernels

Tirex uses custom CUDA kernels of the sLSTM.
This CUDA kernels are compiled when the model is loaded the first time.
The CUDA kernels require GPU hardware that supporst CUDA Compute Capability 8.0 or later.
We also highly suggest to use the provided (conda enviornment spec)[./requirements_py26.yaml].
If don't have such a device or you have unresolveable problems with the kernels you can
use a fallback implementation in pure Pytorch.
**However this can slow down TiRex considerably and might degrade forecasting results!**
To disable the CUDA kernels set the enironment variable
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

For all kinds of custom setups with torch and CUDA, keep in mind that versions have to match. Also, to make sure the correct CUDA libraries are included you can use the `XLSTM_EXTRA_INCLUDE_PATHS` environment variable now to inject different include paths, e.g.:

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
TODO
```


## License