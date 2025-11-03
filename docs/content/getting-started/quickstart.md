# Quickstart

```python
import torch
from tirex import load_model, ForecastModel

# 1) load model (from HF Hub)
model: ForecastModel = load_model("NX-AI/TiRex")

# 2) prepare context (batch x length)
context = torch.rand((8, 256))

# 3) forecast quantiles + mean for 64 steps
quantiles, mean = model.forecast(context=context, prediction_length=64)

print(mean.shape)   # (8, 64)
```

We provide an extended quick start example Jupyter Notebook in [examples/quick_start_tirex.ipynb](https://github.com/NX-AI/tirex/blob/main/examples/quick_start_tirex.ipynb).
This notebook also shows how to use the different input and output types of your time series data.
You can also run it in [Google Colab](https://colab.research.google.com/github/NX-AI/tirex/blob/main/examples/quick_start_tirex.ipynb).

Additional benchmark notebooks:
- [GiftEval](https://github.com/NX-AI/tirex/blob/main/examples/gifteval/gifteval.ipynb)
- [Chronos-ZS](https://github.com/NX-AI/tirex/blob/main/examples/chronos_zs/chronos_zs.ipynb)
