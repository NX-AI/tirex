---
sidebar_position: 3
title: Workflow
---
# Forecasting Workflow with TiRex

This guide walks through the practical steps that connect the high-level theory from the [TiRex paper](https://arxiv.org/abs/2505.23719) to the hands-on tutorials. Combine it with the [Theory](./theory.md) page for architectural context and the [Practice](./practice.md) notebook-style examples.

## 1. Prepare the data

1. **Raw sources** → tidy time-series format (Pandas, NumPy, or PyTorch tensors). Keep channels/dimensions consistent.
2. **Normalize/scale** if the distribution is skewed—TiRex is fairly robust, but per-series normalisation improves stability (see Appendix B in the paper).
3. **Create contexts** (history windows) and target horizons matching your use case. Leaderboards typically use 512-step contexts, but TiRex supports shorter sequences as long as the model sees enough seasonal structure.

### Optional adapters

- **GluonTS** datasets → wrap with TiRex’s GluonTS adapter (`pip install tirex-ts[gluonts]`).
- **Hugging Face datasets** → use the built-in HF dataset loader (`pip install tirex-ts[hfdataset]`).
- **Custom CSVs** → convert to tensors or NumPy arrays and feed directly into `model.forecast`.

## 2. Load the model

```python
from tirex import load_model
model = load_model("NX-AI/TiRex")
```

- **Device**: TiRex runs on CPU and GPU; pass `device="cuda"` if available.
- **Backend**: the paper’s experiments default to the custom CUDA kernels when present; use `backend="torch"` for pure PyTorch (slower but portable).
- **Compile**: enable `compile=True` (PyTorch 2.1+) for medium gains on repeated inference.

## 3. Run forecasts

```python
quantiles, mean = model.forecast(
    context,                 # tensor/array [batch, time]
    prediction_length=64,    # number of steps to forecast
)
```

- **Quantiles**: by default nine quantiles (0.1…0.9) as reported in the paper are returned.
- **Batching**: large batches improve throughput; keep an eye on memory if prediction length is long.

## 4. Evaluate

Stay grounded in the theory that motivates the TiRex leaderboards: the paper emphasises sMAPE, MASE, and CRPS because they each capture complementary error modes.

- **sMAPE** highlights relative accuracy when magnitudes differ across series; look for stability around the 0–10% band on heterogeneous panels.
- **MASE** translates residuals into scale-free terms by normalising against naive seasonal baselines (pick the seasonal period that reflects your domain).
- **CRPS** integrates across the full predictive distribution, so any quantile miscalibration shows up as an increased score.

See also [Evaluation](../evaluation.md) for further descriptions and code examples of the most common metrics.

A forthcoming convenience evaluator will encapsulate the canonical settings for these metrics. Until it lands, anchor your interpretation in the theoretical properties above and match the leaderboard configurations described in the TiRex paper if you need strict comparability.

## 5. Visualise & inspect

Refer to the `tirex.util.plot_forecast` function to visualise predictions. Visual checks (fan plots, residual plots) help catch data issues that metrics alone might miss.

## 6. Troubleshooting checklist

- **Short context** → Try extending the history window; TiRex relies on seasonal cues.
- **Scale mismatch** → Normalise per series or apply log transforms before forecasting.
- **No GPU** → Set `backend="torch"` and `device="cpu"`; latency increases but results stay consistent.

> For a deeper breakdown of the xLSTM architecture and how TiRex handles multi-scale patterns, revisit the [Theory](./theory.md) page and Sections 3–4 of the paper.
