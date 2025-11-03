---
sidebar_position: 1
title: Overview
---
# Time Series Forecasting with TiRex

Forecasting is one of the core capabilities of **TiRex**, leveraging the power of **xLSTM** to capture long-range dependencies and temporal dynamics efficiently — even across highly irregular or complex time series.

This section provides a structured overview of how TiRex approaches forecasting from both **theoretical** and **practical** perspectives.

---

## 📘 [Theory](theory.md)

Learn the fundamental ideas behind forecasting with **xLSTM** —
how TiRex models sequence dependencies, frequency dynamics, and hierarchical temporal structures.
Includes explanations of:
- xLSTM architecture and its advantages for time series
- frequency-based resampling and patching
- zero-shot generalization across datasets

---

## 🧭 [Workflow](workflow.md)

Step-by-step playbook that bridges the paper and the tutorials:
- Data preparation, scaling, and adapters
- Forecast parameters (quantiles, batching, devices)
- Evaluation scripts (sMAPE/MASE/CRPS) and reproducibility tips

---

## ⚙️ [Practice](practice.md)

Dive into practical forecasting with TiRex.
Hands-on tutorials and examples showing:
- How to prepare data using the workflow checklist
- Running forecasts in zero-shot mode with `load_model`
- Evaluating performance and visualising results (metrics covered in [Evaluation](../evaluation.md))
- Integrating TiRex into notebooks, batch jobs, or leaderboard submissions

Each practice recipe references the relevant sections of the TiRex paper so you can trace implementation details back to the underlying research.

---

Together, these guides form a complete introduction to applying **TiRex** for real-world time series forecasting.
