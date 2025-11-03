---
sidebar_position: 2
title: Theory
---
# Theory

Forecasting with **TiRex** builds on the powerful **xLSTM** (Extended Long Short-Term Memory) architecture — a next-generation recurrent model designed to combine the efficiency and interpretability of classic RNNs with the long-range modeling capabilities of Transformers.
This section provides a conceptual overview of how TiRex performs forecasting, followed by detailed descriptions of its input and output handling.

---

## Architectural Foundations

At its core, **xLSTM** introduces several innovations beyond classical LSTMs:

- **Expanded Memory Dynamics:**
  xLSTM extends the traditional LSTM cell with additional multiplicative interactions and normalization mechanisms that improve temporal credit assignment over long horizons.

- **Hierarchical Temporal Abstraction:**
  Time series often contain information at multiple scales (daily, weekly, seasonal).
  xLSTM layers in TiRex are designed to process signals at these varying frequencies efficiently using frequency-aware resampling (based on FFT-derived periodicity).

- **Parameter Efficiency:**
  Compared to Transformers, xLSTM achieves similar or superior accuracy with far fewer parameters and lower latency, making it suitable for **Edge** and **industrial** deployments.

- **Zero-Shot Forecasting Capability:**
  TiRex models are trained on a broad mixture of time series domains, allowing them to generalize to unseen datasets *without fine-tuning*.
  This enables rapid deployment across diverse tasks such as demand, energy, or environmental forecasting.

---

## Conceptual Summary

TiRex’s forecasting process can be viewed as a sequence of transformations:

1. **Input normalization** →
   Ensure consistent scale and alignment of the input series.

2. **Frequency-based resampling** →
   Automatically adjust temporal resolution to align dominant frequencies with model patch size.

3. **xLSTM-based sequence modeling** →
   Encode long- and short-term dependencies in a compact representation.

4. **Probabilistic decoding** →
   Generate multiple quantile forecasts representing the predictive distribution.

Together, these steps enable **accurate, data-efficient, and generalizable** forecasting across time series of varying lengths, domains, and sampling rates.

---

> **Next:**
> - [→ Workflow: Practical Forecasting Checklist](./workflow.md)
> - [→ Practice: Hands-on Forecasting with TiRex](./practice.md)
