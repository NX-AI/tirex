---
sidebar_position: 1
title: Overview
---
# Time Series Regression with TiRex

Regression is a key capability of **TiRex**, leveraging the power of **xLSTM** to extract meaningful features from time series data efficiently — even across highly irregular or complex time series.

TiRex regression uses the same approach as [classification](../classification/index.md): a frozen pre-trained TiRex forecasting model extracts embeddings, which are then fed into a regression head to predict continuous target values.

This section provides a structured overview of how to apply TiRex for regression tasks.

---

## 📘 Theory


For detailed theoretical understanding, please refer to the [Classification Theory](../classification/theory.md) page, which covers:
- Zero-shot protocol with frozen TiRex forecasting model as feature extractor
- Embedding extraction and aggregation strategies
- Multivariate time series handling
- Embedding augmentation techniques

The only difference is the task-specific head: **regression head** maps embeddings to continuous target values.

---

## 🧭 [Workflow](workflow.md)

Step-by-step guide for applying TiRex to regression tasks:
- Data preparation
- Model initialization
- Training and evaluation workflows
- Troubleshooting checklist for common issues

---

## ⚙️ [Practice](practice.md)

Hands-on tutorials and examples showing:
- Installation and setup of TiRex regression
- Data preprocessing
- Initializing and configuring regressors
- Fitting models to your data
- Making predictions and evaluating results

---

Together, these guides provide a complete introduction to applying **TiRex** for real-world time series regression tasks.
