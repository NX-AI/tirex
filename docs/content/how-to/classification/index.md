---
sidebar_position: 1
title: Overview
---
# Time Series Classification with TiRex
[![Paper](https://img.shields.io/static/v1?label=Paper&message=2510.26777&color=B31B1B&logo=arXiv)](http://arxiv.org/abs/2510.26777)

Classification is one of the core capabilities of **TiRex**, leveraging the power of **xLSTM** to extract meaningful features from time series data efficiently — even across highly irregular or complex time series.

This section provides a structured overview of how TiRex approaches classification from both **theoretical** and **practical** perspectives.

---

## 📘 [Theory](theory.md)

Learn the fundamental ideas behind classification with **TiRex**.
Includes explanations of:
- Zero-shot protocol with frozen TiRex forecasting model as feature extractor
- Embedding extraction and aggregation strategies
- Multivariate time series handling
- Embedding augmentation techniques
- Conceptual summary of the classification transformation pipeline

---

## 🧭 [Workflow](workflow.md)

Step-by-step playbook that bridges the paper and the tutorials:
- Data preparation
- Model initialization
- Training and evaluation workflows
- Troubleshooting checklist for common issues

---

## ⚙️ [Practice](practice.md)

Dive into practical classification with TiRex.
Hands-on tutorials and examples showing:
- Installation and setup of TiRex classification
- Data preprocessing
- Initializing and configuring classifiers
- Fitting models to your data
- Making predictions and evaluating results


---

Together, these guides form a complete introduction to applying **TiRex** for real-world time series classification.

## Cite

If you use TiRex for Time Series Classification in your research, please cite our work:

```bibtex
@inproceedings{auer:25tirexclassification,
    title = {Pre-trained Forecasting Models: Strong Zero-Shot Feature Extractors for Time Series Classification},
    author = {Andreas Auer and Daniel Klotz and Sebastinan B{\"o}ck and Sepp Hochreiter},
    booktitle = {NeurIPS 2025 Workshop on Recent Advances in Time Series Foundation Models (BERT2S)},
    year = {2025},
    url = {https://arxiv.org/abs/2510.26777},
}
```
