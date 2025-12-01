---
sidebar_position: 2
title: Theory
---
# Theory

Time series classification with **TiRex** builds on the powerful **xLSTM** (Extended Long Short-Term Memory) architecture — a next-generation recurrent model designed to combine the efficiency of classic RNNs with the long-range modeling capabilities of Transformers — and achieves classification accuracy that surpasses state-of-the-art models pre-trained specifically for classification.

This section provides a conceptual overview of how TiRex classification model works.

---

## TiRex Classification Model Architecture

- **TiRex Forecasting Model as Feature Extractor:**
  Instead of training a classifier on the raw time series, a pre-trained TiRex time series forecasting model is used to map raw time series to latent representation, which is fed into a classification head to output the final prediction.

- **Zero-Shot Protocol:**
  TiRex is used in a zero-shot setting, meaning the pre-trained TiRex forecasting model operates strictly as a frozen feature extractor — none of its parameters are updated or fine-tuned during classification.

- **Embedding Extraction & Aggregation:**
  Hidden states from all xLSTM layers are extracted to preserve information at different abstraction levels. Mean pooling is applied across sequence dimension, and the resulting layer-wise representations are concatenated to produce a fixed-size, robust embedding that accommodates variable-length series.

- **Multivariate Time Series:**
  For multivariate time-series classification, we adopt a proven forecasting technique: treating each variate independently. Each variate is processed separately with TiRex, producing a dedicated embedding for each variate.

- **Embeddings Augmentation - Time Series Differencing:**
  To isolate strong trends in time series that can dominate the signal and mask more subtle patterns, first-order differencing is employed. A new time series is derived by taking the difference between consecutive time steps.

- **Embeddings Augmentation - Absolute Sample Statistic:**
  To preserve information regarding the absolute values and scale of the time series, the model's embedding is augmented with basic sample statistics. The raw time series is split into 8 non-overlapping patches, and for each patch, mean, standard deviation, minimum, and maximum values are calculated. These statistics are then concatenated with the already computed embedding to form the final representation.

---

## Conceptual Summary

TiRex’s classification process can be viewed as a sequence of transformations:

1. **Hidden States Extraction** →
   Hidden states of all layers of the TiRex forecasting model are extracted.

2. **Sequence Aggregation** →
   Hidden states are aggregated in sequence dimension with mean pooling.

3. **Embedding normalization** →
   Embeddings are normalized because different layers might operate in different feature spaces.

4. **Layer Aggregation** →
   Features of each layer are aggregated by concatenation.

5. **Optional embedding augmentations** →
   To improve classification accuracy, additional data augmentation is applied:
   - **Differenced-series embeddings:** Helps to identify more subtle patterns.
   - **Absolute sample statistics:** Restores scale information lost due to normalization.

6. **Simple classifier prediction** →
   A classification head (Random Forest, Linear Layer or Gradient Boosting) is trained and then maps the final embedding to class labels.

---

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

---

> **Next:**
> - [→ Workflow: Practical Classification Checklist](./workflow.md)
> - [→ Practice: Hands-on Classification with TiRex](./practice.md)
