---
sidebar_position: 4
title: Workflow
---
# Classification Workflow with TiRex

This guide walks through the practical steps that connect the high-level theory from the [TiRex classification paper](https://arxiv.org/abs/2510.26777) to the hands-on tutorials. Combine it with the [Theory](./theory.md) page for architectural context and the [Practice](./practice.md) notebook-style examples.

## 1. Prepare the data

1. **Raw sources** → time-series in pytorch format. Keep channels/dimensions consistent.
2. **Label encoding** → Convert string labels to pytorch integer tensor.
3. **Data splitting** → Random Forest classifier works only with a training set (without validation set). For Linear classifier, use training and validation datasets for the classification head training. If you provide  train dataset only, it will be internally split on train and validation datasets. We do recommend the `stratify=True` option, which performs dataset split and ensures the class distribution is preserved between the training and validation datasets.

### Data format requirements

- **Univariate time series**: Shape `(num_samples, 1, sequence_length)`.
- **Multivariate time series**: Shape `(batch_size, num_variates, sequence_length)`.
- **Variable length**: The maximum lengths of the time series must be 2048.

## 2. Choose a classifier

TiRex provides two classifier options:

1. **Random Forest Classifier:** `TirexRFClassifier`
2. **Torch Linear Classifier:** `TirexLinearClassifier`
3. **Gradient Boosting Classifier:** `TirexGBMClassifier`

We recommend starting with `data_augmentation=False`, because the embedding computation will be faster. If you want to improve classification accuracy, you can switch to `data_augmentation=True` in order to add additional information about the dataset to the embedding.


If you have an unbalanced dataset, you can experiment with `class_weights`, which rescales weights for cross entropy loss. For more details, see [CrossEntropyLoss documentation (weight parameter)](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)


## 3. Train the classifier

The TiRex backbone model weights are automatically fetched from HuggingFace. Only the classification head is trained.

```python
# Fit the model
fit_meta = classifier.fit((train_X, train_y))
```

- **Frozen backbone**: The pre-trained TiRex forecasting model remains frozen during training.
- **Embedding extraction**: Time series are converted to fixed-size embeddings automatically.
- **Training**: Only the classification head (Random Forest or Linear layer) is trained.

## 4. Make predictions

```python
# Predict class labels
predictions = classifier.predict(test_X)

# Predict probabilities
probabilities = classifier.predict_proba(test_X)
```

## 5. Evaluate

Common metrics for time series classification:

- **F1 Score**: Harmonic mean of precision and recall (use `average="macro"` for multi-class)
- **Confusion Matrix**: For detailed per-class performance

## 6. Troubleshooting checklist

- **Memory issues** → Reduce `batch_size` for classification model.
- **Embedding computation Speed** → For big datasets, we highly recommend setting `device="cuda:0"`, otherwise the computation will be slow.
- **Variable length sequences** → The maximum length of the time series must be 2048. If bigger, the time series signal will be truncated.

> For a deeper breakdown of the classification architecture and embedding strategies, revisit the [Theory](./theory.md) page and the TiRex classification paper.
