---
sidebar_position: 4
title: Workflow
---
# Regression Workflow with TiRex

This guide walks through the practical steps for applying TiRex to regression tasks. For the theoretical foundation (which is shared with classification), see the [Classification Theory](../classification/theory.md) page.

## 1. Prepare the data

1. **Raw sources** → time-series in pytorch format. Keep channels/dimensions consistent.
2. **Target values** → Convert target values to pytorch float tensor.
3. **Data splitting** → Random Forest regressor works with training set only (without validation set). For Linear and Gradient Boosting regressors, you can provide training and validation datasets for the regression head training. If you provide train dataset only, it will be internally split on train and validation datasets.

### Data format requirements

- **Univariate time series**: Shape `(num_samples, 1, sequence_length)`.
- **Multivariate time series**: Shape `(batch_size, num_variates, sequence_length)`.
- **Variable length**: The maximum lengths of the time series must be 2048.
- **Target values**:
  - For Linear regressor: Shape `(num_samples, 1)`
  - For RF/GBM regressors: Shape `(num_samples,)` or `(num_samples, 1)`

## 2. Choose a regressor

TiRex provides three regressor options:

1. **Random Forest Regressor:** `TirexRFRegressor`
2. **Torch Linear Regressor:** `TirexLinearRegressor`
3. **Gradient Boosting Regressor:** `TirexGBMRegressor`

We recommend starting with `data_augmentation=False`, because the embedding computation will be faster. If you want to improve regression accuracy, you can switch to `data_augmentation=True` in order to add additional information about the dataset to the embedding.

## 3. Train the regressor

The TiRex backbone model weights are automatically fetched from HuggingFace.

```python
# Fit the model
fit_meta = regressor.fit((train_X, train_y))
```

- **Frozen backbone**: The pre-trained TiRex forecasting model remains frozen during training.
- **Embedding extraction**: Time series are converted to fixed-size embeddings automatically.
- **Training**: Only the regression head (Random Forest, Linear layer, or Gradient Boosting) is trained.

## 4. Make predictions

```python
# Predict target values
predictions = regressor.predict(test_X)
```

## 5. Evaluate

Common metrics for time series regression:

- **Mean Squared Error (MSE)**: Average squared difference between predictions and targets
- **Mean Absolute Error (MAE)**: Average absolute difference between predictions and targets
- **R² Score**: Coefficient of determination, measures proportion of variance explained

## 6. Troubleshooting checklist

- **Memory issues** → Reduce `batch_size` for the model.
- **Embedding computation Speed** → For big datasets, we highly recommend setting `device="cuda:0"`, otherwise the computation will be slow.
- **Variable length sequences** → The maximum length of the time series must be 2048. If bigger, the time series signal will be truncated.

> For a deeper breakdown of the embedding architecture and extraction strategies, see the [Classification Theory](../classification/theory.md) page (the approach is identical for regression).
