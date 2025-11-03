---
sidebar_position: 4
title: Evaluation
---
# Forecast Evaluation

Evaluating forecasts is just as critical as producing them. The metrics below capture different aspects of forecast quality and help you understand whether a model is fit for production or needs further tuning.

## 📏 Core Metrics

### Mean Absolute Percentage Error (MAPE)

Calculates the average absolute error as a percentage of the actual values, making it scale-independent.

*Interpretation:* MAPE expresses the average error in percentage terms. A MAPE of 10 % means that, on average, the forecast is off by 10 % of the actual value. This is helpful when you compare time series with different scales.

```python
import numpy as np


def mape(prediction, actual):
    prediction = np.asarray(prediction)
    actual = np.asarray(actual)
    non_zero = actual != 0  # guard against division by zero
    return np.mean(np.abs((actual[non_zero] - prediction[non_zero]) / actual[non_zero])) * 100
```

### Mean Absolute Error (MAE)

Represents the average absolute difference between the forecasts and the actual values. It tells you how far off your predictions are from the real outcomes in the original data units.

```python
def mae(prediction, actual):
    prediction = np.asarray(prediction)
    actual = np.asarray(actual)
    return np.mean(np.abs(actual - prediction))
```

### Mean Squared Error (MSE)

Calculates the average of the squared differences between forecasts and actual values. Squaring the errors increases the penalty for larger deviations.

*Interpretation:* MSE is measured in the square of the original data units (for example, dollars squared). A lower MSE indicates a better fit, but the absolute magnitude can be harder to interpret than MAE.

```python
def mse(prediction, actual):
    prediction = np.asarray(prediction)
    actual = np.asarray(actual)
    return np.mean((actual - prediction) ** 2)
```

### Root Mean Squared Error (RMSE)

The square root of MSE brings the error back to the original scale of the data, making it easier to interpret.

*Interpretation:* RMSE will always be greater than or equal to MAE. The larger the gap between them, the more the metric is dominated by a handful of large errors.

```python
def rmse(prediction, actual):
    return np.sqrt(mse(prediction, actual))
```

### Putting the Metrics to Work

```python
# toy data
ground_truth = np.array([100, 105, 98, 110])
forecast_mean = np.array([102, 103, 101, 107])

print(f"MAPE [%]: {mape(forecast_mean, ground_truth):6.2f}")
print(f"MAE     : {mae(forecast_mean, ground_truth):6.2f}")
print(f"MSE     : {mse(forecast_mean, ground_truth):6.2f}")
print(f"RMSE    : {rmse(forecast_mean, ground_truth):6.2f}")
```

Use these helpers in your evaluation notebooks so you can quickly compare models, tune horizons, and detect regressions.
