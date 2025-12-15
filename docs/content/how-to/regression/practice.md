---
sidebar_position: 3
title: Practice
---
# Practice

In order to utilise TiRex regression, make sure to either locally install it in your preferred Python environment or use a hosted Jupyter Notebook service like [Google Colab](https://colab.google/).

## 1. Install Tirex
```sh
# install with the extra 'regression' for regression support
pip install 'tirex-ts[regression]'
```

Install additional packages (used only for example data).
```bash
pip install aeon
```

## 2. Import TiRex and supporting libraries

```python
# General imports
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from aeon.datasets import load_regression

# Import TiRex regressors
from tirex.models.regression import TirexRFRegressor, TirexLinearRegressor, TirexGBMRegressor
```

## 3. Preprocessing of the Data

```python
# Load dataset
X, y, meta = load_regression("HouseholdPowerConsumption1", return_metadata=True)

# Split dataset into train and test (for example, 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
train_X = torch.tensor(X_train, dtype=torch.float32)
test_X = torch.tensor(X_test, dtype=torch.float32)

train_y = torch.tensor(y_train, dtype=torch.float32)
test_y = torch.tensor(y_test, dtype=torch.float32)

print(train_X.shape, train_y.shape)
# torch.Size([1144, 5, 1440]) torch.Size([1144])
```

Note on target format:
- For Linear regressor: Use `train_y` with shape `(num_samples, 1)` - you can reshape with `train_y = train_y.unsqueeze(1)`
- For RF/GBM regressors: Both `(num_samples,)` and `(num_samples, 1)` work

## 4. Initialize TiRex Regressor
Model weights of the TiRex backbone model are automatically fetched from HuggingFace.

### Option A: Random Forest Regressor

```python
regressor = TirexRFRegressor(
    data_augmentation=False,
    device="cuda:0",
    n_estimators=50,
    max_depth=10,
    random_state=42
)
```
- `data_augmentation` (bool): Whether to use additional data augmentation concatenated to the embeddings. Defaults to False.
- `device` (str): Device used for embedding computation (for example, "cuda:0" for GPU or "cpu"). Note: Random Forest itself always runs on CPU (uses scikit-learn).
- `compile` (bool): Whether to compile the frozen embedding model. Default: False
- The rest of the parameters are kwargs to RandomForest of sklearn. For more details see [scikit-learn RandomForestRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

### Option B: Torch Linear Regressor

```python
regressor = TirexLinearRegressor(
    data_augmentation=False,
    device="cuda:0",
    max_epochs=10,
    lr=1e-4,
    batch_size=32
)
```
- `data_augmentation` (bool): Whether to use additional data augmentation concatenated to the embeddings. Defaults to False.
- `device` (str): Device used for training and inference (for example, "cuda:0" for GPU or "cpu").
- `compile` (bool): Whether to compile the frozen embedding model. Default: False
- `max_epochs` (int): Maximum number of training epochs. Default: 10
- `lr` (float): Learning rate for the optimizer. Default: 1e-4
- `batch_size` (int): Batch size for training and embedding calculations. Default: 512

The rest of the parameters you can see in the API description.

### Option C: Gradient Boosting Regressor

```python
regressor = TirexGBMRegressor(
    data_augmentation=False,
    device="cuda:0",
    batch_size=512,
    early_stopping_rounds=10,
    min_delta=0.0,
    val_split_ratio=0.2,
    random_state=42
)
```
- `data_augmentation` (bool): Whether to use additional data augmentation concatenated to the embeddings. Defaults to False.
- `device` (str | None): Device used for embedding computation (for example, "cuda:0" for GPU or "cpu"). If None, uses CUDA if available, else CPU. Note: LightGBM itself always runs on CPU.
- `compile` (bool): Whether to compile the frozen embedding model. Default: False
- `batch_size` (int): Batch size for embedding calculations. Default: 512.
- `early_stopping_rounds` (int | None): Number of rounds without improvement of all metrics for Early Stopping. Default: 10. Set to None to disable early stopping.
- `min_delta` (float): Minimum improvement in score to keep training. Default: 0.0.
- `val_split_ratio` (float): Proportion of training data to use for validation, if validation data are not provided. Default: 0.2.
- The rest of the parameters are kwargs to LightGBM's LGBMRegressor. For more details see [LGBMRegressor documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html).

## 5. Fit the model to the data

```python
regressor.fit((train_X, train_y))
```

For Linear regressor and GBM regressor, you have the possibility to provide a validation set.

```python
regressor.fit(train_data=(train_X, train_y), val_data=(val_X, val_y))
```

If you don't provide it, the train dataset will be split into train and validation datasets internally.

For both Linear regressor and Gradient Boosting regressor, two parameters control the train/validation split:
- `val_split_ratio` (float): Size of the validation set (between 0 and 1). Defaults to: 0.2
- For reproducibility, the Linear regressor uses the `seed` parameter, while the GBM regressor uses the `random_state` parameter from LightGBM kwargs (if provided).

## 6. Prediction results

Analyze your prediction results
```python
pred_y = regressor.predict(test_X)

# Convert to numpy for metric computation
pred_y_np = pred_y.cpu().numpy()
test_y_np = test_y.cpu().numpy()

# Compute regression metrics
print(f"MAE: {mean_absolute_error(test_y_np, pred_y_np):.4f}")
print(f"R² Score: {r2_score(test_y_np, pred_y_np):.4f}")
```
