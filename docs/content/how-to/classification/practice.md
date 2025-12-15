---
sidebar_position: 3
title: Practice
---
# Practice

In order to utilise TiRex classification, make sure to either locally install it in your preferred Python environment or use a hosted Jupyter Notebook service like [Google Colab](https://colab.google/).

## 1. Install Tirex
```sh
# install with the extra 'classification' for classification support
pip install 'tirex-ts[classification]'
```

## 2. Import TiRex and supporting libraries

Install additional packages (used only for example data).
```bash
pip install aeon
```

Either in a Jupyter Notebook, or in a local Python file in your Python Environment:

```python
# General imports
import torch
from aeon.datasets import load_italy_power_demand
from sklearn.preprocessing import LabelEncoder

# Import TiRex classifiers
from tirex.models.classification import TirexRFClassifier, TirexLinearClassifier, TirexGBMClassifier

```

## 3. Preprocessing of the Data


```python
# Load train data
train_X, train_y = load_italy_power_demand(split="train")

# Load test data
test_X, test_y = load_italy_power_demand(split="test")

# Encode string labels -> integers
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_y)
test_y = label_encoder.transform(test_y)

# Convert to torch tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)

train_y = torch.tensor(train_y, dtype=torch.long)
test_y = torch.tensor(test_y, dtype=torch.long)
```

## 4. Initialize TiRex Classifier
Model weights of the TiRex backbone model are automatically fetched from HuggingFace.

### Option A: Random Forest Classifier

```python
classifier = TirexRFClassifier(data_augmentation=False, device="cuda:0", n_estimators=50, max_depth=10, random_state=42)
```
- `data_augmentation` (bool): Whether to use additional data augmentation concatenated to the embeddings. Defaults to False.
- `device` (str): Device used for embedding computation (for example, "cuda:0" for GPU or "cpu"). Note: Random Forest itself always runs on CPU (uses scikit-learn).
- `compile` (bool): Wheter to compile the frozen embedding model. Default: False
- The rest of the parameters are kwargs to RandomForest of sklearn. For more details see [scikit-learn RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

### Option B: Torch Linear Classifier

```python
classifier = TirexLinearClassifier(
    data_augmentation=False,
    device="cuda:0"
)
```
- `data_augmentation` (bool): Whether to use additional data augmentation concatenated to the embeddings. Defaults to False.
- `device` (str): Device used for training and inference (for example, "cuda:0" for GPU or "cpu").
- `compile` (bool): Wheter to compile the frozen embedding model. Default: False

The rest of the parameters you can see in the API description.

### Option C: Gradient Boosting Classifier

```python
classifier = TirexGBMClassifier(
    data_augmentation=False,
    device="cuda:0",
    batch_size=512,
    early_stopping_rounds=10,
    min_delta=0.0,
    val_split_ratio=0.2,
    stratify=True
)
```
- `data_augmentation` (bool): Whether to use additional data augmentation concatenated to the embeddings. Defaults to False.
- `device` (str | None): Device used for embedding computation (for example, "cuda:0" for GPU or "cpu"). If None, uses CUDA if available, else CPU. Note: LightGBM itself always runs on CPU.
- `compile` (bool): Wheter to compile the frozen embedding model. Default: False
- `batch_size` (int): Batch size for embedding calculations. Default: 512.
- `early_stopping_rounds` (int | None): Number of rounds without improvement of all metrics for Early Stopping. Default: 10. Set to None to disable early stopping.
- `min_delta` (float): Minimum improvement in score to keep training. Default: 0.0.
- `val_split_ratio` (float): Proportion of training data to use for validation, if validation data are not provided. Default: 0.2.
- `stratify` (bool): Whether to stratify the train/validation split by class labels. Default: True.
- The rest of the parameters are kwargs to LightGBM's LGBMClassifier. For more details see [LGBMClassifier documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html).

## 5. Fit the model to the data

```python
classifier.fit((train_X, train_y))
```

For Linear classifier and GBM classifier, you have the possibility to provide a validation set.

```python
classifier.fit(train_data=(train_X, train_y), val_data=(val_X, val_y))
```

If you don't provide it, the train dataset will be split into train and validation datasets internally.

For both Linear classifier and Gradient Boosting classifier, two parameters control the train/validation split:
- `val_split_ratio` (float): Size of the validation set (between 0 and 1). Defaults to: 0.2
- `stratify` (bool): Whether to stratify the train/validation split by class labels. Default: True

For reproducibility, the Linear classifier uses the `seed` parameter, while the GBM classifier uses the `random_state` parameter from LightGBM kwargs (if provided).

## 6. Prediction results

Analyze your prediction results
```python
from sklearn.metrics import accuracy_score, f1_score

pred_y = classifier.predict(test_X)

pred_y = pred_y.cpu().numpy()
test_y_np = test_y.cpu().numpy()

print("Accuracy: " + str(accuracy_score(test_y_np, pred_y)))
print("F1 Score: " + str(f1_score(test_y_np, pred_y, average="macro")))
```
