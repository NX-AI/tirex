# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
import tempfile

import numpy as np
import pytest
import torch
from aeon.datasets import load_italy_power_demand
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from tirex.models.classification import TirexGBMClassifier

# Suppress LightGBM warning about feature names
pytestmark = pytest.mark.filterwarnings("ignore:.*does not have valid feature names.*:UserWarning")


@pytest.fixture
def classification_data():
    torch.manual_seed(42)
    np.random.seed(42)
    n_train = 50
    n_test = 10
    n_vars = 1
    seq_len = 128
    n_classes = 3

    X_train = torch.randn(n_train, n_vars, seq_len)
    y_train = torch.randint(0, n_classes, (n_train,))

    X_test = torch.randn(n_test, n_vars, seq_len)
    y_test = torch.randint(0, n_classes, (n_test,))

    return X_train, y_train, X_test, y_test


@pytest.fixture
def classification_data_real():
    # Load train data
    X_train, y_train = load_italy_power_demand(split="train")

    # Load test data
    X_test, y_test = load_italy_power_demand(split="test")

    # Encode string labels -> integers
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test


def test_initialization_default():
    classifier = TirexGBMClassifier()

    assert classifier.emb_model is not None
    assert classifier.head is not None


def test_initialization_with_gbm_params():
    classifier = TirexGBMClassifier(
        n_estimators=30,
        max_depth=5,
        random_state=42,
    )

    assert classifier.emb_model is not None
    assert classifier.head is not None
    assert classifier.head.n_estimators == 30
    assert classifier.head.max_depth == 5
    assert classifier.head.random_state == 42


def test_f1_score_on_real_data(classification_data_real):
    X_train, y_train, X_test, y_test = classification_data_real

    classifier = TirexGBMClassifier(n_estimators=50, verbosity=-1, random_state=42)
    classifier.fit((X_train, y_train))

    predictions = classifier.predict(X_test)

    pred_y = predictions.cpu().numpy()
    test_y = y_test.cpu().numpy()

    f1 = f1_score(test_y, pred_y, average="macro")

    assert f1 >= 0.90, f"F1-score {f1:.4f} is below the required threshold of 0.90"


def test_predict_proba(classification_data):
    X_train, y_train, X_test, _ = classification_data
    n_classes = 3

    classifier = TirexGBMClassifier(n_estimators=10, verbosity=-1)
    classifier.fit((X_train, y_train))
    probabilities = classifier.predict_proba(X_test)

    assert isinstance(probabilities, torch.Tensor)
    assert probabilities.shape == (len(X_test), n_classes)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(X_test), dtype=probabilities.dtype), atol=1e-6)


def test_save_and_load_model(classification_data):
    X_train, y_train, X_test, _ = classification_data

    # Train and save model
    early_stopping_rounds = 15
    min_delta = 0.001
    classifier = TirexGBMClassifier(
        n_estimators=10, random_state=42, early_stopping_rounds=early_stopping_rounds, min_delta=min_delta, verbosity=-1
    )
    classifier.fit((X_train, y_train))
    predictions_before = classifier.predict(X_test)
    probabilities_before = classifier.predict_proba(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".joblib") as f:
        model_path = f.name

    try:
        classifier.save_model(model_path)

        # Load model
        loaded_classifier = TirexGBMClassifier.load_model(model_path)
        predictions_after = loaded_classifier.predict(X_test)
        probabilities_after = loaded_classifier.predict_proba(X_test)

        # Check predictions match
        assert torch.all(predictions_before == predictions_after)

        # Check probabilities are valid
        assert isinstance(probabilities_after, torch.Tensor)
        assert probabilities_after.shape == (len(X_test), 3)
        assert torch.allclose(probabilities_before, probabilities_after, atol=1e-8)
        assert torch.all((probabilities_after >= 0) & (probabilities_after <= 1))
        assert torch.allclose(
            probabilities_after.sum(dim=1), torch.ones(len(X_test), dtype=probabilities_after.dtype), atol=1e-6
        )

        # Check early stopping and min_delta are preserved
        assert loaded_classifier.early_stopping_rounds == early_stopping_rounds
        assert loaded_classifier.min_delta == min_delta
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


def test_multivariate_data():
    torch.manual_seed(42)
    np.random.seed(42)

    n_train = 80
    n_test = 20
    n_vars = 3
    seq_len = 128
    n_classes = 2

    # Create torch tensors instead of numpy arrays
    X_train = torch.randn(n_train, n_vars, seq_len, dtype=torch.float32)
    y_train = torch.randint(0, n_classes, (n_train,), dtype=torch.long)
    X_test = torch.randn(n_test, n_vars, seq_len, dtype=torch.float32)

    classifier = TirexGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
    classifier.fit((X_train, y_train))
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

    assert predictions.shape == (n_test,)
    assert torch.all((predictions >= 0) & (predictions < n_classes))

    # Check probabilities
    assert isinstance(probabilities, torch.Tensor)
    assert probabilities.shape == (n_test, n_classes)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(n_test, dtype=probabilities.dtype), atol=1e-6)


################################### TESTS WITH COMPILE #####################################################
def test_compile_initialization():
    classifier = TirexGBMClassifier(compile=True)
    assert classifier._compile is True
    assert classifier.emb_model is not None


def test_compile_fit_and_predict(classification_data):
    X_train, y_train, X_test, _ = classification_data

    classifier = TirexGBMClassifier(compile=True, n_estimators=10, verbosity=-1)
    classifier.fit((X_train, y_train))
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(X_test),)
    assert torch.all((predictions >= 0) & (predictions < 3))

    # Check probabilities
    assert isinstance(probabilities, torch.Tensor)
    assert probabilities.shape == (len(X_test), 3)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(X_test), dtype=probabilities.dtype), atol=1e-6)


def test_save_and_load_with_compile(classification_data):
    X_train, y_train, X_test, _ = classification_data

    # Train and save model with compile=True
    classifier = TirexGBMClassifier(compile=True, n_estimators=10, random_state=42, verbosity=-1)
    classifier.fit((X_train, y_train))
    predictions_before = classifier.predict(X_test)
    probabilities_before = classifier.predict_proba(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".joblib") as f:
        model_path = f.name

    try:
        classifier.save_model(model_path)

        # Load model
        loaded_classifier = TirexGBMClassifier.load_model(model_path)
        predictions_after = loaded_classifier.predict(X_test)
        probabilities_after = loaded_classifier.predict_proba(X_test)

        # Check predictions match
        assert torch.all(predictions_before == predictions_after)

        # Check probabilities match
        assert torch.allclose(probabilities_before, probabilities_after, atol=1e-8)

        # Check compile parameter is preserved
        assert loaded_classifier._compile is True
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
