# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
import tempfile

import numpy as np
import pytest
import torch

from tirex.models.classification import TirexRFClassifier


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


def test_initialization_default():
    classifier = TirexRFClassifier()

    assert classifier.emb_model is not None
    assert classifier.head is not None


def test_initialization_with_rf_params():
    classifier = TirexRFClassifier(
        n_estimators=30,
        max_depth=5,
        random_state=42,
    )

    assert classifier.emb_model is not None
    assert classifier.head.n_estimators == 30
    assert classifier.head.max_depth == 5
    assert classifier.head.random_state == 42


def test_fit_with_torch_tensors(classification_data):
    X_train, y_train, _, _ = classification_data

    classifier = TirexRFClassifier(n_estimators=10)
    classifier.fit((X_train, y_train))


def test_predict_with_torch_tensors(classification_data):
    X_train, y_train, X_test, _ = classification_data

    classifier = TirexRFClassifier(n_estimators=10)
    classifier.fit((X_train, y_train))
    predictions = classifier.predict(X_test)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(X_test),)
    assert torch.all((predictions >= 0) & (predictions < 3))


def test_save_and_load_model(classification_data):
    X_train, y_train, X_test, _ = classification_data

    # Train and save model
    classifier = TirexRFClassifier(n_estimators=10, random_state=42)
    classifier.fit((X_train, y_train))
    predictions_before = classifier.predict(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".joblib") as f:
        model_path = f.name

    try:
        classifier.save_model(model_path)

        # Load model
        loaded_classifier = TirexRFClassifier.load_model(model_path)
        predictions_after = loaded_classifier.predict(X_test)

        # Check predictions match
        assert torch.all(predictions_before == predictions_after)
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

    classifier = TirexRFClassifier(n_estimators=10, random_state=42)
    classifier.fit((X_train, y_train))
    predictions = classifier.predict(X_test)

    # Convert predictions to tensor if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)

    assert predictions.shape == (n_test,)
    assert torch.all((predictions >= 0) & (predictions < n_classes))
