# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
import tempfile

import numpy as np
import pytest
import torch

from tirex.models.regression import TirexRFRegressor


@pytest.fixture
def regression_data():
    torch.manual_seed(42)
    np.random.seed(42)
    n_train = 50
    n_test = 10
    n_vars = 1
    seq_len = 128

    X_train = torch.randn(n_train, n_vars, seq_len)
    y_train = torch.randn(
        n_train,
    )

    X_test = torch.randn(n_test, n_vars, seq_len)
    y_test = torch.randn(
        n_test,
    )

    return X_train, y_train, X_test, y_test


def test_initialization_default():
    regressor = TirexRFRegressor()

    assert regressor.emb_model is not None
    assert regressor.head is not None


def test_initialization_with_rf_params():
    regressor = TirexRFRegressor(
        n_estimators=30,
        max_depth=5,
        random_state=42,
    )

    assert regressor.emb_model is not None
    assert regressor.head.n_estimators == 30
    assert regressor.head.max_depth == 5
    assert regressor.head.random_state == 42


def test_predict(regression_data):
    X_train, y_train, X_test, _ = regression_data

    regressor = TirexRFRegressor(n_estimators=10)
    regressor.fit((X_train, y_train))
    predictions = regressor.predict(X_test)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(X_test),)
    assert predictions.dtype == torch.float32


def test_save_and_load_model(regression_data):
    X_train, y_train, X_test, _ = regression_data

    # Train and save model
    regressor = TirexRFRegressor(n_estimators=10, random_state=42)
    regressor.fit((X_train, y_train))
    predictions_before = regressor.predict(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".joblib") as f:
        model_path = f.name

    try:
        regressor.save_model(model_path)

        # Load model
        loaded_regressor = TirexRFRegressor.load_model(model_path)
        predictions_after = loaded_regressor.predict(X_test)

        # Check predictions match (within numerical precision)
        assert torch.allclose(predictions_before, predictions_after, atol=1e-6)
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

    # Create torch tensors instead of numpy arrays
    X_train = torch.randn(n_train, n_vars, seq_len, dtype=torch.float32)
    y_train = torch.randn(n_train, dtype=torch.float32)
    X_test = torch.randn(n_test, n_vars, seq_len, dtype=torch.float32)

    regressor = TirexRFRegressor(n_estimators=10, random_state=42)
    regressor.fit((X_train, y_train))
    predictions = regressor.predict(X_test)

    assert predictions.shape == (n_test,)
    assert predictions.dtype == torch.float32


################################### TESTS WITH COMPILE #####################################################
def test_compile_initialization():
    regressor = TirexRFRegressor(compile=True)
    assert regressor._compile is True
    assert regressor.emb_model is not None


def test_compile_fit_and_predict(regression_data):
    X_train, y_train, X_test, _ = regression_data

    regressor = TirexRFRegressor(compile=True, n_estimators=10)
    regressor.fit((X_train, y_train))
    predictions = regressor.predict(X_test)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(X_test),)
    assert predictions.dtype == torch.float32


def test_save_and_load_with_compile(regression_data):
    X_train, y_train, X_test, _ = regression_data

    # Train and save model with compile=True
    regressor = TirexRFRegressor(compile=True, n_estimators=10, random_state=42)
    regressor.fit((X_train, y_train))
    predictions_before = regressor.predict(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".joblib") as f:
        model_path = f.name

    try:
        regressor.save_model(model_path)

        # Load model
        loaded_regressor = TirexRFRegressor.load_model(model_path)
        predictions_after = loaded_regressor.predict(X_test)

        # Check predictions match (within numerical precision)
        assert torch.allclose(predictions_before, predictions_after, atol=1e-6)

        # Check compile parameter is preserved
        assert loaded_regressor._compile is True
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
