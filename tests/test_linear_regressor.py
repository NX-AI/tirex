# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
import tempfile

import numpy as np
import pytest
import torch

from tirex.models.regression import TirexLinearRegressor


@pytest.fixture
def regression_data():
    torch.manual_seed(42)
    np.random.seed(42)
    n_train = 50
    n_test = 10
    n_vars = 1
    seq_len = 128

    X_train = torch.randn(n_train, n_vars, seq_len)
    y_train = torch.randn(n_train, 1)

    X_test = torch.randn(n_test, n_vars, seq_len)
    y_test = torch.randn(n_test, 1)

    return X_train, y_train, X_test, y_test


def test_initialization_default():
    regressor = TirexLinearRegressor()
    assert regressor.emb_model is not None
    assert isinstance(regressor.trainer.loss_fn, torch.nn.MSELoss)


def test_initialization_with_custom_params():
    regressor = TirexLinearRegressor(
        max_epochs=20,
        lr=1e-3,
        batch_size=64,
        dropout=0.2,
    )

    assert regressor.trainer.train_config.max_epochs == 20
    assert regressor.trainer.train_config.lr == 1e-3
    assert regressor.trainer.train_config.batch_size == 64
    assert regressor.dropout == 0.2
    assert isinstance(regressor.trainer.loss_fn, torch.nn.MSELoss)


def test_fit_basic(regression_data):
    X_train, y_train, _, _ = regression_data

    regressor = TirexLinearRegressor(
        max_epochs=1,
        batch_size=32,
        log_every_n_steps=5,
    )

    regressor.fit((X_train, y_train))

    assert regressor.head is not None
    assert regressor.emb_dim is not None
    assert regressor.output_dim == 1


def test_fit_with_val_data(regression_data):
    X_train, y_train, X_test, y_test = regression_data

    regressor = TirexLinearRegressor(
        max_epochs=1,
        batch_size=32,
    )

    regressor.fit((X_train, y_train), val_data=(X_test, y_test))
    assert regressor.head is not None


def test_predict(regression_data):
    X_train, y_train, X_test, y_test = regression_data

    regressor = TirexLinearRegressor(max_epochs=1, batch_size=32)
    regressor.fit((X_train, y_train))
    predictions = regressor.predict(X_test)

    assert predictions.shape == (len(X_test), 1)
    assert predictions.dtype == torch.float32


def test_predict_before_fit_raises_error():
    regressor = TirexLinearRegressor()
    X_test = torch.randn(10, 1, 128)

    with pytest.raises(RuntimeError, match="Head not initialized"):
        regressor.predict(X_test)


def test_forward_pass(regression_data):
    X_train, y_train, X_test, y_test = regression_data

    regressor = TirexLinearRegressor(max_epochs=1, batch_size=32)
    regressor.fit((X_train, y_train))
    predictions = regressor.forward(X_test[:5])

    assert predictions.shape == (5, 1)
    assert not torch.isnan(predictions).any()


def test_save_and_load_model(regression_data):
    X_train, y_train, X_test, y_test = regression_data

    # Train and save model
    regressor = TirexLinearRegressor(
        max_epochs=1,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        val_split_ratio=0.1,
        seed=42,
    )
    regressor.fit((X_train, y_train))
    predictions_before = regressor.predict(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pt") as f:
        model_path = f.name

    try:
        regressor.save_model(model_path)

        # Load model
        loaded_regressor = TirexLinearRegressor.load_model(model_path)
        predictions_after = loaded_regressor.predict(X_test)

        # Check predictions match (within numerical precision)
        assert torch.allclose(predictions_before, predictions_after, atol=1e-6)
        assert loaded_regressor.emb_dim == regressor.emb_dim
        assert loaded_regressor.output_dim == regressor.output_dim
        assert loaded_regressor.dropout == regressor.dropout
        assert loaded_regressor.trainer.train_config.max_epochs == regressor.trainer.train_config.max_epochs
        assert loaded_regressor.trainer.train_config.lr == regressor.trainer.train_config.lr
        assert loaded_regressor.trainer.train_config.batch_size == regressor.trainer.train_config.batch_size
        assert loaded_regressor.trainer.train_config.val_split_ratio == regressor.trainer.train_config.val_split_ratio
        assert loaded_regressor.trainer.train_config.task_type == "regression"
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


def test_multivariate_data():
    torch.manual_seed(42)
    n_train = 80
    n_test = 20
    n_vars = 3
    seq_len = 128

    X_train = torch.randn(n_train, n_vars, seq_len)
    y_train = torch.randn(n_train, 1)
    X_test = torch.randn(n_test, n_vars, seq_len)

    regressor = TirexLinearRegressor(max_epochs=1, batch_size=32)
    regressor.fit((X_train, y_train))
    predictions = regressor.predict(X_test)

    assert predictions.shape == (n_test, 1)
    assert predictions.dtype == torch.float32


################################### TESTS WITH COMPILE #####################################################
def test_compile_initialization():
    regressor = TirexLinearRegressor(compile=True)
    assert regressor._compile is True
    assert regressor.emb_model is not None


def test_compile_fit_and_predict(regression_data):
    X_train, y_train, X_test, _ = regression_data

    regressor = TirexLinearRegressor(compile=True, max_epochs=2, batch_size=32)
    regressor.fit((X_train, y_train))
    predictions = regressor.predict(X_test)

    assert predictions.shape == (len(X_test), 1)
    assert predictions.dtype == torch.float32


def test_save_and_load_with_compile(regression_data):
    X_train, y_train, X_test, _ = regression_data

    # Train and save model with compile=True
    regressor = TirexLinearRegressor(compile=True, max_epochs=1, batch_size=32)
    regressor.fit((X_train, y_train))
    predictions_before = regressor.predict(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pt") as f:
        model_path = f.name

    try:
        regressor.save_model(model_path)

        # Load model
        loaded_regressor = TirexLinearRegressor.load_model(model_path)
        predictions_after = loaded_regressor.predict(X_test)

        # Check predictions match (within numerical precision)
        assert torch.allclose(predictions_before, predictions_after, atol=1e-6)

        # Check compile parameter is preserved
        assert loaded_regressor._compile is True
    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
