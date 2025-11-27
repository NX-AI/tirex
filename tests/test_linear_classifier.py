# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import os
import tempfile

import numpy as np
import pytest
import torch

from tirex.models.classification import TirexClassifierTorch


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
    classifier = TirexClassifierTorch()
    assert classifier.emb_model is not None


def test_initialization_with_custom_params():
    classifier = TirexClassifierTorch(
        max_epochs=20,
        lr=1e-3,
        batch_size=64,
        dropout=0.2,
    )

    assert classifier.trainer.train_config.max_epochs == 20
    assert classifier.trainer.train_config.lr == 1e-3
    assert classifier.trainer.train_config.batch_size == 64
    assert classifier.dropout == 0.2


def test_fit_basic(classification_data):
    X_train, y_train, _, _ = classification_data

    classifier = TirexClassifierTorch(
        max_epochs=1,
        batch_size=32,
        log_every_n_steps=5,
    )

    classifier.fit((X_train, y_train))

    assert classifier.head is not None
    assert classifier.emb_dim is not None
    assert classifier.num_classes == 3


def test_fit_with_val_data(classification_data):
    X_train, y_train, X_test, y_test = classification_data

    classifier = TirexClassifierTorch(
        max_epochs=1,
        batch_size=32,
    )

    classifier.fit((X_train, y_train), val_data=(X_test, y_test))
    assert classifier.head is not None


def test_predict_shape(classification_data):
    X_train, y_train, X_test, y_test = classification_data

    classifier = TirexClassifierTorch(max_epochs=1, batch_size=32)
    classifier.fit((X_train, y_train))
    predictions = classifier.predict(X_test)

    assert predictions.shape == (len(X_test),)
    assert torch.all((predictions >= 0) & (predictions < 3))


def test_predict_before_fit_raises_error():
    classifier = TirexClassifierTorch()
    X_test = torch.randn(10, 1, 128)

    with pytest.raises(RuntimeError, match="Head not initialized"):
        classifier.predict(X_test)


def test_forward_pass(classification_data):
    X_train, y_train, X_test, y_test = classification_data

    classifier = TirexClassifierTorch(max_epochs=1, batch_size=32)
    classifier.fit((X_train, y_train))
    logits = classifier.forward(X_test[:5])

    assert logits.shape == (5, 3)
    assert not torch.isnan(logits).any()


def test_save_and_load_model(classification_data):
    X_train, y_train, X_test, y_test = classification_data

    # Train and save model
    classifier = TirexClassifierTorch(
        max_epochs=1,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        val_split_ratio=0.1,
        seed=42,
        class_weights=torch.tensor([1.0, 2.0, 3.0]),
    )
    classifier.fit((X_train, y_train))
    predictions_before = classifier.predict(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pt") as f:
        model_path = f.name

    try:
        classifier.save_model(model_path)

        # Load model
        loaded_classifier = TirexClassifierTorch.load_model(model_path)
        predictions_after = loaded_classifier.predict(X_test)

        # Check predictions match
        assert torch.all(predictions_before == predictions_after)
        assert loaded_classifier.emb_dim == classifier.emb_dim
        assert loaded_classifier.num_classes == classifier.num_classes
        assert loaded_classifier.dropout == classifier.dropout
        assert loaded_classifier.trainer.train_config.max_epochs == classifier.trainer.train_config.max_epochs
        assert loaded_classifier.trainer.train_config.lr == classifier.trainer.train_config.lr
        assert loaded_classifier.trainer.train_config.batch_size == classifier.trainer.train_config.batch_size
        assert loaded_classifier.trainer.train_config.val_split_ratio == classifier.trainer.train_config.val_split_ratio
        assert loaded_classifier.trainer.train_config.stratify == classifier.trainer.train_config.stratify
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
    n_classes = 2

    X_train = torch.randn(n_train, n_vars, seq_len)
    y_train = torch.randint(0, n_classes, (n_train,))
    X_test = torch.randn(n_test, n_vars, seq_len)

    classifier = TirexClassifierTorch(max_epochs=1, batch_size=32)
    classifier.fit((X_train, y_train))
    predictions = classifier.predict(X_test)

    assert predictions.shape == (n_test,)
    assert torch.all((predictions >= 0) & (predictions < n_classes))
