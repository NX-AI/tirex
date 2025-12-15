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

from tirex.models.classification import TirexLinearClassifier


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
    classifier = TirexLinearClassifier()
    assert classifier.emb_model is not None
    assert isinstance(classifier.trainer.loss_fn, torch.nn.CrossEntropyLoss)


def test_initialization_with_custom_params():
    classifier = TirexLinearClassifier(
        max_epochs=20,
        lr=1e-3,
        batch_size=64,
        dropout=0.2,
    )

    assert classifier.trainer.train_config.max_epochs == 20
    assert classifier.trainer.train_config.lr == 1e-3
    assert classifier.trainer.train_config.batch_size == 64
    assert classifier.dropout == 0.2
    assert isinstance(classifier.trainer.loss_fn, torch.nn.CrossEntropyLoss)


def test_f1_score_on_real_data(classification_data_real):
    X_train, y_train, X_test, y_test = classification_data_real

    classifier = TirexLinearClassifier(max_epochs=5, batch_size=32, seed=42)
    classifier.fit((X_train, y_train))

    predictions = classifier.predict(X_test)

    pred_y = predictions.cpu().numpy()
    test_y = y_test.cpu().numpy()

    f1 = f1_score(test_y, pred_y, average="macro")

    assert f1 >= 0.85, f"F1-score {f1:.4f} is below the required threshold of 0.85"


def test_fit_with_val_data(classification_data):
    X_train, y_train, X_test, y_test = classification_data

    classifier = TirexLinearClassifier(
        max_epochs=1,
        batch_size=32,
    )

    classifier.fit((X_train, y_train), val_data=(X_test, y_test))
    assert classifier.head is not None


def test_predict_proba(classification_data):
    X_train, y_train, X_test, y_test = classification_data
    n_classes = 3

    classifier = TirexLinearClassifier(max_epochs=1, batch_size=32)
    classifier.fit((X_train, y_train))
    probabilities = classifier.predict_proba(X_test)

    assert isinstance(probabilities, torch.Tensor)
    assert probabilities.shape == (len(X_test), n_classes)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(len(X_test), dtype=probabilities.dtype), atol=1e-6)


def test_predict_before_fit_raises_error():
    classifier = TirexLinearClassifier()
    X_test = torch.randn(10, 1, 128)

    with pytest.raises(RuntimeError, match="Head not initialized"):
        classifier.predict(X_test)


def test_forward_pass(classification_data):
    X_train, y_train, X_test, y_test = classification_data

    classifier = TirexLinearClassifier(max_epochs=1, batch_size=32)
    classifier.fit((X_train, y_train))
    logits = classifier.forward(X_test[:5])

    assert logits.shape == (5, 3)
    assert not torch.isnan(logits).any()


def test_save_and_load_model(classification_data):
    X_train, y_train, X_test, y_test = classification_data

    # Train and save model
    classifier = TirexLinearClassifier(
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
    probabilities_before = classifier.predict_proba(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pt") as f:
        model_path = f.name

    try:
        classifier.save_model(model_path)

        # Load model
        loaded_classifier = TirexLinearClassifier.load_model(model_path)
        predictions_after = loaded_classifier.predict(X_test)
        probabilities_after = loaded_classifier.predict_proba(X_test)

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

        # check probability match
        assert torch.allclose(probabilities_before, probabilities_after, atol=1e-8)
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

    classifier = TirexLinearClassifier(max_epochs=1, batch_size=32)
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
    classifier = TirexLinearClassifier(compile=True)
    assert classifier._compile is True
    assert classifier.emb_model is not None


def test_compile_fit_and_predict(classification_data):
    X_train, y_train, X_test, _ = classification_data

    classifier = TirexLinearClassifier(compile=True, max_epochs=2, batch_size=32)
    classifier.fit((X_train, y_train))
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

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
    classifier = TirexLinearClassifier(compile=True, max_epochs=1, batch_size=32)
    classifier.fit((X_train, y_train))
    predictions_before = classifier.predict(X_test)
    probabilities_before = classifier.predict_proba(X_test)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".pt") as f:
        model_path = f.name

    try:
        classifier.save_model(model_path)

        # Load model
        loaded_classifier = TirexLinearClassifier.load_model(model_path)
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


def test_class_weights_parameter(classification_data):
    X_train, y_train, X_test, _ = classification_data
    n_classes = 3

    # Create class weights
    class_weights = torch.tensor([1.0, 2.0, 3.0])

    classifier = TirexLinearClassifier(max_epochs=2, batch_size=32, class_weights=class_weights)
    classifier.fit((X_train, y_train))
    predictions = classifier.predict(X_test)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (len(X_test),)
    assert torch.all((predictions >= 0) & (predictions < n_classes))
    assert torch.allclose(classifier.trainer.train_config.class_weights, class_weights)
