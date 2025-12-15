# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from pathlib import Path

import pytest
import torch

from tirex import load_model
from tirex.models.embedding import TiRexEmbedding


def load_tensor_from_pt_file(path):
    base_path = Path(__file__).parent.resolve() / "data_embeddings"
    return torch.load(base_path / path, weights_only=True)


@pytest.fixture
def tirex_model():
    return load_model("NX-AI/TiRex", backend="torch")


def test_embed_context(tirex_model):
    # Load input data and reference output
    context = load_tensor_from_pt_file("embed_context_data.pt")
    ref_hidden_states = load_tensor_from_pt_file("embed_context_ref.pt")

    # Get embeddings from model
    hidden_states = tirex_model._embed_context(context)

    # Test shape is correct
    assert hidden_states.shape == ref_hidden_states.shape

    # Test output matches reference
    torch.testing.assert_close(hidden_states, ref_hidden_states, rtol=1e-5, atol=2.5e-1)


def test_embed_context(tirex_model):
    # Load input data and reference output
    context = load_tensor_from_pt_file("embed_context_data.pt")
    ref_hidden_states = load_tensor_from_pt_file("embed_context_ref.pt")

    # Get embeddings from model
    hidden_states = tirex_model._embed_context(context)

    # Test shape is correct
    assert hidden_states.shape == ref_hidden_states.shape

    # Test output matches reference
    torch.testing.assert_close(hidden_states, ref_hidden_states, rtol=1e-5, atol=2.5e-1)


def test_tirex_embedding():
    context_data = load_tensor_from_pt_file("context_data_wrapper.pt")
    ref_embedding = load_tensor_from_pt_file("tirex_embedding_ref.pt")

    # TiRexEmbedding with default settings
    embedding_model = TiRexEmbedding()
    embedding = embedding_model(context_data).cpu()

    assert embedding.shape == ref_embedding.shape
    torch.testing.assert_close(embedding, ref_embedding, rtol=1e-5, atol=2e-2)


def test_stats_diff_tirex():
    context_data = load_tensor_from_pt_file("context_data_wrapper.pt")
    ref_embedding = load_tensor_from_pt_file("tirex_embedding_augmented_ref.pt")

    # StatisticsAugmentedEmbedding(DifferenceAugmentedEmbedding(TiRexEmbedding))
    embedding_model = TiRexEmbedding(data_augmentation=True)
    embedding = embedding_model(context_data).cpu()

    assert embedding.shape == ref_embedding.shape
    torch.testing.assert_close(embedding, ref_embedding, rtol=1e-5, atol=2e-2)
