import pytest
import torch
import numpy as np
from tirex.api_adapter.standard_adapter import _batched_slice, _batched, _batch_pad_iterable, get_batches



@pytest.mark.parametrize("context", [
    np.array([1.0, 2.0, 3.0]),
    torch.tensor([1.0, 2.0, 3.0]),
    [np.array([1.0, 2.0]), np.array([1.0, 5.0, 4.0])],
    [torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0, 4.0, 4.0])]

])

# ----- Tests: Basic batching with Tensor -----

def test_get_batches_various_types(context):
    batches = list(get_batches(context, batch_size=2))
    assert all(isinstance(batch, tuple) and isinstance(batch[0], torch.Tensor) for batch in batches)


def test_single_sample_tensor():
    context = torch.arange(10)
    batches = list(get_batches(context, batch_size=4))
    assert len(batches) == 1
    assert batches[0][0].shape == (1, 10)

def test_fewer_samples_than_batchsize():
    context = torch.arange(6).reshape(2, 3)
    batches = list(get_batches(context, batch_size=5))
    assert len(batches) == 1
    assert batches[0][0].shape == (2, 3)

def test_more_samples_than_batchsize():
    context = torch.arange(45).reshape(9, 5)
    batches = list(get_batches(context, batch_size=5))
    assert len(batches) == 2
    assert batches[0][0].shape == (5, 5)
    assert batches[1][0].shape == (4, 5)


def test_list_of_variable_length_arrays():
    context = [torch.arange(5), torch.arange(8), torch.arange(6)]
    batches = list(get_batches(context, batch_size=3))
    assert len(batches) == 1    
    batch, _ = next(iter(batches))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (3, 8)  # Padded to max length 8
    assert torch.isnan(batch[0, 0:3]).all()  # Left padding
    assert (batch[0, -5:] == torch.tensor([0,1,2,3,4])).all()

def test_list_of_variable_length_arrays_more_than_batchsize_torch():
    context = [torch.arange(5), torch.arange(8), torch.arange(6), torch.arange(12), torch.arange(22)]
    batches = list(get_batches(context, batch_size=3))
    assert len(batches) == 2
    batch_1, _ = batches[0]
    batch_2, _ = batches[1]
    assert isinstance(batch_1, torch.Tensor)
    assert isinstance(batch_2, torch.Tensor)
    assert batch_1.shape == (3, 8)  # Padded to max length 8
    assert batch_2.shape == (2, 22)  # Padded to max length 8
    assert torch.isnan(batch_1[0, 0:3]).all()  # Left padding
    assert (batch_1[0, -5:] == torch.arange(5)).all()
    assert (batch_2[0, -12:] == torch.arange(12)).all()



# ----- Tests: Basic batching with Numpy -----

def test_single_sample_numpy():
    context = np.arange(10)
    batches = list(get_batches(context, batch_size=4))
    assert len(batches) == 1
    assert batches[0][0].shape == (1, 10)

def test_fewer_samples_than_batchsize_np():
    context = np.arange(6).reshape(2, 3)
    batches = list(get_batches(context, batch_size=5))
    assert len(batches) == 1
    assert batches[0][0].shape == (2, 3)

def test_more_samples_than_batchsize_np():
    context = np.arange(45).reshape(9, 5)
    batches = list(get_batches(context, batch_size=5))
    assert len(batches) == 2
    assert batches[0][0].shape == (5, 5)
    assert batches[1][0].shape == (4, 5)


def test_list_of_variable_length_arrays_np():
    context = [np.arange(5), np.arange(8), np.arange(6)]
    batches = list(get_batches(context, batch_size=3))
    assert len(batches) == 1
    batch, _ = next(iter(batches))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (3, 8)  # Padded to max length 8
    assert torch.isnan(batch[0, 0:3]).all()  # Left padding
    assert (batch[0, -5:] == torch.tensor([0,1,2,3,4])).all()


def test_list_of_variable_length_arrays_more_than_batchsize_np():
    context = [np.arange(5), np.arange(8), np.arange(6), np.arange(12), np.arange(22)]
    batches = list(get_batches(context, batch_size=3))
    assert len(batches) == 2
    batch_1, _ = batches[0]
    batch_2, _ = batches[1]
    assert isinstance(batch_1, torch.Tensor)
    assert isinstance(batch_2, torch.Tensor)
    assert batch_1.shape == (3, 8)  # Padded to max length 8
    assert batch_2.shape == (2, 22)  # Padded to max length 8
    assert torch.isnan(batch_1[0, 0:3]).all()  # Left padding
    assert (batch_1[0, -5:] == torch.arange(5)).all()
    assert (batch_2[0, -12:] == torch.arange(12)).all()


# ----- Tests for _batched_slice -----
def test_batched_slice_basic():
    data = list(range(10))
    meta = [{"id": i} for i in range(10)]
    batches = list(_batched_slice(data, meta, 4))
    assert len(batches) == 3
    assert all(isinstance(m, list) and isinstance(d, list) for d, m in batches)
    assert batches[0][0] == [0, 1, 2, 3]
    assert batches[-1][0] == [8, 9]
    assert batches[-1][1] == [{"id": 8}, {"id": 9}]

def test_batched_slice_no_meta():
    data = list(range(3))
    batches = list(_batched_slice(data, None, 2))
    assert len(batches) == 2
    assert all(len(m) == len(d) for d, m in batches)

# ----- Tests for _batched -----
def test_batched_even_split():
    data = list(range(8))
    batches = list(_batched(data, 2))
    assert len(batches) == 4
    assert all(len(batch) == 2 for batch in batches)

def test_batched_uneven_split():
    data = list(range(5))
    batches = list(_batched(data, 2))
    assert all(len(batch) == 2 for batch in batches[0:-1])
    assert batches[-1] == (4,)

# ----- Tests for _batch_pad_iterable -----
def test_batch_pad_iterable_correct_padding():
    data = [
        (torch.tensor([1.0, 2.0]), {}),
        (torch.tensor([3.0, 4.0, 5.0]), {}),
        (torch.tensor([6.0]), {}),
    ]
    batches = list(_batch_pad_iterable(data, batch_size=3))
    assert len(batches) == 1
    padded_batch, meta = batches[0]
    assert padded_batch.shape == (3, 3)
    assert torch.isnan(padded_batch[0, 0]).item()
    assert torch.isnan(padded_batch[2, :2]).all()
    assert padded_batch[2, 2].item() == 6.0

def test_batch_pad_iterable_multiple_batches():
    data = [
        (torch.tensor([i + 1.0]), {}) for i in range(7)
    ]
    batches = list(_batch_pad_iterable(data, batch_size=3))
    assert len(batches) == 3
    for padded_batch, meta in batches:
        assert padded_batch.ndim == 2
        assert padded_batch.shape[1] == 1

