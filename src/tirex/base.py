# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
import os
from abc import ABC, abstractmethod
from typing import Literal, TypeVar

import torch
from huggingface_hub import hf_hub_download

from tirex.models.slstm.cell import sLSTMCellTorch

T = TypeVar("T", bound="PretrainedModel")
VERSION_DELIMITER = "-"


def skip_cuda():
    return os.getenv("TIREX_NO_CUDA", "False").lower() in ("true", "1", "t")


def xlstm_available():
    try:
        from xlstm.blocks.slstm.cell import sLSTMCellConfig, sLSTMCellFuncGenerator

        return True
    except ModuleNotFoundError:
        return False


def parse_hf_repo_id(path):
    parts = path.split("/")
    return "/".join(parts[0:2])


def parse_model_string(model_string):
    if VERSION_DELIMITER in model_string:
        parts = model_string.split(VERSION_DELIMITER)
        model_id, version = parts[0], parts[0]
    else:
        model_id = model_string
        version = None

    return model_id, version


class PretrainedModel(ABC):
    REGISTRY: dict[str, "PretrainedModel"] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.REGISTRY[cls.register_name()] = cls

    @classmethod
    def from_pretrained(
        cls: type[T], path: str, backend: str, device: str | None = None, compile=False, hf_kwargs=None, ckp_kwargs=None
    ) -> T:
        if hf_kwargs is None:
            hf_kwargs = {}
        if ckp_kwargs is None:
            ckp_kwargs = {}
        if device is None:
            device = "cuda:0" if backend == "cuda" else "cpu"
        if os.path.exists(path):
            print("Loading weights from local directory")
            checkpoint_path = path
        else:
            repo_id = parse_hf_repo_id(path)
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.ckpt", **hf_kwargs)

        # load lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, **ckp_kwargs, weights_only=True)
        model: T = cls(backend=backend, **checkpoint["hyper_parameters"])
        model.on_load_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)

        if compile and backend == "torch":
            compiled_slstm_forward = torch.compile(sLSTMCellTorch.slstm_forward)
            for block in model.blocks:
                block.slstm_layer.slstm_cell._impl_forward_torch = compiled_slstm_forward
        return model

    @classmethod
    @abstractmethod
    def register_name(cls) -> str:
        pass

    def on_load_checkpoint(self):
        pass


def load_model(
    path: str,
    device: str | None = None,
    backend: Literal["torch", "cuda"] | None = None,
    compile: bool = False,
    hf_kwargs=None,
    ckp_kwargs=None,
) -> PretrainedModel:
    """Loads a TiRex model. This function attempts to load the specified model.

    Args:
        path (str): Hugging Face path to the model (e.g. NX-AI/TiRex)
        device (str, optional): The device on which to load the model (e.g., "cuda:0", "cpu").
        backend (torch | cuda): What backend to use, torch or the custom CUDA kernels. Defaults to cuda when xlstm is installed, else torch.
        compile (bool, optional): toch.compile the sLSTM cells, only works with the torch backend
        hf_kwargs (dict, optional): Keyword arguments to pass to the Hugging Face Hub download method.
        ckp_kwargs (dict, optional): Keyword arguments to pass when loading the checkpoint.

    Returns:
        PretrainedModel: The loaded model.

    Examples:
        model: ForecastModel = load_model("NX-AI/TiRex")
    """

    if backend is None:
        backend = "torch" if skip_cuda() or not xlstm_available() else "cuda"

    try:
        _, model_string = parse_hf_repo_id(path).split("/")
        model_id, version = parse_model_string(model_string)
    except:
        raise ValueError(f"Invalid model path {path}")
    model_cls = PretrainedModel.REGISTRY.get(model_id, None)
    if model_cls is None:
        raise ValueError(f"Invalid model id {model_id}")

    return model_cls.from_pretrained(
        path, device=device, backend=backend, compile=compile, hf_kwargs=hf_kwargs, ckp_kwargs=ckp_kwargs
    )
