
from abc import ABC, abstractmethod
from typing import Dict


class PretrainedModel(ABC):
    REGISTRY: Dict[str, "PretrainedModel"] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.REGISTRY[cls.register_name()] = cls

    @classmethod
    def from_pretrained(cls, path, map_location="cuda:0"):
        return cls.load_from_checkpoint(path, map_location=map_location)

    @classmethod
    @abstractmethod
    def register_name(cls) -> str:
        pass


def load_model(hf_path: str, device: str = "cuda:0"):
    model_cls = PretrainedModel.REGISTRY.get(hf_path, None)
    if model_cls is None:
        raise ValueError(f"No registered model found for {hf_path}")
    return model_cls.from_pretrained(hf_path, map_location=device)
