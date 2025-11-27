import numpy as np
import torch
from sklearn.model_selection import train_test_split


# Remove after Issue will be solved: https://github.com/pytorch/pytorch/issues/61474
def nanmax(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim)
    return output.values


def nanmin(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).min(dim=dim, keepdim=keepdim)
    return output.values


def nanvar(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


def train_val_split(
    train_data: tuple[torch.Tensor, torch.Tensor],
    val_split_ratio: float,
    stratify: bool,
    seed: int | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    idx_train, idx_val = train_test_split(
        np.arange(len(train_data[0])),
        test_size=val_split_ratio,
        random_state=seed,
        shuffle=True,
        stratify=train_data[1] if stratify else None,
    )

    return (
        (train_data[0][idx_train], train_data[1][idx_train]),
        (train_data[0][idx_val], train_data[1][idx_val]),
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        delta: float = 0.0001,
    ) -> None:
        self.patience: int = patience
        self.delta: float = delta

        self.best: float = np.inf
        self.wait_count: int = 0
        self.early_stop: bool = False

    def __call__(self, epoch: int, val_loss: float) -> bool:
        improved = val_loss < (self.best - self.delta)
        if improved:
            self.best = val_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {epoch}.")
        return self.early_stop
