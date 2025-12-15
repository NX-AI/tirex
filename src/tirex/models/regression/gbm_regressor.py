# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import joblib
import torch
from lightgbm import LGBMRegressor, early_stopping

from ..base.base_regressor import BaseTirexRegressor


class TirexGBMRegressor(BaseTirexRegressor):
    """
    A Gradient Boosting regressor that uses time series embeddings as features.

    This regressor combines a pre-trained embedding model for feature extraction with a
    Gradient Boosting regressor.

    Example:
        >>> import torch
        >>> from tirex.models.regression import TirexGBMRegressor
        >>>
        >>> # Create model with custom LightGBM parameters
        >>> model = TirexGBMRegressor(
        ...     data_augmentation=True,
        ...     n_estimators=50,
        ...     random_state=42
        ... )
        >>>
        >>> # Prepare data (can use NumPy arrays or PyTorch tensors)
        >>> X_train = torch.randn(100, 1, 128)  # 100 samples, 1 number of variates, 128 sequence length
        >>> y_train = torch.randn(100,)  # target values
        >>>
        >>> # Train the model
        >>> model.fit((X_train, y_train))
        >>>
        >>> # Make predictions
        >>> X_test = torch.randn(20, 1, 128)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        data_augmentation: bool = False,
        device: str | None = None,
        compile: bool = False,
        batch_size: int = 512,
        early_stopping_rounds: int | None = 10,
        min_delta: float = 0.0,
        val_split_ratio: float = 0.2,
        # LightGBM kwargs
        **lgbm_kwargs,
    ) -> None:
        """Initializes Embedding Based Gradient Boosting Regression model.

        Args:
            data_augmentation : bool
                Whether to use data_augmentation for embeddings (sample statistics and first-order differences of the original data). Default: False
            device : str | None
                Device to run the embedding model on. If None, uses CUDA if available, else CPU. Default: None
            compile: bool
                Whether to compile the frozen embedding model. Default: False
            batch_size : int
                Batch size for embedding calculations. Default: 512
            early_stopping_rounds: int | None
                Number of rounds without improvement of all metrics for Early Stopping. Default: 10
            min_delta: float
                Minimum improvement in score to keep training. Default 0.0
            val_split_ratio : float
                Proportion of training data to use for validation, if validation data are not provided. Default: 0.2
            **lgbm_kwargs
                Additional keyword arguments to pass to LightGBM's LGBMRegressor.
                Common options include n_estimators, max_depth, learning_rate, random_state, etc.
        """
        super().__init__(data_augmentation=data_augmentation, device=device, compile=compile, batch_size=batch_size)

        # Early Stopping callback
        self.early_stopping_rounds = early_stopping_rounds
        self.min_delta = min_delta

        # Data split parameters:
        self.val_split_ratio = val_split_ratio

        # Extract random_state for train_val_split if provided
        self.random_state = lgbm_kwargs.get("random_state", None)

        self.head = LGBMRegressor(**lgbm_kwargs)

    @torch.inference_mode()
    def fit(
        self,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Train the LightGBM regressor on embedded time series data.

        This method generates embeddings for the training data using the embedding
        model, then trains the LightGBM regressor on these embeddings.

        Args:
            train_data: Tuple of (X_train, y_train) where X_train is the input time
                series data (torch.Tensor) and y_train is a torch.Tensor
                of target values.
            val_data: Optional tuple of (X_val, y_val) for validation where X_val is the input time
                series data (torch.Tensor) and y_val is a torch.Tensor
                of target values. If None, validation data will be automatically split from train_data (20% split).
        """

        (X_train, y_train), (X_val, y_val) = self._create_train_val_datasets(
            train_data=train_data,
            val_data=val_data,
            val_split_ratio=self.val_split_ratio,
            seed=self.random_state,
        )

        X_train = X_train.to(self.device)
        X_val = X_val.to(self.device)

        embeddings_train = self._compute_embeddings(X_train)
        embeddings_val = self._compute_embeddings(X_val)

        y_train = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        y_val = y_val.detach().cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val

        self.head.fit(
            embeddings_train,
            y_train,
            eval_set=[(embeddings_val, y_val)],
            callbacks=[early_stopping(stopping_rounds=self.early_stopping_rounds, min_delta=self.min_delta)]
            if self.early_stopping_rounds is not None
            else None,
        )

    def save_model(self, path: str) -> None:
        """This method saves the trained LightGBM regressor head (joblib format) and embedding information.

        Args:
            path: File path where the model should be saved (e.g., 'model.joblib').
        """

        payload = {
            "data_augmentation": self.data_augmentation,
            "compile": self._compile,
            "batch_size": self.batch_size,
            "early_stopping_rounds": self.early_stopping_rounds,
            "min_delta": self.min_delta,
            "val_split_ratio": self.val_split_ratio,
            "head": self.head,
        }
        joblib.dump(payload, path)

    @classmethod
    def load_model(cls, path: str) -> "TirexGBMRegressor":
        """Load a saved model from file.

        This reconstructs the model with the embedding configuration and loads
        the trained LightGBM regressor from a checkpoint file created by save_model().

        Args:
            path: File path to the saved model checkpoint.
        Returns:
            TirexGBMRegressor: The loaded model with trained Gradient Boosting regressor, ready for inference.
        """
        checkpoint = joblib.load(path)

        # Create new instance with saved configuration
        model = cls(
            data_augmentation=checkpoint["data_augmentation"],
            compile=checkpoint["compile"],
            batch_size=checkpoint["batch_size"],
            early_stopping_rounds=checkpoint["early_stopping_rounds"],
            min_delta=checkpoint["min_delta"],
            val_split_ratio=checkpoint["val_split_ratio"],
        )

        # Load the trained LightGBM head
        model.head = checkpoint["head"]

        # Extract random_state from the loaded head if available
        model.random_state = getattr(model.head, "random_state", None)

        return model
