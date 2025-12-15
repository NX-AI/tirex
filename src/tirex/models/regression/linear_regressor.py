# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from dataclasses import asdict

import torch

from ..base.base_regressor import BaseTirexRegressor
from ..trainer import TrainConfig, Trainer, TrainingMetrics


class TirexLinearRegressor(BaseTirexRegressor, torch.nn.Module):
    """
    A PyTorch regressor that combines time series embeddings with a linear regression head.

    This model uses a pre-trained TiRex embedding model to generate feature representations from time series
    data, followed by a linear layer (with optional dropout) for regression. The embedding backbone
    is frozen during training, and only the regression head is trained.

    Example:
        >>> import torch
        >>> from tirex.models.regression import TirexLinearRegressor
        >>>
        >>> # Create model with TiRex embeddings
        >>> model = TirexLinearRegressor(
        ...     data_augmentation=True,
        ...     max_epochs=2,
        ...     lr=1e-4,
        ...     batch_size=32
        ... )
        >>>
        >>> # Prepare data
        >>> X_train = torch.randn(100, 1, 128)  # 100 samples, 1 number of variates, 128 sequence length
        >>> y_train = torch.randn(100, 1)  # target values
        >>>
        >>> # Train the model
        >>> metrics = model.fit((X_train, y_train)) # doctest: +ELLIPSIS
        Epoch 1, Train Loss: ...
        >>> # Make predictions
        >>> X_test = torch.randn(20, 1, 128)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        data_augmentation: bool = False,
        device: str | None = None,
        compile: bool = False,
        # Training parameters
        max_epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 512,
        val_split_ratio: float = 0.2,
        patience: int = 7,
        delta: float = 0.001,
        log_every_n_steps: int = 5,
        seed: int | None = None,
        # Head parameters
        dropout: float | None = None,
    ) -> None:
        """Initializes Embedding Based Linear Regression model.

        Args:
            data_augmentation : bool | None
                Whether to use data_augmentation for embeddings (sample statistics and first-order differences of the original data). Default: False
            device : str | None
                Device to run the embedding model on. If None, uses CUDA if available, else CPU. Default: None
            compile: bool
                Whether to compile the frozen embedding model. Default: False
            max_epochs : int
                Maximum number of training epochs. Default: 10
            lr : float
                Learning rate for the optimizer. Default: 1e-4
            weight_decay : float
                Weight decay coefficient. Default: 0.01
            batch_size : int
                Batch size for training and embedding calculations. Default: 512
            val_split_ratio : float
                Proportion of training data to use for validation, if validation data are not provided. Default: 0.2
            patience : int
                Number of epochs to wait for improvement before early stopping. Default: 7
            delta : float
                Minimum change in validation loss to qualify as an improvement. Default: 0.001
            log_every_n_steps : int
                Frequency of logging during training. Default: 5
            seed : int | None
                Random seed for reproducibility. If None, no seed is set. Default: None
            dropout : float | None
                Dropout probability for the regression head. If None, no dropout is used. Default: None
        """

        torch.nn.Module.__init__(self)

        super().__init__(data_augmentation=data_augmentation, device=device, compile=compile, batch_size=batch_size)

        # Head parameters
        self.dropout = dropout
        self.head = None
        self.emb_dim = None
        self.output_dim = None

        # Train config
        train_config = TrainConfig(
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            device=self.device,
            lr=lr,
            weight_decay=weight_decay,
            class_weights=None,
            task_type="regression",
            batch_size=batch_size,
            val_split_ratio=val_split_ratio,
            patience=patience,
            delta=delta,
            seed=seed,
        )
        self.trainer = Trainer(self, train_config=train_config)

    def _init_regressor(self, emb_dim: int, output_dim: int, dropout: float | None) -> torch.nn.Module:
        if dropout:
            return torch.nn.Sequential(torch.nn.Dropout(p=dropout), torch.nn.Linear(emb_dim, output_dim))
        else:
            return torch.nn.Linear(emb_dim, output_dim)

    @torch.inference_mode()
    def _identify_head_dims(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.emb_dim = self._compute_embeddings(x[:1]).shape[-1]
        self.output_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding model and regression head.

        Args:
            x: Input tensor of time series data with shape (batch_size, num_variates, seq_len).
        Returns:
            torch.Tensor: Predicted values with shape (batch_size, 1).
        Raises:
            RuntimeError: If the regression head has not been initialized via fit().
        """
        if self.head is None:
            raise RuntimeError("Head not initialized. Call fit() first to automatically build the head.")

        embedding = self.emb_model(x).to(self.device)
        return self.head(embedding)

    def fit(
        self, train_data: tuple[torch.Tensor, torch.Tensor], val_data: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> TrainingMetrics:
        """Train the regression head on the provided data.

        This method initializes the regression head based on the data dimensions,
        then trains it on provided data. The embedding model remains frozen.

        Args:
            train_data: Tuple of (X_train, y_train) where X_train is the input time series
                data and y_train are the corresponding target values.
            val_data: Optional tuple of (X_val, y_val) for validation. If None and
                val_split_ratio > 0, validation data will be split from train_data.

        Returns:
            dict[str, float]: Dictionary containing final training and validation losses.
        """
        X_train, y_train = train_data

        self._identify_head_dims(X_train, y_train)
        self.head = self._init_regressor(self.emb_dim, self.output_dim, self.dropout)
        self.head = self.head.to(self.trainer.device)

        return self.trainer.fit(train_data, val_data=val_data)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict values for input time series data.

        Args:
            x: Input tensor of time series data with shape (batch_size, num_variates, seq_len).
        Returns:
            torch.Tensor: Predicted values with shape (batch_size, 1).
        """
        self.eval()
        x = x.to(self.device)
        return self.forward(x)

    def save_model(self, path: str) -> None:
        """Save the trained regression head.

        This function saves the trained regression head weights (.pt format), embedding configuration,
        model dimensions, and device information. The embedding model itself is not
        saved as it uses a pre-trained backbone that can be reloaded.

        Args:
            path: File path where the model should be saved (e.g., 'model.pt').
        """
        train_config_dict = asdict(self.trainer.train_config)
        torch.save(
            {
                "head_state_dict": self.head.state_dict(),  # need to save only head, embedding is frozen
                "data_augmentation": self.data_augmentation,
                "compile": self._compile,
                "emb_dim": self.emb_dim,
                "output_dim": self.output_dim,
                "dropout": self.dropout,
                "train_config": train_config_dict,
            },
            path,
        )

    @classmethod
    def load_model(cls, path: str) -> "TirexLinearRegressor":
        """Load a saved model from file.

        This reconstructs the model architecture and loads the trained weights from
        a checkpoint file created by save_model().

        Args:
            path: File path to the saved model checkpoint.
        Returns:
            TirexLinearRegressor: The loaded model with trained weights, ready for inference.
        """
        checkpoint = torch.load(path)

        # Extract train_config if available, otherwise use defaults
        train_config_dict = checkpoint.get("train_config", {})

        model = cls(
            data_augmentation=checkpoint["data_augmentation"],
            compile=checkpoint["compile"],
            dropout=checkpoint["dropout"],
            max_epochs=train_config_dict.get("max_epochs", 50),
            lr=train_config_dict.get("lr", 1e-4),
            weight_decay=train_config_dict.get("weight_decay", 0.01),
            batch_size=train_config_dict.get("batch_size", 512),
            val_split_ratio=train_config_dict.get("val_split_ratio", 0.2),
            patience=train_config_dict.get("patience", 7),
            delta=train_config_dict.get("delta", 0.001),
            log_every_n_steps=train_config_dict.get("log_every_n_steps", 5),
            seed=train_config_dict.get("seed", None),
        )

        # Initialize head with dimensions
        model.emb_dim = checkpoint["emb_dim"]
        model.output_dim = checkpoint.get("output_dim", checkpoint.get("num_classes", 1))  # Backward compatibility
        model.head = model._init_regressor(model.emb_dim, model.output_dim, model.dropout)

        # Load the trained weights
        model.head.load_state_dict(checkpoint["head_state_dict"])
        model.to(model.device)

        return model
