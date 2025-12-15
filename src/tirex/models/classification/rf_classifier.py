# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import joblib
import torch
from sklearn.ensemble import RandomForestClassifier

from ..base.base_classifier import BaseTirexClassifier


class TirexRFClassifier(BaseTirexClassifier):
    """
    A Random Forest classifier that uses time series embeddings as features.

    This classifier combines a pre-trained embedding model for feature extraction with a scikit-learn
    Random Forest classifier. The embedding model generates fixed-size feature vectors from variable-length
    time series, which are then used to train the Random Forest.

    Example:
        >>> from tirex.models.classification import TirexRFClassifier
        >>>
        >>> # Create model with custom Random Forest parameters
        >>> model = TirexRFClassifier(
        ...     data_augmentation=True,
        ...     n_estimators=50,
        ...     max_depth=10,
        ...     random_state=42
        ... )
        >>>
        >>> # Prepare data (can use NumPy arrays or PyTorch tensors)
        >>> X_train = torch.randn(100, 1, 128)  # 100 samples, 1 number of variates, 128 sequence length
        >>> y_train = torch.randint(0, 3, (100,))  # 3 classes
        >>>
        >>> # Train the model
        >>> model.fit((X_train, y_train))
        >>>
        >>> # Make predictions
        >>> X_test = torch.randn(20, 1, 128)
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
    """

    def __init__(
        self,
        data_augmentation: bool = False,
        device: str | None = None,
        compile: bool = False,
        batch_size: int = 512,
        # Random Forest parameters
        **rf_kwargs,
    ) -> None:
        """Initializes Embedding Based Random Forest Classification model.

        Args:
            data_augmentation : bool
                Whether to use data_augmentation for embeddings (sample statistics and first-order differences of the original data). Default: False
            device : str | None
                Device to run the embedding model on. If None, uses CUDA if available, else CPU. Default: None
            compile: bool
                Whether to compile the frozen embedding model. Default: False
            batch_size : int
                Batch size for embedding calculations. Default: 512
            **rf_kwargs
                Additional keyword arguments to pass to sklearn's RandomForestClassifier.
                Common options include n_estimators, max_depth, min_samples_split, random_state, etc.
        """
        super().__init__(data_augmentation=data_augmentation, device=device, compile=compile, batch_size=batch_size)
        self.head = RandomForestClassifier(**rf_kwargs)

    @torch.inference_mode()
    def fit(self, train_data: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Train the Random Forest classifier on embedded time series data.

        This method generates embeddings for the training data using the embedding
        model, then trains the Random Forest on these embeddings.

        Args:
            train_data: Tuple of (X_train, y_train) where X_train is the input time
                series data (torch.Tensor) and y_train is a torch.Tensor
                of class labels.
        """
        X_train, y_train = train_data

        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        X_train = X_train.to(self.device)
        embeddings = self._compute_embeddings(X_train)

        self.head.fit(embeddings, y_train)

    def save_model(self, path: str) -> None:
        """This method saves the trained Random Forest classifier head and embedding information in joblib format

        Args:
            path: File path where the model should be saved (e.g., 'model.joblib').
        """
        payload = {
            "data_augmentation": self.data_augmentation,
            "compile": self._compile,
            "batch_size": self.batch_size,
            "head": self.head,
        }
        joblib.dump(payload, path)

    @classmethod
    def load_model(cls, path: str) -> "TirexRFClassifier":
        """Load a saved model from file.

        This reconstructs the model with the embedding configuration and loads
        the trained Random Forest classifier from a checkpoint file created by save_model().

        Args:
            path: File path to the saved model checkpoint.
        Returns:
            TirexRFClassifier: The loaded model with trained Random Forest, ready for inference.
        """
        checkpoint = joblib.load(path)

        # Create new instance with saved configuration
        model = cls(
            data_augmentation=checkpoint["data_augmentation"],
            compile=checkpoint["compile"],
            batch_size=checkpoint["batch_size"],
        )

        # Load the trained Random Forest head
        model.head = checkpoint["head"]

        return model
