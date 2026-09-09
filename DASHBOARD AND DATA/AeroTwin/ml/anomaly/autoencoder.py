"""
AeroTwin-4 Model 3: PyTorch Feed-Forward Autoencoder Anomaly Detector.

Trained strictly using healthy training feature vectors to minimize MSE reconstruction error.
Higher reconstruction error = HIGHER ANOMALY SCORE.
"""

from typing import Optional, Dict, Any, Tuple, List
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class AutoencoderNet(nn.Module):
    """
    Compact Feed-Forward Autoencoder Network.
    """

    def __init__(self, input_dim: int, hidden_dim1: int = 32, latent_dim: int = 16):
        super(AutoencoderNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderAnomalyDetector:
    """
    PyTorch Reconstruction Autoencoder Anomaly Detector.
    """

    def __init__(
        self,
        hidden_dim1: int = 32,
        latent_dim: int = 16,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_seed: int = 42,
    ):
        self.hidden_dim1 = hidden_dim1
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net: Optional[AutoencoderNet] = None
        self.input_dim: int = 0
        self.feature_names: Optional[list] = None
        self.threshold: float = 0.5
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, X_val_healthy: Optional[pd.DataFrame] = None) -> "AutoencoderAnomalyDetector":
        """
        Train PyTorch Autoencoder on Healthy Training features.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_mat = X_train.values
        else:
            X_mat = np.array(X_train)

        self.input_dim = X_mat.shape[1]
        self.net = AutoencoderNet(self.input_dim, self.hidden_dim1, self.latent_dim)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        tensor_train = torch.tensor(X_mat, dtype=torch.float32)

        best_loss = float("inf")
        patience = 10
        patience_counter = 0

        self.net.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(tensor_train.size(0))
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, tensor_train.size(0), self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_x = tensor_train[indices]

                optimizer.zero_grad()
                outputs = self.net(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(1, num_batches)

            # Early stopping check if val set provided
            if X_val_healthy is not None:
                val_mat = X_val_healthy[self.feature_names].values if isinstance(X_val_healthy, pd.DataFrame) else np.array(X_val_healthy)
                tensor_val = torch.tensor(val_mat, dtype=torch.float32)
                self.net.eval()
                with torch.no_grad():
                    val_out = self.net(tensor_val)
                    val_loss = criterion(val_out, tensor_val).item()
                self.net.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        self.net.eval()
        self.is_fitted = True
        return self

    def compute_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute reconstruction MSE loss per sample.
        Higher loss = MORE ANOMALOUS.
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError("AutoencoderAnomalyDetector is not fitted!")

        X_mat = X[self.feature_names].values if isinstance(X, pd.DataFrame) else np.array(X)
        tensor_x = torch.tensor(X_mat, dtype=torch.float32)

        self.net.eval()
        with torch.no_grad():
            reconstructed = self.net(tensor_x)
            # Sample-wise MSE loss across features
            errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()

        return errors

    def fit_threshold(self, X_val_healthy: pd.DataFrame, target_fpr: float = 0.05) -> float:
        """
        Derive decision threshold on healthy validation data to target maximum FPR.
        """
        val_scores = self.compute_anomaly_score(X_val_healthy)
        percentile = (1.0 - target_fpr) * 100.0
        self.threshold = float(np.percentile(val_scores, percentile))
        return self.threshold

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict (anomaly_score, anomaly_flag) for feature matrix X.
        """
        scores = self.compute_anomaly_score(X)
        flags = scores > self.threshold
        return scores, flags

    def save(self, model_dir: str):
        """
        Save PyTorch model weights and config JSON.
        """
        os.makedirs(model_dir, exist_ok=True)
        weights_path = os.path.join(model_dir, "autoencoder_weights.pt")
        config_path = os.path.join(model_dir, "autoencoder_config.json")

        torch.save(self.net.state_dict(), weights_path)
        config = {
            "input_dim": self.input_dim,
            "hidden_dim1": self.hidden_dim1,
            "latent_dim": self.latent_dim,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "is_fitted": self.is_fitted,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load(self, model_dir: str) -> "AutoencoderAnomalyDetector":
        """
        Load PyTorch model weights and config JSON.
        """
        weights_path = os.path.join(model_dir, "autoencoder_weights.pt")
        config_path = os.path.join(model_dir, "autoencoder_config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.input_dim = config["input_dim"]
        self.hidden_dim1 = config["hidden_dim1"]
        self.latent_dim = config["latent_dim"]
        self.feature_names = config["feature_names"]
        self.threshold = float(config["threshold"])
        self.is_fitted = config["is_fitted"]

        self.net = AutoencoderNet(self.input_dim, self.hidden_dim1, self.latent_dim)
        self.net.load_state_dict(torch.load(weights_path, weights_only=True))
        self.net.eval()
        return self
