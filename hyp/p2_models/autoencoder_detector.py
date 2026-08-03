"""
P2 - Model Training & Tuning (UC2.1)
AutoencoderDetector: FR-MT-01 - undercomplete autoencoder trained on
benign traffic only; anomaly score = per-flow reconstruction error (MSE).

PyTorch implementation of the architecture in the P2 class diagram.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyp.p2_models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class AutoencoderDetector(AnomalyDetector):
    """Undercomplete autoencoder; score = reconstruction MSE per flow."""

    def __init__(
        self,
        latent_dim: int = 16,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 512,
        hidden_dims: list[int] | None = None,
        seed: int = 0,
    ):
        hidden_dims = hidden_dims or [64, 32]
        super().__init__("Autoencoder", {
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
        })
        self.input_dim: int | None = None
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.training_losses: list[float] = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder: nn.Sequential | None = None
        self.decoder: nn.Sequential | None = None

    # ------------------------------------------------------------------ API
    def train(self, train_data: np.ndarray) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_data = np.asarray(train_data, dtype=np.float32)
        self._build_network(train_data.shape[1])

        dataset = TensorDataset(torch.from_numpy(train_data))
        # Deterministic shuffling under the fixed seed:
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimiser = torch.optim.Adam(params, lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.encoder.train(); self.decoder.train()
        self.training_losses = []
        for epoch in range(self.epochs):
            epoch_loss, n_batches = 0.0, 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimiser.zero_grad()
                recon = self._forward(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item(); n_batches += 1
            mean_loss = epoch_loss / max(n_batches, 1)
            self.training_losses.append(mean_loss)
            logger.info("AE epoch %d/%d loss=%.6f", epoch + 1, self.epochs, mean_loss)

        self.is_trained = True
        return {
            "model": self.model_name,
            "epochs": self.epochs,
            "final_loss": self.training_losses[-1],
            "training_losses": self.training_losses,
        }

    @torch.no_grad()
    def score(self, data: np.ndarray) -> np.ndarray:
        self._require_trained()
        data = np.asarray(data, dtype=np.float32)
        self.encoder.eval(); self.decoder.eval()
        scores = []
        for start in range(0, len(data), self.batch_size):
            batch = torch.from_numpy(data[start:start + self.batch_size]).to(self.device)
            scores.append(self._compute_reconstruction_error(batch).cpu().numpy())
        return self._as_float64_vector(np.concatenate(scores), len(data))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "hyperparameters": self.hyperparameters,
            "input_dim": self.input_dim,
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "training_losses": self.training_losses,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.hyperparameters = ckpt["hyperparameters"]
        self.latent_dim = self.hyperparameters["latent_dim"]
        self.hidden_dims = self.hyperparameters["hidden_dims"]
        self._build_network(ckpt["input_dim"])
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.decoder.load_state_dict(ckpt["decoder_state"])
        self.training_losses = ckpt["training_losses"]
        self.is_trained = True

    def get_score_type(self) -> str:
        return "reconstruction_error_mse"

    # -------------------------------------------------------------- helpers
    def _build_network(self, input_dim: int) -> None:
        self.input_dim = input_dim
        enc_layers, prev = [], input_dim
        for h in self.hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers += [nn.Linear(prev, self.latent_dim)]
        self.encoder = nn.Sequential(*enc_layers).to(self.device)

        dec_layers, prev = [], self.latent_dim
        for h in reversed(self.hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec_layers).to(self.device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def _compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self._forward(x)
        return torch.mean((x - recon) ** 2, dim=1)
