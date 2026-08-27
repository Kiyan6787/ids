"""LSTM next-step predictor trained on benign traffic sequences only;
anomaly score = per-flow prediction error.

Sequence construction: sliding windows of ``sequence_length`` consecutive
flows predict the following flow. Each flow's score is the MSE between
the model's prediction of that flow and the flow itself.

Boundary handling: the first ``sequence_length`` flows of a scored set
have no full history window; they are assigned the score of the first
scoreable flow so the output stays one score per input row.
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


class LSTMDetector(AnomalyDetector):
    """LSTM next-flow predictor; score = prediction MSE per flow."""

    def __init__(
        self,
        seq_len: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 256,
        seed: int = 0,
    ):
        super().__init__("LSTM", {
            "sequence_length": seq_len,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
        })
        self.input_dim: int | None = None
        self.sequence_length = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.training_losses: list[float] = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lstm_layers: nn.LSTM | None = None
        self.output_layer: nn.Linear | None = None

    # ------------------------------------------------------------------ API
    def train(self, train_data: np.ndarray) -> dict:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        train_data = np.asarray(train_data, dtype=np.float32)
        self._build_network(train_data.shape[1])

        X, y = self._create_sequences(train_data)
        if len(X) == 0:
            raise ValueError(
                f"Training data ({len(train_data)} rows) shorter than "
                f"sequence_length={self.sequence_length}."
            )
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        params = list(self.lstm_layers.parameters()) + list(self.output_layer.parameters())
        optimiser = torch.optim.Adam(params, lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.lstm_layers.train(); self.output_layer.train()
        self.training_losses = []
        for epoch in range(self.epochs):
            epoch_loss, n_batches = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                pred = self._forward(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item(); n_batches += 1
            mean_loss = epoch_loss / max(n_batches, 1)
            self.training_losses.append(mean_loss)
            logger.info("LSTM epoch %d/%d loss=%.6f", epoch + 1, self.epochs, mean_loss)

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
        n = len(data)
        self.lstm_layers.eval(); self.output_layer.eval()

        X, y = self._create_sequences(data)
        if len(X) == 0:
            raise ValueError(
                f"Cannot score {n} rows with sequence_length={self.sequence_length}."
            )
        errors = []
        for start in range(0, len(X), self.batch_size):
            xb = torch.from_numpy(X[start:start + self.batch_size]).to(self.device)
            yb = torch.from_numpy(y[start:start + self.batch_size]).to(self.device)
            errors.append(self._compute_prediction_error(xb, yb).cpu().numpy())
        errors = np.concatenate(errors)  # one error per predicted flow

        # Pad the un-scoreable warm-up prefix so output stays (N,).
        scores = np.empty(n, dtype=np.float64)
        scores[self.sequence_length:] = errors
        scores[: self.sequence_length] = errors[0]
        return self._as_float64_vector(scores, n)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "hyperparameters": self.hyperparameters,
            "input_dim": self.input_dim,
            "lstm_state": self.lstm_layers.state_dict(),
            "output_state": self.output_layer.state_dict(),
            "training_losses": self.training_losses,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.hyperparameters = ckpt["hyperparameters"]
        self.sequence_length = self.hyperparameters["sequence_length"]
        self.hidden_dim = self.hyperparameters["hidden_dim"]
        self.num_layers = self.hyperparameters["num_layers"]
        self._build_network(ckpt["input_dim"])
        self.lstm_layers.load_state_dict(ckpt["lstm_state"])
        self.output_layer.load_state_dict(ckpt["output_state"])
        self.training_losses = ckpt["training_losses"]
        self.is_trained = True

    def get_score_type(self) -> str:
        return "prediction_error_mse"

    # -------------------------------------------------------------- helpers
    def _build_network(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.lstm_layers = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        ).to(self.device)
        self.output_layer = nn.Linear(self.hidden_dim, input_dim).to(self.device)

    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sliding windows: X[i] = flows[i : i+L], y[i] = flows[i+L]."""
        L = self.sequence_length
        n = len(data)
        if n <= L:
            return (np.empty((0, L, data.shape[1]), dtype=np.float32),
                    np.empty((0, data.shape[1]), dtype=np.float32))
        # Strided view -> copy, keeps memory bounded for laptop-scale runs.
        X = np.stack([data[i:i + L] for i in range(n - L)]).astype(np.float32)
        y = data[L:].astype(np.float32)
        return X, y

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm_layers(x)          # (B, L, H)
        return self.output_layer(out[:, -1])  # predict next flow from last state

    def _compute_prediction_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self._forward(x)
        return torch.mean((pred - y) ** 2, dim=1)
