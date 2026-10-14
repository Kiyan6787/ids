import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyp.p2_models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class LSTMDetector(AnomalyDetector):

    model_name = "LSTM"

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
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.training_losses: list[float] = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lstm = None 
        self.output_layer = None

    # API
    def train(self, train_data: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        train_data = np.asarray(train_data, dtype=np.float32)
        X, y = self._create_sequences(train_data)

        n_features = train_data.shape[1]
        self.lstm = nn.LSTM(n_features, self.hidden_dim, self.num_layers,
                            batch_first=True).to(self.device)
        self.output_layer = nn.Linear(self.hidden_dim, n_features).to(self.device)

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        params = list(self.lstm.parameters()) + list(self.output_layer.parameters())
        optimiser = torch.optim.Adam(params, lr=self.lr)
        criterion = nn.MSELoss()

        self.lstm.train()
        self.output_layer.train()
        self.training_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self._forward(xb), yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            self.training_losses.append(epoch_loss / len(loader))
            logger.info("LSTM epoch %d/%d loss=%.6f",
                        epoch + 1, self.epochs, self.training_losses[-1])

    @torch.no_grad()
    def score(self, data: np.ndarray) -> np.ndarray:
        X, y = self._create_sequences(np.asarray(data, dtype=np.float32))
        self.lstm.eval()
        self.output_layer.eval()
        errors = []
        for start in range(0, len(X), self.batch_size):
            xb = torch.from_numpy(X[start:start + self.batch_size]).to(self.device)
            yb = torch.from_numpy(y[start:start + self.batch_size]).to(self.device)
            errors.append(torch.mean((self._forward(xb) - yb) ** 2, dim=1).cpu().numpy())
        errors = np.concatenate(errors)

        warm_up = np.full(self.seq_len, errors[0])
        return np.concatenate([warm_up, errors]).astype(np.float64)

    # helpers
    def _create_sequences(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sliding windows: X[i] = flows[i : i+L], y[i] = flows[i+L]."""
        L = self.seq_len
        if len(data) <= L:
            raise ValueError(f"Need more than seq_len={L} rows, got {len(data)}.")
        X = np.stack([data[i:i + L] for i in range(len(data) - L)])
        return X, data[L:]

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.output_layer(out[:, -1]) 
