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
        self.network = None

    # API
    def train(self, train_data: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        train_data = np.asarray(train_data, dtype=np.float32)
        windows = self._create_windows(train_data)
        self.network = self._build_network(train_data.shape[1])

        loader = DataLoader(
            TensorDataset(torch.from_numpy(windows)),
            batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.network.train()
        self.training_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self._forward(batch), batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            self.training_losses.append(epoch_loss / len(loader))
            logger.info("LSTM epoch %d/%d loss=%.6f",
                        epoch + 1, self.epochs, self.training_losses[-1])

    @torch.no_grad()
    def score(self, data: np.ndarray) -> np.ndarray:
        windows = self._create_windows(np.asarray(data, dtype=np.float32))
        self.network.eval()
        errors = []
        for start in range(0, len(windows), self.batch_size):
            batch = torch.from_numpy(windows[start:start + self.batch_size]).to(self.device)
            # per-flow error inside each window: shape (batch, seq_len)
            errors.append(torch.mean((self._forward(batch) - batch) ** 2, dim=2).cpu().numpy())
        errors = np.concatenate(errors)

        return np.concatenate([errors[0, :-1], errors[:, -1]]).astype(np.float64)

    # helpers
    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        L = self.seq_len
        if len(data) < L:
            raise ValueError(f"Need at least seq_len={L} rows, got {len(data)}.")
        return np.stack([data[i:i + L] for i in range(len(data) - L + 1)])

    def _build_network(self, n_features: int) -> nn.ModuleDict:
        return nn.ModuleDict({
            "encoder": nn.LSTM(n_features, self.hidden_dim, self.num_layers,
                               batch_first=True),
            "decoder": nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers,
                               batch_first=True),
            "output": nn.Linear(self.hidden_dim, n_features),
        }).to(self.device)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.network["encoder"](x)
        latent = hidden[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        out, _ = self.network["decoder"](latent)
        return self.network["output"](out)
