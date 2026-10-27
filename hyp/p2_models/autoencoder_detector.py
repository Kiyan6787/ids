import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyp.p2_models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class AutoencoderDetector(AnomalyDetector):

    model_name = "Autoencoder"

    def __init__(
        self,
        latent_dim: int = 16,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 512,
        hidden_dims: list[int] | None = None,
        seed: int = 0,
    ):
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32]
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
        self.network = self._build_network(train_data.shape[1])

        # Deterministic shuffling under the fixed seed:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_data)), batch_size=self.batch_size,
            shuffle=True, generator=torch.Generator().manual_seed(self.seed),
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
                loss = criterion(self.network(batch), batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            self.training_losses.append(epoch_loss / len(loader))
            logger.info("AE epoch %d/%d loss=%.6f",
                        epoch + 1, self.epochs, self.training_losses[-1])

    @torch.no_grad()
    def score(self, data: np.ndarray) -> np.ndarray:
        data = torch.from_numpy(np.asarray(data, dtype=np.float32))
        self.network.eval()
        scores = []
        for batch in data.split(self.batch_size):
            batch = batch.to(self.device)
            recon = self.network(batch)
            scores.append(torch.mean((batch - recon) ** 2, dim=1).cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    # helpers
    def _build_network(self, input_dim: int) -> nn.Sequential:
        enc_layers, prev = [], input_dim
        for h in self.hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers += [nn.Linear(prev, self.latent_dim)]

        dec_layers, prev = [], self.latent_dim
        for h in reversed(self.hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim)]
        return nn.Sequential(*enc_layers, *dec_layers).to(self.device)
