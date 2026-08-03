"""
P2 - Model Training & Tuning
ScratchAutoencoderDetector: the pipeline DEVELOPED FROM SCRATCH required
by the AI project guidelines.

No machine-learning library is used anywhere in this file. The forward
pass, mean-squared-error loss, backpropagation gradients, and the Adam
optimiser update rule are all implemented by hand in NumPy. It plugs
into the same AnomalyDetector contract as the library-based detectors,
so the identical downstream pipeline (threshold tuning, evaluation,
comparison) runs unchanged - which is exactly the FR-EV-03 point.

Architecture: input -> hidden (ReLU) -> latent -> hidden (ReLU) -> output
Loss:         mean squared reconstruction error
Optimiser:    Adam (Kingma & Ba, 2015), implemented manually
Score:        per-flow reconstruction MSE (same semantics as the
              PyTorch AutoencoderDetector, enabling a like-for-like
              correctness cross-check between the two pipelines)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from hyp.p2_models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class _DenseLayer:
    """A fully connected layer with manually implemented forward/backward."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.RandomState,
                 activation: str = "linear"):
        # He initialisation for ReLU layers, Xavier for linear.
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.randn(n_in, n_out) * scale
        self.b = np.zeros(n_out)
        self.activation = activation
        # caches for backprop
        self._x: np.ndarray | None = None
        self._z: np.ndarray | None = None
        # gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # Adam moment estimates
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        if self.activation == "relu":
            return np.maximum(self._z, 0.0)
        return self._z

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """grad_out = dL/d(layer output); returns dL/d(layer input)."""
        if self.activation == "relu":
            grad_out = grad_out * (self._z > 0.0)
        self.dW = self._x.T @ grad_out / len(self._x)
        self.db = grad_out.mean(axis=0)
        return grad_out @ self.W.T

    def adam_step(self, lr: float, t: int,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """One Adam update, implemented from the update equations."""
        for p, g, m, v in ((self.W, self.dW, self.mW, self.vW),
                           (self.b, self.db, self.mb, self.vb)):
            m *= beta1; m += (1 - beta1) * g
            v *= beta2; v += (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


class ScratchAutoencoderDetector(AnomalyDetector):
    """Hand-implemented NumPy autoencoder; score = reconstruction MSE."""

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32,
                 lr: float = 1e-3, epochs: int = 30, batch_size: int = 512,
                 seed: int = 0):
        super().__init__("ScratchAutoencoder", {
            "latent_dim": latent_dim, "hidden_dim": hidden_dim,
            "learning_rate": lr, "epochs": epochs,
            "batch_size": batch_size, "seed": seed,
        })
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.layers: list[_DenseLayer] = []
        self.input_dim: int | None = None
        self.training_losses: list[float] = []
        self._adam_t = 0

    # ------------------------------------------------------------------ API
    def train(self, train_data: np.ndarray) -> dict:
        rng = np.random.RandomState(self.seed)
        X = np.asarray(train_data, dtype=np.float64)
        self._build_network(X.shape[1], rng)

        n = len(X)
        self.training_losses = []
        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n, self.batch_size):
                batch = X[perm[start:start + self.batch_size]]
                recon = self._forward(batch)
                # MSE loss over the batch
                diff = recon - batch
                loss = float(np.mean(diff ** 2))
                # dL/d(recon) for L = mean over batch AND features:
                grad = 2.0 * diff / batch.shape[1]
                self._backward(grad)
                self._adam_t += 1
                for layer in self.layers:
                    layer.adam_step(self.learning_rate, self._adam_t)
                epoch_loss += loss; n_batches += 1
            mean_loss = epoch_loss / max(n_batches, 1)
            self.training_losses.append(mean_loss)
            logger.info("ScratchAE epoch %d/%d loss=%.6f",
                        epoch + 1, self.epochs, mean_loss)

        self.is_trained = True
        return {"model": self.model_name, "epochs": self.epochs,
                "final_loss": self.training_losses[-1],
                "training_losses": self.training_losses}

    def score(self, data: np.ndarray) -> np.ndarray:
        self._require_trained()
        X = np.asarray(data, dtype=np.float64)
        scores = []
        for start in range(0, len(X), self.batch_size):
            batch = X[start:start + self.batch_size]
            recon = self._forward(batch)
            scores.append(np.mean((batch - recon) ** 2, axis=1))
        return self._as_float64_vector(np.concatenate(scores), len(X))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "hyperparameters": self.hyperparameters,
                "input_dim": self.input_dim,
                "weights": [(l.W, l.b, l.activation) for l in self.layers],
                "training_losses": self.training_losses,
            }, fh)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            artefact = pickle.load(fh)
        self.hyperparameters = artefact["hyperparameters"]
        self.input_dim = artefact["input_dim"]
        rng = np.random.RandomState(0)
        self.layers = []
        for W, b, activation in artefact["weights"]:
            layer = _DenseLayer(W.shape[0], W.shape[1], rng, activation)
            layer.W, layer.b = W, b
            self.layers.append(layer)
        self.training_losses = artefact["training_losses"]
        self.is_trained = True

    def get_score_type(self) -> str:
        return "reconstruction_error_mse"

    # -------------------------------------------------------------- helpers
    def _build_network(self, input_dim: int, rng: np.random.RandomState) -> None:
        self.input_dim = input_dim
        self.layers = [
            _DenseLayer(input_dim, self.hidden_dim, rng, "relu"),
            _DenseLayer(self.hidden_dim, self.latent_dim, rng, "linear"),
            _DenseLayer(self.latent_dim, self.hidden_dim, rng, "relu"),
            _DenseLayer(self.hidden_dim, input_dim, rng, "linear"),
        ]

    def _forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
