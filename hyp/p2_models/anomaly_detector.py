"""
P2 - Model Training & Tuning (UC2)
AnomalyDetector: the polymorphic contract shared by all three detectors.

This abstract base class is what enforces FR-EV-03 structurally: every
concrete detector exposes the identical train / score / save / load
contract, and score() returns shape (N,) float64 for ALL models
(test 2.3, cross-model uniformity), so the downstream evaluation code
is literally the same code path for each model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AnomalyDetector(ABC):
    """Abstract unsupervised anomaly detector (higher score = more anomalous)."""

    def __init__(self, name: str, hyperparams: dict[str, Any]):
        self.model_name = name
        self.hyperparameters = dict(hyperparams)
        self.is_trained = False

    # ------------------------------------------------------------- contract
    @abstractmethod
    def train(self, train_data: np.ndarray) -> dict:
        """Fit on benign-only data; return training-time metrics."""

    @abstractmethod
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return one anomaly score per row: shape (N,), dtype float64."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model artefact (FR-MT-04)."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Reload a persisted artefact into this instance."""

    @abstractmethod
    def get_score_type(self) -> str:
        """Human-readable score semantics (reconstruction error, etc.)."""

    # ------------------------------------------------------------- concrete
    def get_model_name(self) -> str:
        return self.model_name

    def get_hyperparameters(self) -> dict[str, Any]:
        return dict(self.hyperparameters)

    def is_model_trained(self) -> bool:
        return self.is_trained

    # Shared guards -------------------------------------------------------
    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                f"{self.model_name}: score() called before train()/load()."
            )

    @staticmethod
    def _as_float64_vector(scores: np.ndarray, n_expected: int) -> np.ndarray:
        """Normalise the output contract: shape (N,), float64 (test 2.3)."""
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != n_expected:
            raise ValueError(
                f"Score vector length {scores.shape[0]} != input rows {n_expected}"
            )
        return scores
