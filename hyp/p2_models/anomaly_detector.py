"""Abstract contract shared by every anomaly detector."""

from abc import ABC, abstractmethod

import numpy as np


class AnomalyDetector(ABC):

    model_name: str

    @abstractmethod
    def train(self, train_data: np.ndarray) -> None:
        """Fit on benign-only data."""

    @abstractmethod
    def score(self, data: np.ndarray) -> np.ndarray:
        """Return one anomaly score per row: shape (N,), dtype float64."""
