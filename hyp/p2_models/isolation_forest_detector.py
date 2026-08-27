"""scikit-learn Isolation Forest detector; anomaly score = negated
normalised path length. Serves as the classical benchmark against
which the deep models are compared."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from hyp.p2_models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class IsolationForestDetector(AnomalyDetector):
    """scikit-learn Isolation Forest; score = -score_samples(x)."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: int = 0,
        max_samples: int | str = 256,
    ):
        super().__init__("IsolationForest", {
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": random_state,
            "max_samples": max_samples,
        })
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.max_samples = max_samples
        self.forest: IsolationForest | None = None

    # ------------------------------------------------------------------ API
    def train(self, train_data: np.ndarray) -> dict:
        train_data = np.asarray(train_data, dtype=np.float64)
        self.forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples=self.max_samples,
            n_jobs=-1,
        )
        self.forest.fit(train_data)
        self.is_trained = True
        logger.info("IF trained: %d trees on %d rows", self.n_estimators, len(train_data))
        return {"model": self.model_name, "n_estimators": self.n_estimators,
                "n_train_rows": int(len(train_data))}

    def score(self, data: np.ndarray) -> np.ndarray:
        self._require_trained()
        data = np.asarray(data, dtype=np.float64)
        # score_samples: higher = more normal, so negate to match the
        # "higher = more anomalous" convention shared by all detectors.
        scores = -self.forest.score_samples(data)
        return self._as_float64_vector(scores, len(data))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"hyperparameters": self.hyperparameters,
                     "forest": self.forest}, path)

    def load(self, path: str) -> None:
        artefact = joblib.load(path)
        self.hyperparameters = artefact["hyperparameters"]
        self.forest = artefact["forest"]
        self.is_trained = True

    def get_score_type(self) -> str:
        return "negated_path_length"
