import logging

import numpy as np
from sklearn.ensemble import IsolationForest

from hyp.p2_models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class IsolationForestDetector(AnomalyDetector):

    model_name = "IsolationForest"

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        max_samples: int | str = 256,
        seed: int = 0,
    ):
        self.forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=seed,
            n_jobs=-1,
        )

    def train(self, train_data: np.ndarray) -> None:
        self.forest.fit(np.asarray(train_data, dtype=np.float64))
        logger.info("IF trained: %d trees on %d rows",
                    self.forest.n_estimators, len(train_data))

    def score(self, data: np.ndarray) -> np.ndarray:
        return -self.forest.score_samples(np.asarray(data, dtype=np.float64))
