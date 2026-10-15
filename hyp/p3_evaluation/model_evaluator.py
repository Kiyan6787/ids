import logging

import numpy as np

from hyp.p2_models.anomaly_detector import AnomalyDetector
from hyp.p3_evaluation.metrics_calculator import MetricsCalculator as MC

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Scores flows and evaluates detection performance for one model."""

    def __init__(self, model: AnomalyDetector, threshold: float, manifest_hash: str):
        self.model = model
        self.threshold = float(threshold)
        self.manifest_hash = manifest_hash

    def evaluate(self, test_data: np.ndarray, labels: np.ndarray) -> dict:
        labels = np.asarray(labels).reshape(-1).astype(int)
        scores = self.model.score(test_data)
        predictions = (scores > self.threshold).astype(int)

        cm = MC.compute_confusion_matrix(labels, predictions)
        metrics = MC.compute_metrics(cm)
        logger.info(
            "%s: P=%.3f R=%.3f F1=%.3f FPR=%.4f", self.model.model_name,
            metrics["precision"], metrics["recall"], metrics["f1"], metrics["fpr"],
        )
        return {
            "model_name": self.model.model_name,
            "threshold": self.threshold,
            "manifest_hash": self.manifest_hash,
            "n_test_rows": len(labels),
            "confusion_matrix": cm,
            "metrics": metrics,
        }
