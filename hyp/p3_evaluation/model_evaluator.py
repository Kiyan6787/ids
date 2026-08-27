"""Scores the test split with any AnomalyDetector, applies the tuned
threshold, and assembles the full per-model results dictionary consumed
by ResultsAnalyser and ResultsExporter.

Each result carries the preprocessing manifest hash, so ResultsAnalyser
can verify all models were evaluated under identical conditions.
"""

from __future__ import annotations

import logging

import numpy as np

from hyp.p2_models.anomaly_detector import AnomalyDetector
from hyp.p3_evaluation.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Scores flows and evaluates detection performance for one model."""

    def __init__(self, model: AnomalyDetector, threshold: float, manifest_hash: str):
        self.model = model
        self.threshold = float(threshold)
        self.metrics_calc = MetricsCalculator()
        self.manifest_hash = manifest_hash
        self.results: dict = {}

    def evaluate(self, test_data: np.ndarray, labels: np.ndarray) -> dict:
        """Score -> threshold -> confusion matrix -> metrics -> ROC/PR
        curves. Returns the assembled results dict."""
        labels = np.asarray(labels).reshape(-1).astype(int)
        scores, predictions = self._score_and_predict(test_data)
        if scores.shape != labels.shape:
            raise ValueError("scores and labels are misaligned")

        cm = self.metrics_calc.compute_confusion_matrix(labels, predictions)
        metrics = self.metrics_calc.compute_metrics(cm)
        roc = self.metrics_calc.compute_roc_curve(labels, scores)
        pr = self.metrics_calc.compute_pr_curve(labels, scores)

        self.results = self._assemble_results(cm, metrics, roc, pr)
        self.results["scores"] = scores          # kept for per-category disaggregation
        self.results["labels"] = labels
        logger.info(
            "%s: P=%.3f R=%.3f F1=%.3f FPR=%.4f",
            self.model.get_model_name(),
            metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["fpr"],
        )
        return self.results

    def get_predictions(self, scores: np.ndarray) -> np.ndarray:
        """Binary decisions at the tuned threshold (1 = anomaly)."""
        return (np.asarray(scores, dtype=np.float64) > self.threshold).astype(int)

    def get_model_name(self) -> str:
        return self.model.get_model_name()

    def get_manifest_hash(self) -> str:
        return self.manifest_hash

    def get_results(self) -> dict:
        return self.results

    # -------------------------------------------------------------- helpers
    def _score_and_predict(self, test_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self.model.score(test_data)
        return scores, self.get_predictions(scores)

    def _assemble_results(self, cm: dict, metrics: dict, roc: dict, pr: dict) -> dict:
        roc_auc = self.metrics_calc.compute_auc(
            np.array(roc["fpr"]), np.array(roc["tpr"])
        )
        return {
            "model_name": self.model.get_model_name(),
            "score_type": self.model.get_score_type(),
            "hyperparameters": self.model.get_hyperparameters(),
            "threshold": self.threshold,
            "manifest_hash": self.manifest_hash,
            "n_test_rows": cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"],
            "confusion_matrix": cm,
            "metrics": metrics,
            "roc_curve": roc,
            "pr_curve": pr,
            "roc_auc": roc_auc,
        }
