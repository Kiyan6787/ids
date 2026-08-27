"""Cross-model comparison and per-category disaggregation.

verify_invariant() checks that every evaluation carries the same
preprocessing manifest hash and the same test-split size. compare_models()
and the per-category methods refuse to run if that invariant fails,
rather than return an unfair comparison.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hyp.p3_evaluation.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class InvariantViolation(RuntimeError):
    """Raised when evaluations were not produced under identical conditions."""


class ResultsAnalyser:
    """Cross-model comparison and per-category analysis."""

    def __init__(self, evaluations: list[dict]):
        if not evaluations:
            raise ValueError("ResultsAnalyser requires at least one evaluation")
        self.evaluations = list(evaluations)

    def verify_invariant(self) -> bool:
        return self._check_manifest_consistency() and self._check_test_size_consistency()

    def _require_invariant(self) -> None:
        if not self._check_manifest_consistency():
            raise InvariantViolation(
                "Evaluations carry different preprocessing manifest hashes - "
                "the models were not scored under identical conditions. "
                "Refusing to produce comparative output."
            )
        if not self._check_test_size_consistency():
            raise InvariantViolation(
                "Evaluations have different test-split sizes. "
                "Refusing to produce comparative output."
            )

    def compare_models(self) -> pd.DataFrame:
        """Side-by-side metrics table: one row per model, all metrics
        plus accuracy, ROC-AUC, and threshold."""
        self._require_invariant()
        rows = []
        for ev in self.evaluations:
            m = ev["metrics"]
            rows.append({
                "model": ev["model_name"],
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "fpr": round(m["fpr"], 4),
                "accuracy": round(m["accuracy"], 4),
                "roc_auc": round(ev["roc_auc"], 4),
                "threshold": ev["threshold"],
            })
        return pd.DataFrame(rows)

    def analyse_by_category(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        categories: np.ndarray,
        threshold: float,
        model_name: str,
    ) -> pd.DataFrame:
        """Disaggregate one model's detection performance by attack category.

        For each attack category: recall on that category's flows, plus
        the TP/FN counts. FP/TN are properties of the benign population
        and are reported once under the 'BENIGN' row (as FPR). Every
        category present in ``categories`` appears in the output."""
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        categories = np.asarray(categories).reshape(-1)
        predictions = (scores > threshold).astype(int)

        rows = []
        for category in sorted(pd.unique(categories)):
            mask = categories == category
            d = self._disaggregate_metrics(mask, labels, predictions)
            rows.append({"model": model_name, "category": category, **d})
        return pd.DataFrame(rows)

    def analyse_all_models_by_category(self, categories: np.ndarray) -> pd.DataFrame:
        """Per-category analysis across every evaluation held by the analyser."""
        self._require_invariant()
        frames = []
        for ev in self.evaluations:
            frames.append(self.analyse_by_category(
                ev["scores"], ev["labels"], categories,
                ev["threshold"], ev["model_name"],
            ))
        return pd.concat(frames, ignore_index=True)

    def get_roc_overlay_data(self) -> dict:
        self._require_invariant()
        return {ev["model_name"]: ev["roc_curve"] for ev in self.evaluations}

    def get_pr_overlay_data(self) -> dict:
        self._require_invariant()
        return {ev["model_name"]: ev["pr_curve"] for ev in self.evaluations}

    # -------------------------------------------------------------- helpers
    def _check_manifest_consistency(self) -> bool:
        hashes = {ev["manifest_hash"] for ev in self.evaluations}
        ok = len(hashes) == 1
        if not ok:
            logger.error("Manifest hash mismatch across evaluations: %s", hashes)
        return ok

    def _check_test_size_consistency(self) -> bool:
        sizes = {ev["n_test_rows"] for ev in self.evaluations}
        ok = len(sizes) == 1
        if not ok:
            logger.error("Test-split size mismatch across evaluations: %s", sizes)
        return ok

    @staticmethod
    def _disaggregate_metrics(
        mask: np.ndarray, labels: np.ndarray, predictions: np.ndarray
    ) -> dict:
        """Confusion counts restricted to the masked subset. Summing these
        counts across all category rows reconciles exactly to the overall
        confusion matrix."""
        cm = MetricsCalculator.compute_confusion_matrix(labels[mask], predictions[mask])
        n = int(mask.sum())
        recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else None
        fpr = cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) > 0 else None
        return {"n_flows": n, **cm,
                "recall": None if recall is None else round(recall, 4),
                "fpr": None if fpr is None else round(fpr, 4)}
