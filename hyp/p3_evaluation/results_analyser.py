"""
P3 - Evaluation & Analysis (UC6 + UC7, consolidated)
ResultsAnalyser: cross-model comparison and per-category disaggregation.

FR-EV-03 is ENFORCED here: verify_invariant() checks that every
evaluation carries the same preprocessing manifest hash and the same
test-split size. compare_models() and the per-category methods refuse
to run if the invariant fails (contract test 3.6: "the comparator must
refuse to produce output rather than return an unfair comparison").
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from hyp.p3_evaluation.metrics_calculator import MetricsCalculator

logger = logging.getLogger(__name__)


class InvariantViolation(RuntimeError):
    """Raised when the FR-EV-03 identical-conditions invariant fails."""


class ResultsAnalyser:
    """Consolidated UC6 (compare models) + UC7 (per-category analysis)."""

    def __init__(self, evaluations: list[dict]):
        if not evaluations:
            raise ValueError("ResultsAnalyser requires at least one evaluation")
        self.evaluations = list(evaluations)

    # ------------------------------------------------------- FR-EV-03 gate
    def verify_invariant(self) -> bool:
        return self._check_manifest_consistency() and self._check_test_size_consistency()

    def _require_invariant(self) -> None:
        if not self._check_manifest_consistency():
            raise InvariantViolation(
                "FR-EV-03 violated: evaluations carry different preprocessing "
                "manifest hashes - the models were not scored under identical "
                "conditions. Refusing to produce comparative output."
            )
        if not self._check_test_size_consistency():
            raise InvariantViolation(
                "FR-EV-03 violated: evaluations have different test-split "
                "sizes. Refusing to produce comparative output."
            )

    # ---------------------------------------------------------------- UC6
    def compare_models(self) -> pd.DataFrame:
        """Side-by-side metrics table: one row per model, all four metrics
        (+ accuracy, ROC-AUC, threshold). Test 3.7: complete, no NaNs."""
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

    # ---------------------------------------------------------------- UC7
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
        and are reported once under the 'BENIGN' row (as FPR).
        Coverage guarantee (test 3.8): every category present in
        ``categories`` appears in the output."""
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
        """UC7 across every evaluation held by the analyser."""
        self._require_invariant()
        frames = []
        for ev in self.evaluations:
            frames.append(self.analyse_by_category(
                ev["scores"], ev["labels"], categories,
                ev["threshold"], ev["model_name"],
            ))
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------- overlays
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
        confusion matrix (test 3.9)."""
        cm = MetricsCalculator.compute_confusion_matrix(labels[mask], predictions[mask])
        n = int(mask.sum())
        recall = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) > 0 else None
        fpr = cm["fp"] / (cm["fp"] + cm["tn"]) if (cm["fp"] + cm["tn"]) > 0 else None
        return {"n_flows": n, **cm,
                "recall": None if recall is None else round(recall, 4),
                "fpr": None if fpr is None else round(fpr, 4)}
