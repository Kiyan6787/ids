"""Sweep-based threshold selection against a configurable operating-point
criterion (default FPR <= 1 %).

The sweep runs over quantiles of the validation score distribution. FPR
is monotonically non-increasing as the threshold rises, so the tuner
selects the LOWEST threshold whose FPR is at or below the target - this
maximises detection sensitivity while still meeting the FPR constraint.

The validation split is benign-only, so precision and recall are
reported as None when no positive labels are present; FPR is always
computable.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ThresholdTuner:
    """Selects an anomaly-score threshold satisfying an FPR criterion."""

    def __init__(self):
        self.criterion: str = ""
        self.threshold: float | None = None
        self.sweep_results: list[dict] = []
        self.target_fpr: float | None = None

    # ------------------------------------------------------------------ API
    def tune(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        criterion: str = "fpr<=0.01",
        n_points: int = 200,
    ) -> float:
        """Sweep candidate thresholds and select per the criterion.

        Parameters
        ----------
        scores : anomaly scores on the validation slice (higher = worse)
        labels : 0 = benign, 1 = attack (all zeros for benign-only val)
        criterion : e.g. "fpr<=0.01"
        """
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels).reshape(-1).astype(int)
        if scores.shape != labels.shape:
            raise ValueError("scores and labels must be the same length")
        if len(scores) == 0:
            raise ValueError("Cannot tune on an empty validation slice")

        self.criterion = criterion
        self.target_fpr = self._parse_criterion(criterion)

        benign_mask = labels == 0
        attack_mask = ~benign_mask
        n_benign = int(benign_mask.sum())
        n_attack = int(attack_mask.sum())
        if n_benign == 0:
            raise ValueError("Validation slice contains no benign flows; "
                             "FPR criterion is undefined.")

        # Candidate thresholds: quantiles of the observed score range.
        quantiles = np.linspace(0.0, 1.0, n_points)
        candidates = np.unique(np.quantile(scores, quantiles))

        self.sweep_results = []
        for thr in candidates:
            predicted_pos = scores > thr
            fp = int((predicted_pos & benign_mask).sum())
            fpr = fp / n_benign
            entry = {"threshold": float(thr), "fpr": float(fpr)}
            if n_attack > 0:
                tp = int((predicted_pos & attack_mask).sum())
                fn = n_attack - tp
                entry["recall"] = tp / n_attack
                entry["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                entry["fn"] = fn
            else:
                entry["recall"] = None
                entry["precision"] = None
            self.sweep_results.append(entry)

        self.threshold = self._select_threshold(self.target_fpr)
        logger.info("ThresholdTuner: criterion=%s -> threshold=%.6f",
                    criterion, self.threshold)
        return self.threshold

    def get_sweep_curve(self) -> list[dict]:
        return list(self.sweep_results)

    def get_threshold(self) -> float:
        if self.threshold is None:
            raise RuntimeError("tune() has not been called")
        return self.threshold

    def get_criterion(self) -> str:
        return self.criterion

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _parse_criterion(criterion: str) -> float:
        """Parse 'fpr<=0.01' -> 0.01."""
        criterion = criterion.replace(" ", "").lower()
        if not criterion.startswith("fpr<="):
            raise ValueError(
                f"Unsupported criterion '{criterion}'. Expected form 'fpr<=X'."
            )
        return float(criterion[len("fpr<="):])

    def _select_threshold(self, target_fpr: float) -> float:
        """Lowest candidate threshold whose FPR <= target (max sensitivity)."""
        feasible = [e for e in self.sweep_results if e["fpr"] <= target_fpr]
        if not feasible:
            # Fall back loudly to the strictest threshold available.
            worst = max(self.sweep_results, key=lambda e: e["threshold"])
            logger.warning(
                "No candidate threshold satisfies FPR <= %.4f; "
                "falling back to max threshold %.6f (FPR=%.4f)",
                target_fpr, worst["threshold"], worst["fpr"],
            )
            return worst["threshold"]
        return min(feasible, key=lambda e: e["threshold"])["threshold"]
