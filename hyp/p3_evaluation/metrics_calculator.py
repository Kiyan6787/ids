"""Stateless detection-metric computations: confusion matrix, precision/
recall/F1/FPR, and ROC/PR curves - all implemented by hand (no sklearn),
cross-checked against scikit-learn in the test suite.

Every ratio with a zero denominator is defined as 0.0 rather than
raising, so degenerate inputs (no positives, no negatives, no
predictions) produce well-defined output.
"""

from __future__ import annotations

import numpy as np


class MetricsCalculator:
    """Static, side-effect-free detection-metric computations."""

    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have identical shape")
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    @staticmethod
    def compute_metrics(cm: dict) -> dict:
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = ((tp + tn) / (tp + fp + tn + fn)
                    if (tp + fp + tn + fn) > 0 else 0.0)
        return {"precision": precision, "recall": recall, "f1": f1,
                "fpr": fpr, "accuracy": accuracy}

    @staticmethod
    def compute_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> dict:
        """ROC curve computed from first principles.

        Sort by descending score; sweep every distinct score as a
        threshold; accumulate TP/FP counts. Returns fpr, tpr and the
        corresponding thresholds (descending)."""
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        order = np.argsort(-scores, kind="mergesort")   # stable
        y_sorted = y_true[order]
        s_sorted = scores[order]

        p = int(y_true.sum())
        n = int(len(y_true) - p)

        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        # Keep only the last index of each distinct score (threshold points)
        distinct = np.where(np.diff(s_sorted))[0]
        idx = np.r_[distinct, len(s_sorted) - 1]

        tpr = tps[idx] / p if p > 0 else np.zeros(len(idx))
        fpr = fps[idx] / n if n > 0 else np.zeros(len(idx))
        # Prepend the (0, 0) origin
        return {
            "fpr": np.r_[0.0, fpr].tolist(),
            "tpr": np.r_[0.0, tpr].tolist(),
            "thresholds": np.r_[s_sorted[idx][0] + 1.0, s_sorted[idx]].tolist(),
        }

    @staticmethod
    def compute_pr_curve(y_true: np.ndarray, scores: np.ndarray) -> dict:
        y_true = np.asarray(y_true).reshape(-1).astype(int)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        order = np.argsort(-scores, kind="mergesort")
        y_sorted = y_true[order]
        s_sorted = scores[order]

        p = int(y_true.sum())
        tps = np.cumsum(y_sorted)
        preds = np.arange(1, len(y_sorted) + 1)

        distinct = np.where(np.diff(s_sorted))[0]
        idx = np.r_[distinct, len(s_sorted) - 1]

        precision = tps[idx] / preds[idx]
        recall = tps[idx] / p if p > 0 else np.zeros(len(idx))
        # Standard endpoint: recall 0 at precision 1
        return {
            "precision": np.r_[1.0, precision].tolist(),
            "recall": np.r_[0.0, recall].tolist(),
            "thresholds": s_sorted[idx].tolist(),
        }

    @staticmethod
    def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
        """Area under a curve via the trapezoidal rule."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("compute_auc needs two equal-length arrays, n >= 2")
        order = np.argsort(x, kind="mergesort")
        return float(np.trapezoid(y[order], x[order]))
