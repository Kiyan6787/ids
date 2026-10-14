import logging

import numpy as np

logger = logging.getLogger(__name__)


class ThresholdTuner:
    """Selects an anomaly-score threshold satisfying an FPR target."""

    @staticmethod
    def tune(scores: np.ndarray, labels: np.ndarray,
             target_fpr: float = 0.01, n_points: int = 200) -> float:
        """scores: validation anomaly scores (higher = worse);
        labels: 0 = benign, 1 = attack (all zeros for benign-only val)."""
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        benign_scores = scores[np.asarray(labels).reshape(-1) == 0]
        if len(benign_scores) == 0:
            raise ValueError("Validation slice contains no benign flows; "
                             "FPR criterion is undefined.")

        candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_points)))
        fprs = np.array([np.mean(benign_scores > t) for t in candidates])
        threshold = float(candidates[fprs <= target_fpr][0])  # candidates ascend
        logger.info("ThresholdTuner: FPR <= %.4f -> threshold=%.6f", target_fpr, threshold)
        return threshold
