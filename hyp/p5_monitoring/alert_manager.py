from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from hyp.p2_models.anomaly_detector import AnomalyDetector
from hyp.p5_monitoring.alert_record import AlertRecord

logger = logging.getLogger(__name__)


class AlertManager:

    def __init__(
        self,
        detector: AnomalyDetector,
        threshold: float,
        fpr_window_size: int = 100,
        re_tune_trigger: float = 0.05,
    ):
        self.detector = detector
        self.threshold = float(threshold)
        self.running_fpr: float = 0.0
        self.alert_history: list[AlertRecord] = []
        self.fpr_window_size = fpr_window_size
        self.re_tune_trigger = re_tune_trigger
        self._verdict_window: deque[int] = deque(maxlen=fpr_window_size)

    def score_flow(self, flow: dict[str, Any]) -> float:
        features = np.asarray(flow["features"], dtype=np.float64).reshape(1, -1)
        return float(self.detector.score(features)[0])

    def create_alert(self, flow: dict[str, Any], score: float) -> AlertRecord | None:
        if not self._evaluate_score(score):
            return None
        record = AlertRecord(flow, score, self.threshold)
        self.alert_history.append(record)
        logger.info("Alert %s raised: score=%.4f > thr=%.4f",
                    record.alert_id, score, self.threshold)
        return record

    def update_fpr(self, verdict: str) -> float:
        if verdict not in {"true_positive", "false_positive"}:
            raise ValueError(
                "update_fpr expects 'true_positive' or 'false_positive' "
                "(inconclusive verdicts do not update the estimate)"
            )
        self._update_running_window(verdict)
        return self.running_fpr

    def should_re_tune(self) -> bool:
        return (len(self._verdict_window) == self.fpr_window_size
                and self.running_fpr > self.re_tune_trigger)

    def get_running_fpr(self) -> float:
        return self.running_fpr

    def get_alert_history(self) -> list[AlertRecord]:
        return list(self.alert_history)

    # helpers
    def _evaluate_score(self, score: float) -> bool:
        return score > self.threshold

    def _update_running_window(self, verdict: str) -> None:
        self._verdict_window.append(1 if verdict == "false_positive" else 0)
        self.running_fpr = sum(self._verdict_window) / len(self._verdict_window)
