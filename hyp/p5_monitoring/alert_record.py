from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

VALID_VERDICTS = {"true_positive", "false_positive", "inconclusive"}


class AlertRecord:

    def __init__(self, flow: dict[str, Any], score: float, threshold: float):
        self.alert_id = str(uuid.uuid4())
        self.flow = dict(flow)
        self.score = float(score)
        self.threshold = float(threshold)
        self.timestamp = datetime.now(timezone.utc)
        self.verdict: str | None = None
        self.reviewer: str | None = None
        self.notes: str | None = None

    # API
    def set_verdict(self, verdict: str, reviewer: str, notes: str = "") -> None:
        if verdict not in VALID_VERDICTS:
            raise ValueError(
                f"Invalid verdict '{verdict}'. Expected one of {sorted(VALID_VERDICTS)}"
            )
        self.verdict = verdict
        self.reviewer = reviewer
        self.notes = notes

    def is_reviewed(self) -> bool:
        return self.verdict is not None

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "flow": self.flow,
            "score": self.score,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "verdict": self.verdict,
            "reviewer": self.reviewer,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict) -> "AlertRecord":
        record = AlertRecord(data["flow"], data["score"], data["threshold"])
        record.alert_id = data["alert_id"]
        record.timestamp = datetime.fromisoformat(data["timestamp"])
        record.verdict = data.get("verdict")
        record.reviewer = data.get("reviewer")
        record.notes = data.get("notes")
        return record
