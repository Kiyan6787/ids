import logging

import pandas as pd

logger = logging.getLogger(__name__)


class ResultsAnalyser:

    def __init__(self, evaluations: list[dict]):
        self.evaluations = list(evaluations)

    def verify_invariant(self) -> bool:
        hashes = {ev["manifest_hash"] for ev in self.evaluations}
        sizes = {ev["n_test_rows"] for ev in self.evaluations}
        if len(hashes) == 1 and len(sizes) == 1:
            return True
        logger.error("Invariant violated: manifest hashes %s, test sizes %s", hashes, sizes)
        return False

    def compare_models(self) -> pd.DataFrame:
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
                "threshold": ev["threshold"],
            })
        return pd.DataFrame(rows)

    # helpers
    def _require_invariant(self) -> None:
        if not self.verify_invariant():
            raise ValueError(
                "Evaluations were not produced under identical conditions "
                "(preprocessing hash or test-split size differ). "
                "Cant produce comparative output."
            )
