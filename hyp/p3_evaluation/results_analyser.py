import logging

import numpy as np
import pandas as pd

from hyp.p3_evaluation.metrics_calculator import MetricsCalculator

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
                "roc_auc": round(ev["roc_auc"], 4),
                "threshold": ev["threshold"],
            })
        return pd.DataFrame(rows)

    def analyse_by_category(self, scores: np.ndarray, labels: np.ndarray,
                            categories: np.ndarray, threshold: float,
                            model_name: str) -> pd.DataFrame:
        labels = np.asarray(labels).reshape(-1).astype(int)
        categories = np.asarray(categories).reshape(-1)
        predictions = (np.asarray(scores).reshape(-1) > threshold).astype(int)

        rows = []
        for category in sorted(pd.unique(categories)):
            mask = categories == category
            cm = MetricsCalculator.compute_confusion_matrix(labels[mask], predictions[mask])
            recall = cm["tp"] / (cm["tp"] + cm["fn"]) if cm["tp"] + cm["fn"] > 0 else None
            fpr = cm["fp"] / (cm["fp"] + cm["tn"]) if cm["fp"] + cm["tn"] > 0 else None
            rows.append({
                "model": model_name,
                "category": category,
                "n_flows": int(mask.sum()),
                "tp": cm["tp"],
                "fp": cm["fp"],
                "tn": cm["tn"],
                "fn": cm["fn"],
                "recall": None if recall is None else round(recall, 4),
                "fpr": None if fpr is None else round(fpr, 4),
            })
        return pd.DataFrame(rows)

    def analyse_all_models_by_category(self, categories: np.ndarray) -> pd.DataFrame:
        self._require_invariant()
        tables = []
        for ev in self.evaluations:
            table = self.analyse_by_category(ev["scores"], ev["labels"], categories,
                                             ev["threshold"], ev["model_name"])
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    # helpers
    def _require_invariant(self) -> None:
        if not self.verify_invariant():
            raise ValueError(
                "Evaluations were not produced under identical conditions "
                "(preprocessing hash or test-split size differ). "
                "Cant produce comparative output."
            )
