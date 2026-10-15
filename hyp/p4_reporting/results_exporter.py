import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultsExporter:
    """Collates evaluation outputs into report-ready artefacts."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_comparison_table(self, comparison_df: pd.DataFrame) -> None:
        comparison_df.to_csv(self.output_dir / "comparison_table.csv", index=False)

    def export_confusion_matrices(self, evaluations: list[dict]) -> None:
        for ev in evaluations:
            cm = ev["confusion_matrix"]
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]),
                        annot=True, fmt="d", cbar=False, cmap="Blues",
                        xticklabels=["Pred benign", "Pred attack"],
                        yticklabels=["True benign", "True attack"], ax=ax)
            ax.set_title(f"{ev['model_name']} - confusion matrix @ thr={ev['threshold']:.4f}",
                         fontsize=10)
            self._save_figure(fig, f"confusion_{ev['model_name']}")

    def export_dashboard_json(self, evaluations: list[dict], summary: dict) -> None:
        """Headline results in one JSON file, served to the Angular dashboard by api.py."""
        models = []
        for ev in evaluations:
            m = ev["metrics"]
            models.append({
                "name": ev["model_name"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fpr": m["fpr"],
                "accuracy": m["accuracy"],
                "threshold": ev["threshold"],
                "confusion_matrix": ev["confusion_matrix"],
            })
        dashboard = {
            "summary": summary,
            "models": models,
        }
        path = self.output_dir / "dashboard.json"
        path.write_text(json.dumps(dashboard, indent=2, allow_nan=False))

    # helpers
    def _save_figure(self, fig, filename: str) -> None:
        fig.savefig(self.output_dir / f"{filename}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

