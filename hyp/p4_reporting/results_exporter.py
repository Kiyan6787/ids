import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hyp.p3_evaluation.metrics_calculator import MetricsCalculator as MC


class ResultsExporter:
    """Collates evaluation outputs into report-ready artefacts."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_tables(self, comparison_df: pd.DataFrame, category_df: pd.DataFrame) -> None:
        comparison_df.to_csv(self.output_dir / "comparison_table.csv", index=False)
        category_df.to_csv(self.output_dir / "per_category_table.csv", index=False)

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

    def export_roc_curves(self, evaluations: list[dict]) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        for ev in evaluations:
            ax.plot(ev["roc_curve"]["fpr"], ev["roc_curve"]["tpr"],
                    label=f"{ev['model_name']} (AUC={ev['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title="ROC curves - all models")
        ax.legend()
        self._save_figure(fig, "roc_overlay")

    def export_pr_curves(self, evaluations: list[dict]) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        for ev in evaluations:
            ax.plot(ev["pr_curve"]["recall"], ev["pr_curve"]["precision"],
                    label=ev["model_name"])
        ax.set(xlabel="Recall", ylabel="Precision",
               title="Precision-Recall curves - all models")
        ax.legend()
        self._save_figure(fig, "pr_overlay")

    def export_category_charts(self, category_df: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=category_df[category_df["category"] != "BENIGN"],
                    x="category", y="recall", hue="model", ax=ax)
        ax.set_title("Per-attack-category recall by model")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        self._save_figure(fig, "per_category_recall")

    def export_dashboard_json(self, evaluations: list[dict], category_df: pd.DataFrame,
                              summary: dict) -> None:
        """Headline results in one JSON file, served to the Angular dashboard by api.py."""
        models = []
        for ev in evaluations:
            m = ev["metrics"]
            roc, pr = ev["roc_curve"], ev["pr_curve"]
            models.append({
                "name": ev["model_name"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fpr": m["fpr"],
                "accuracy": m["accuracy"],
                "roc_auc": ev["roc_auc"],
                "pr_auc": MC.compute_auc(pr["recall"], pr["precision"]),
                "threshold": ev["threshold"],
                "confusion_matrix": ev["confusion_matrix"],
                "roc_curve": {"fpr": _downsample(roc["fpr"]), "tpr": _downsample(roc["tpr"])},
                "pr_curve": {"recall": _downsample(pr["recall"]),
                             "precision": _downsample(pr["precision"])},
            })
        dashboard = {
            "summary": summary,
            "models": models,
            "categories": json.loads(category_df.to_json(orient="records")),
        }
        path = self.output_dir / "dashboard.json"
        path.write_text(json.dumps(dashboard, indent=2, allow_nan=False))

    # helpers
    def _save_figure(self, fig, filename: str) -> None:
        fig.savefig(self.output_dir / f"{filename}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _downsample(values: list, n_points: int = 200) -> list:
    if len(values) <= n_points:
        return list(values)
    indices = np.linspace(0, len(values) - 1, n_points).astype(int)
    return [values[i] for i in indices]
