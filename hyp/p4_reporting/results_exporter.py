"""Exports comparison tables, confusion matrices, ROC/PR overlays, and
per-category bar charts to disk, alongside a manifest that maps each
output file back to the pipeline stage that produced it.

Matplotlib runs headless ('Agg' backend) so this works on a CI box
with no display.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless - must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsExporter:
    """Collates evaluation outputs into report-ready artefacts."""

    def __init__(self, output_dir: str, figure_format: str = "png", figure_dpi: int = 200):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_format = figure_format
        self.figure_dpi = figure_dpi
        self.manifest: list[dict] = []

    def export_tables(self, comparison_df: pd.DataFrame, category_df: pd.DataFrame) -> None:
        comp_path = self.output_dir / "comparison_table.csv"
        comparison_df.to_csv(comp_path, index=False)
        self._record(comp_path, "UC6", "Side-by-side model comparison table")

        cat_path = self.output_dir / "per_category_table.csv"
        category_df.to_csv(cat_path, index=False)
        self._record(cat_path, "UC7", "Per-attack-category disaggregation")

    def export_confusion_matrices(self, evaluations: list[dict]) -> None:
        for ev in evaluations:
            fig = self._create_heatmap(
                ev["confusion_matrix"],
                f"{ev['model_name']} - confusion matrix @ thr={ev['threshold']:.4f}",
            )
            self._save_figure(
                fig, f"confusion_{ev['model_name']}",
                "UC5", f"Confusion matrix for {ev['model_name']}",
            )

    def export_roc_curves(self, evaluations: list[dict]) -> None:
        fig = self._create_curve_overlay(evaluations, "roc")
        self._save_figure(fig, "roc_overlay", "UC6", "ROC overlay, all models")

    def export_pr_curves(self, evaluations: list[dict]) -> None:
        fig = self._create_curve_overlay(evaluations, "pr")
        self._save_figure(fig, "pr_overlay", "UC6", "PR overlay, all models")

    def export_category_charts(self, category_df: pd.DataFrame) -> None:
        fig = self._create_category_barchart(category_df, "recall")
        self._save_figure(fig, "per_category_recall", "UC7",
                          "Per-category recall by model")

    # --------------------------------------------------------------- manifest
    def write_manifest(self) -> None:
        path = self.output_dir / "export_manifest.json"
        path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "traceability": "UC8 -> UC6/UC7 -> UC5 -> UC4 -> UC2 -> UC1",
            "files": self.manifest,
        }, indent=2))
        logger.info("Export manifest written: %d artefacts", len(self.manifest))

    def get_exported_files(self) -> list[str]:
        return [entry["file"] for entry in self.manifest]

    # -------------------------------------------------------------- helpers
    def _record(self, path: Path, source_uc: str, description: str) -> None:
        self.manifest.append({
            "file": path.name, "source_uc": source_uc, "description": description,
        })

    def _save_figure(self, fig, filename: str, source_uc: str, description: str) -> None:
        path = self.output_dir / f"{filename}.{self.figure_format}"
        fig.savefig(path, dpi=self.figure_dpi, bbox_inches="tight")
        plt.close(fig)
        self._record(path, source_uc, description)

    @staticmethod
    def _create_heatmap(cm: dict, title: str):
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cbar=False, cmap="Blues",
                    xticklabels=["Pred benign", "Pred attack"],
                    yticklabels=["True benign", "True attack"], ax=ax)
        ax.set_title(title, fontsize=10)
        return fig

    @staticmethod
    def _create_curve_overlay(evaluations: list[dict], curve_type: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        for ev in evaluations:
            if curve_type == "roc":
                curve = ev["roc_curve"]
                ax.plot(curve["fpr"], curve["tpr"],
                        label=f"{ev['model_name']} (AUC={ev['roc_auc']:.3f})")
            else:
                curve = ev["pr_curve"]
                ax.plot(curve["recall"], curve["precision"], label=ev["model_name"])
        if curve_type == "roc":
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title("ROC curves - all models")
        else:
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall curves - all models")
        ax.legend()
        return fig

    @staticmethod
    def _create_category_barchart(df: pd.DataFrame, metric: str):
        plot_df = df[df["category"] != "BENIGN"].copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=plot_df, x="category", y=metric, hue="model", ax=ax)
        ax.set_title(f"Per-attack-category {metric} by model")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30)
        return fig
