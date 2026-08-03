"""
P1 - Data Management (UC1)
Preprocessor: FR-DM-02 (NaN/Inf repair), FR-DM-03 (normalisation),
FR-DM-04 (categorical encoding).

The preprocessing manifest and its SHA-256 hash are the anchor of the
FR-EV-03 invariant: every model artefact and every evaluation carries
this hash, and ResultsAnalyser.verify_invariant() refuses to compare
results whose hashes differ.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor:
    """Cleans, encodes and normalises CICIDS 2017 flow records."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.manifest: dict[str, Any] = {"steps": []}
        self.drop_log: list[dict] = []
        self.is_fitted: bool = False
        self.feature_columns: list[str] = []
        self.label_column: str = ""

    # ------------------------------------------------------------------ API
    def repair_nan_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """FR-DM-02: repair NaN/Inf artefacts introduced by CICFlowMeter.

        Strategy (Engelen et al. [33]): the affected columns are
        'Flow Bytes/s' and 'Flow Packets/s'; +/-Inf and NaN rows are
        dropped, and each drop is recorded in ``drop_log`` so that
        row-count arithmetic remains auditable (test 1.4)."""
        df = df.copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        n_before = len(df)
        nan_mask = df.isna().any(axis=1)
        n_dropped = int(nan_mask.sum())
        if n_dropped:
            per_column = df.loc[nan_mask].isna().sum()
            self.drop_log.append({
                "step": "repair_nan_inf",
                "rows_before": n_before,
                "rows_dropped": n_dropped,
                "rows_after": n_before - n_dropped,
                "columns_affected": {
                    c: int(v) for c, v in per_column.items() if v > 0
                },
            })
            df = df.loc[~nan_mask].reset_index(drop=True)
        self.manifest["steps"].append({"repair_nan_inf": {"rows_dropped": n_dropped}})
        logger.info("repair_nan_inf: dropped %d rows", n_dropped)
        return df

    def encode_categorical(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """FR-DM-04: numerically encode the label column.

        CICIDS 2017 feature columns are already numeric; the only
        categorical field is the Label. The original string label is
        preserved in 'Label_str' for UC7 per-category analysis."""
        df = df.copy()
        self.label_column = label_col
        df[label_col] = df[label_col].astype(str).str.strip()
        df["Label_str"] = df[label_col]
        df[label_col] = self.label_encoder.fit_transform(df[label_col])
        classes = list(self.label_encoder.classes_)
        self.manifest["steps"].append({"encode_categorical": {"classes": classes}})
        logger.info("encode_categorical: %d classes", len(classes))
        return df

    def normalise(self, df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
        """FR-DM-03: standardise numeric features to mean 0 / std 1.

        The scaler is FIT here on the full cleaned frame and the fitted
        parameters recorded in the manifest, so that any later run with
        the same inputs yields byte-identical artefacts (test 1.6)."""
        df = df.copy()
        numeric_cols = self._identify_numeric_columns(df)
        self.feature_columns = [c for c in numeric_cols if c not in exclude]
        # Guard: zero-variance columns produce NaN under StandardScaler
        # division; drop them and log it.
        variances = df[self.feature_columns].var()
        zero_var = [c for c in self.feature_columns if variances[c] == 0.0]
        if zero_var:
            self.feature_columns = [c for c in self.feature_columns if c not in zero_var]
            self.manifest["steps"].append({"dropped_zero_variance": zero_var})
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        self.is_fitted = True
        self.manifest["steps"].append({
            "normalise": {
                "n_features": len(self.feature_columns),
                "feature_columns": self.feature_columns,
                "scaler_mean": [round(float(m), 8) for m in self.scaler.mean_],
                "scaler_scale": [round(float(s), 8) for s in self.scaler.scale_],
            }
        })
        logger.info("normalise: standardised %d features", len(self.feature_columns))
        return df

    def get_manifest(self) -> dict:
        """Return the manifest including its SHA-256 hash (FR-EV-03 anchor).

        Required keys (test 1.7): steps, feature_columns, label_column,
        drop_log, preprocessing_hash."""
        manifest = {
            "steps": self.manifest["steps"],
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
            "drop_log": self.drop_log,
        }
        manifest["preprocessing_hash"] = self._compute_manifest_hash(manifest)
        return manifest

    def get_feature_columns(self) -> list[str]:
        return list(self.feature_columns)

    def get_drop_log(self) -> list[dict]:
        return list(self.drop_log)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _compute_manifest_hash(manifest: dict) -> str:
        payload = json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _identify_numeric_columns(df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include=[np.number]).columns.tolist()
