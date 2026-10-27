"""Cleans and normalises CICIDS 2017 flow records.
"""

import hashlib
import json
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.steps: list[dict] = []
        self.feature_columns: list[str] = []
        self.label_column: str = ""
        self.rows_dropped = 0

    def repair_nan_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        n_dropped = n_before - len(df)
        self.rows_dropped += n_dropped
        self.steps.append({"repair_nan_inf": {"rows_dropped": n_dropped}})
        logger.info("repair_nan_inf: dropped %d rows", n_dropped)
        return df

    def clean_labels(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        df = df.copy()
        self.label_column = label_col
        df["Label_str"] = df[label_col].astype(str).str.strip()
        classes = sorted(df["Label_str"].unique())
        self.steps.append({"clean_labels": {"classes": classes}})
        logger.info("clean_labels: %d classes", len(classes))
        return df

    def normalise(self, df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.feature_columns = [c for c in numeric_cols if c not in exclude]
        variances = df[self.feature_columns].var()
        zero_var = [c for c in self.feature_columns if variances[c] == 0.0]
        if zero_var:
            self.feature_columns = [c for c in self.feature_columns if c not in zero_var]
            self.steps.append({"dropped_zero_variance": zero_var})
        df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        self.steps.append({
            "normalise": {
                "n_features": len(self.feature_columns),
                "scaler_mean": [round(float(m), 8) for m in self.scaler.mean_],
                "scaler_scale": [round(float(s), 8) for s in self.scaler.scale_],
            }
        })
        logger.info("normalise: standardised %d features", len(self.feature_columns))
        return df

    def get_manifest_hash(self) -> str:
        manifest = {
            "steps": self.steps,
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
        }
        payload = json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
