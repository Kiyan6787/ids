"""
P1 - Data Management (UC1)
DataSplitter: FR-DM-05 (benign-only train split), FR-DM-06 (deterministic
persistence).

Split protocol (activity diagram):
  * TRAIN      - benign only            (unsupervised training)
  * VALIDATION - benign only            (threshold tuning at FPR <= 1 %)
  * TEST       - remaining benign + ALL attacks (held-out evaluation)

The same persisted splits feed every model, which is what makes the
three-way comparison controlled (FR-EV-03).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class DataSplitter:
    """Deterministic benign/attack-aware splitting of the cleaned dataset."""

    def __init__(self, random_seed: int):
        self.random_seed = random_seed
        self.train_ratio = config.TRAIN_RATIO
        self.val_ratio = config.VAL_RATIO
        self.train_set: pd.DataFrame | None = None
        self.validation_set: pd.DataFrame | None = None
        self.test_set: pd.DataFrame | None = None

    # ------------------------------------------------------------------ API
    def split(
        self, df: pd.DataFrame, label_col: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """FR-DM-05: benign-only train/val, mixed test."""
        benign, attack = self._separate_benign_attack(df)
        benign = self._shuffle_deterministic(benign)

        n = len(benign)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        self.train_set = benign.iloc[:n_train].reset_index(drop=True)
        self.validation_set = benign.iloc[n_train:n_train + n_val].reset_index(drop=True)
        benign_test = benign.iloc[n_train + n_val:]

        test = pd.concat([benign_test, attack], ignore_index=True)
        self.test_set = self._shuffle_deterministic(test).reset_index(drop=True)

        logger.info(
            "split: train=%d (benign only), val=%d (benign only), "
            "test=%d (%d benign + %d attack)",
            len(self.train_set), len(self.validation_set), len(self.test_set),
            len(benign_test), len(attack),
        )
        return self.train_set, self.validation_set, self.test_set

    def persist_to_disk(self, output_dir: str) -> None:
        """FR-DM-06: persist splits + split metadata for deterministic reuse."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.train_set.to_parquet(out / "train.parquet", index=False)
        self.validation_set.to_parquet(out / "validation.parquet", index=False)
        self.test_set.to_parquet(out / "test.parquet", index=False)
        meta = {
            "random_seed": self.random_seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "sizes": self.get_split_sizes(),
        }
        (out / "split_meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("persist_to_disk: wrote splits to %s", out)

    def load_from_disk(
        self, output_dir: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        out = Path(output_dir)
        self.train_set = pd.read_parquet(out / "train.parquet")
        self.validation_set = pd.read_parquet(out / "validation.parquet")
        self.test_set = pd.read_parquet(out / "test.parquet")
        return self.train_set, self.validation_set, self.test_set

    def get_split_sizes(self) -> dict[str, int]:
        return {
            "train": 0 if self.train_set is None else len(self.train_set),
            "validation": 0 if self.validation_set is None else len(self.validation_set),
            "test": 0 if self.test_set is None else len(self.test_set),
        }

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _separate_benign_attack(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        benign_mask = df["Label_str"] == config.BENIGN_LABEL
        return df.loc[benign_mask], df.loc[~benign_mask]

    def _shuffle_deterministic(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.RandomState(self.random_seed)
        idx = rng.permutation(len(df))
        return df.iloc[idx]
