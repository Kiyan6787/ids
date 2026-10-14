"""Deterministic benign/attack-aware train/validation/test splitting.

  * Train - benign only (unsupervised training)
  * Validation - benign only (threshold tuning at FPR <= 1 %)
  * Test - remaining benign + all attacks

"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class DataSplitter:
    #Deterministic benign/attack-aware splitting of the cleaned dataset.

    def __init__(self, random_seed: int):
        self.random_seed = random_seed

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        benign_mask = df["Label_str"] == config.BENIGN_LABEL
        benign = self._shuffle(df.loc[benign_mask])
        attack = df.loc[~benign_mask]

        n_train = int(len(benign) * config.TRAIN_RATIO)
        n_val = int(len(benign) * config.VAL_RATIO)
        train = benign.iloc[:n_train].reset_index(drop=True)
        val = benign.iloc[n_train:n_train + n_val].reset_index(drop=True)
        benign_test = benign.iloc[n_train + n_val:]
        test = self._shuffle(pd.concat([benign_test, attack], ignore_index=True))
        test = test.reset_index(drop=True)

        logger.info(
            "split: train=%d (benign only), val=%d (benign only), "
            "test=%d (%d benign + %d attack)",
            len(train), len(val), len(test), len(benign_test), len(attack),
        )
        return train, val, test

    def _shuffle(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[np.random.RandomState(self.random_seed).permutation(len(df))]
