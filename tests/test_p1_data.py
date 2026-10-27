"""Tests for data cleaning and splitting."""

import numpy as np
import pandas as pd

import config
from hyp.p1_data.preprocessor import Preprocessor
from hyp.p1_data.data_splitter import DataSplitter


def make_df(n=200, n_feat=5, seed=0, attack_frac=0.2):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.randn(n, n_feat), columns=[f"f{i}" for i in range(n_feat)])
    df["Label"] = np.where(rng.rand(n) < attack_frac, "DoS Hulk", "BENIGN")
    return df


def test_repair_nan_inf_removes_all_bad_values():
    df = make_df()
    df.loc[0, "f0"] = np.nan
    df.loc[1, "f1"] = np.inf
    df.loc[2, "f2"] = -np.inf
    cleaned = Preprocessor().repair_nan_inf(df)
    assert np.isfinite(cleaned.select_dtypes(include=[np.number]).to_numpy()).all()
    assert len(cleaned) == len(df) - 3


def test_clean_labels_and_normalise():
    pre = Preprocessor()
    df = pre.clean_labels(make_df(n=500), "Label")
    assert "Label_str" in df.columns
    feats = pre.normalise(df, exclude=["Label"])[pre.feature_columns]
    assert np.allclose(feats.mean(), 0.0, atol=1e-9)
    assert np.allclose(feats.std(ddof=0), 1.0, atol=1e-6)


def test_split_benign_only_train_mixed_test_deterministic():
    df = Preprocessor().clean_labels(make_df(n=1000, attack_frac=0.3), "Label")
    train, val, test = DataSplitter(random_seed=0).split(df)
    assert (train["Label_str"] == config.BENIGN_LABEL).all()
    assert (val["Label_str"] == config.BENIGN_LABEL).all()
    assert (test["Label_str"] == config.BENIGN_LABEL).any()
    # every attack row lands in test
    n_attacks = (df["Label_str"] != config.BENIGN_LABEL).sum()
    assert (test["Label_str"] != config.BENIGN_LABEL).sum() == n_attacks
    # same seed -> identical split
    pd.testing.assert_frame_equal(test, DataSplitter(random_seed=0).split(df)[2])


def test_split_keeps_time_order():
    df = Preprocessor().clean_labels(make_df(n=1000, attack_frac=0.3), "Label")
    df["row"] = np.arange(len(df))          # original (time) position
    for part in DataSplitter(random_seed=0).split(df):
        assert part["row"].is_monotonic_increasing
