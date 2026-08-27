"""Tests for data loading, cleaning, and splitting."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import config
from hyp.p1_data.preprocessor import Preprocessor
from hyp.p1_data.data_splitter import DataSplitter


def make_df(n=200, n_feat=5, seed=0, attack_frac=0.2):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.randn(n, n_feat), columns=[f"f{i}" for i in range(n_feat)])
    labels = np.where(rng.rand(n) < attack_frac, "DoS Hulk", "BENIGN")
    df["Label"] = labels
    return df


def test_repair_nan_inf_removes_all_bad_values():
    df = make_df()
    df.loc[0, "f0"] = np.nan
    df.loc[1, "f1"] = np.inf
    df.loc[2, "f2"] = -np.inf
    pre = Preprocessor()
    cleaned = pre.repair_nan_inf(df)
    assert not cleaned.isna().any().any()
    assert np.isfinite(cleaned.select_dtypes(include=[np.number]).to_numpy()).all()
    assert len(cleaned) == len(df) - 3


def test_normalise_zero_mean_unit_std():
    df = make_df(n=500)
    pre = Preprocessor()
    df = pre.encode_categorical(df, "Label")
    out = pre.normalise(df, exclude=["Label"])
    feats = out[pre.get_feature_columns()]
    assert np.allclose(feats.mean(), 0.0, atol=1e-9)
    assert np.allclose(feats.std(ddof=0), 1.0, atol=1e-6)


def test_encode_categorical_integer_column():
    df = make_df()
    pre = Preprocessor()
    out = pre.encode_categorical(df, "Label")
    assert pd.api.types.is_integer_dtype(out["Label"])
    assert "Label_str" in out.columns


# Property test: row count should equal input rows minus drops, for any random DataFrame.
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=5, max_value=200),
    n_bad=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_row_count_arithmetic_property(n, n_bad, seed):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.randn(n, 3), columns=["a", "b", "c"])
    n_bad = min(n_bad, n)
    bad_idx = rng.choice(n, size=n_bad, replace=False)
    df.loc[bad_idx, "a"] = np.nan
    pre = Preprocessor()
    cleaned = pre.repair_nan_inf(df)
    dropped = sum(e["rows_dropped"] for e in pre.get_drop_log())
    assert len(cleaned) == len(df) - dropped


def test_split_benign_only_train_mixed_test():
    df = make_df(n=1000, attack_frac=0.3)
    pre = Preprocessor()
    df = pre.encode_categorical(df, "Label")
    train, val, test = DataSplitter(random_seed=0).split(df, "Label")
    assert (train["Label_str"] == config.BENIGN_LABEL).all()
    assert (val["Label_str"] == config.BENIGN_LABEL).all()
    assert (test["Label_str"] == config.BENIGN_LABEL).any()
    assert (test["Label_str"] != config.BENIGN_LABEL).any()
    # every attack row lands in test
    n_attacks = (df["Label_str"] != config.BENIGN_LABEL).sum()
    assert (test["Label_str"] != config.BENIGN_LABEL).sum() == n_attacks


def test_split_determinism_same_seed():
    df = make_df(n=800, attack_frac=0.25)
    pre = Preprocessor()
    df = pre.encode_categorical(df, "Label")
    t1, v1, s1 = DataSplitter(random_seed=7).split(df, "Label")
    t2, v2, s2 = DataSplitter(random_seed=7).split(df, "Label")
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(s1, s2)


def test_manifest_required_keys():
    df = make_df()
    pre = Preprocessor()
    df = pre.repair_nan_inf(df)
    df = pre.encode_categorical(df, "Label")
    pre.normalise(df, exclude=["Label"])
    manifest = pre.get_manifest()
    for key in ("steps", "feature_columns", "label_column",
                "drop_log", "preprocessing_hash"):
        assert key in manifest
    assert len(manifest["preprocessing_hash"]) == 64  # SHA-256 hex


# Integration: full clean -> split -> persist -> reload round trip.
def test_uc1_end_to_end_persist_and_reload(tmp_path):
    df = make_df(n=600, attack_frac=0.2)
    pre = Preprocessor()
    df = pre.repair_nan_inf(df)
    df = pre.encode_categorical(df, "Label")
    df = pre.normalise(df, exclude=["Label"])
    splitter = DataSplitter(random_seed=1)
    train, val, test = splitter.split(df, "Label")
    splitter.persist_to_disk(str(tmp_path))
    t2, v2, s2 = DataSplitter(random_seed=1).load_from_disk(str(tmp_path))
    pd.testing.assert_frame_equal(train, t2)
    pd.testing.assert_frame_equal(val, v2)
    pd.testing.assert_frame_equal(test, s2)
