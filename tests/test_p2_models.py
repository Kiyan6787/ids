"""Tests for model training and scoring.

Deep models run with tiny epochs/dimensions so the suite stays fast.
"""

import numpy as np
import pytest
from scipy import stats

from hyp.p2_models.autoencoder_detector import AutoencoderDetector
from hyp.p2_models.lstm_detector import LSTMDetector
from hyp.p2_models.isolation_forest_detector import IsolationForestDetector
from hyp.p2_models.scratch_isolation_forest import (ScratchIsolationForestDetector,
                                                    average_path_length)


def benign_data(n=400, d=8, seed=0):
    return np.random.RandomState(seed).normal(0, 1, size=(n, d))


def temporal_data(n=400, d=8, seed=0):
    """AR(1)-style sequences: i.i.d. noise has no temporal structure for an
    LSTM to learn, so the loss-decrease test uses autocorrelated data
    instead (real network traffic is temporally structured)."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    X[0] = rng.normal(0, 1, d)
    for t in range(1, n):
        X[t] = 0.9 * X[t - 1] + 0.1 * rng.normal(0, 1, d)
    return X


@pytest.mark.parametrize("detector_cls,kwargs", [
    (AutoencoderDetector, dict(latent_dim=4, hidden_dims=[16], epochs=8,
                               batch_size=64, seed=0)),
    (LSTMDetector, dict(seq_len=5, hidden_dim=8, num_layers=1, epochs=8,
                        batch_size=64, seed=0)),
])
def test_training_loss_decreases(detector_cls, kwargs):
    det = detector_cls(**kwargs)
    det.train(temporal_data() if detector_cls is LSTMDetector else benign_data())
    assert det.training_losses[-1] < det.training_losses[0]


def test_score_shape_and_dtype_uniform():
    X_test = benign_data(n=120, seed=1)
    for det in [
        AutoencoderDetector(latent_dim=4, hidden_dims=[16], epochs=2, batch_size=64),
        LSTMDetector(seq_len=5, hidden_dim=8, num_layers=1, epochs=2, batch_size=64),
        IsolationForestDetector(n_estimators=50),
        ScratchIsolationForestDetector(n_estimators=50),
    ]:
        det.train(benign_data())
        scores = det.score(X_test)
        assert scores.shape == (len(X_test),), det.model_name
        assert scores.dtype == np.float64, det.model_name


def test_average_path_length_matches_paper():
    # c(n) = 2H(n-1) - 2(n-1)/n, with c(1) = 0 and c(2) = 1
    assert average_path_length(1) == 0.0
    assert average_path_length(2) == 1.0
    harmonic_255 = sum(1.0 / i for i in range(1, 256))
    assert average_path_length(256) == pytest.approx(2 * harmonic_255 - 2 * 255 / 256,
                                                     abs=0.01)


def test_scratch_isolation_forest_agrees_with_sklearn():
    """Cross-check against scikit-learn's IsolationForest: both should rank
    flows in nearly the same order, and both should flag the injected
    outliers above every normal point."""
    X_train = benign_data(n=2000)
    X_test = np.vstack([benign_data(n=300, seed=1),
                        np.random.RandomState(2).normal(6, 1, size=(10, 8))])
    scratch = ScratchIsolationForestDetector(n_estimators=200, seed=0)
    library = IsolationForestDetector(n_estimators=200, seed=0)
    scratch.train(X_train)
    library.train(X_train)
    s_scratch, s_library = scratch.score(X_test), library.score(X_test)

    assert stats.spearmanr(s_scratch, s_library).statistic > 0.9
    assert s_scratch[-10:].min() > s_scratch[:-10].max()
    assert np.all((s_scratch > 0) & (s_scratch <= 1))


def test_scratch_isolation_forest_is_reproducible():
    X = benign_data()
    a = ScratchIsolationForestDetector(n_estimators=20, seed=3)
    b = ScratchIsolationForestDetector(n_estimators=20, seed=3)
    a.train(X)
    b.train(X)
    np.testing.assert_array_equal(a.score(X), b.score(X))

