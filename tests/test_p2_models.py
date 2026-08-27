"""Tests for model training and threshold tuning.

Deep models run with tiny epochs/dimensions so the suite stays fast.
"""

import numpy as np
import pytest

from hyp.p2_models.autoencoder_detector import AutoencoderDetector
from hyp.p2_models.lstm_detector import LSTMDetector
from hyp.p2_models.isolation_forest_detector import IsolationForestDetector
from hyp.p2_models.scratch_autoencoder import ScratchAutoencoderDetector
from hyp.p2_models.threshold_tuner import ThresholdTuner


def benign_data(n=400, d=8, seed=0):
    return np.random.RandomState(seed).normal(0, 1, size=(n, d))


def temporal_data(n=400, d=8, seed=0):
    """AR(1)-style sequences: i.i.d. noise has no temporal structure for an
    LSTM to learn, so the loss-decrease smoke test uses autocorrelated data
    instead (real network traffic is temporally structured)."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    X[0] = rng.normal(0, 1, d)
    for t in range(1, n):
        X[t] = 0.9 * X[t - 1] + 0.1 * rng.normal(0, 1, d)
    return X


def make_detectors():
    return [
        AutoencoderDetector(latent_dim=4, hidden_dims=[16], epochs=5,
                            batch_size=64, seed=0),
        LSTMDetector(seq_len=5, hidden_dim=8, num_layers=1, epochs=5,
                     batch_size=64, seed=0),
        IsolationForestDetector(n_estimators=50, random_state=0),
        ScratchAutoencoderDetector(latent_dim=4, hidden_dim=16, epochs=5,
                                   batch_size=64, seed=0),
    ]


@pytest.mark.parametrize("detector_cls,kwargs", [
    (AutoencoderDetector, dict(latent_dim=4, hidden_dims=[16], epochs=8,
                               batch_size=64, seed=0)),
    (LSTMDetector, dict(seq_len=5, hidden_dim=8, num_layers=1, epochs=8,
                        batch_size=64, seed=0)),
    (ScratchAutoencoderDetector, dict(latent_dim=4, hidden_dim=16, epochs=8,
                                      batch_size=64, seed=0)),
])
def test_training_loss_decreases(detector_cls, kwargs):
    det = detector_cls(**kwargs)
    # LSTM predicts the next flow from a sequence, so it needs temporally
    # correlated data (AR(1)) to have anything learnable; i.i.d. Gaussian
    # noise has no next-step signal and the loss plateaus at the noise floor.
    X = temporal_data() if detector_cls is LSTMDetector else benign_data()
    metrics = det.train(X)
    losses = metrics["training_losses"]
    assert losses[-1] < losses[0]


def test_isolation_forest_trains():
    det = IsolationForestDetector(n_estimators=20, random_state=0)
    assert not det.is_model_trained()
    det.train(benign_data())
    assert det.is_model_trained()


def test_score_shape_and_dtype_uniform():
    X = benign_data()
    X_test = benign_data(n=120, seed=1)
    for det in make_detectors():
        det.train(X)
        scores = det.score(X_test)
        assert scores.shape == (len(X_test),), det.get_model_name()
        assert scores.dtype == np.float64, det.get_model_name()


def test_save_load_round_trip(tmp_path):
    X = benign_data()
    X_test = benign_data(n=100, seed=2)
    for det in make_detectors():
        det.train(X)
        original = det.score(X_test)
        path = tmp_path / f"{det.get_model_name()}.bin"
        det.save(str(path))
        fresh = type(det)()
        fresh.load(str(path))
        reloaded = fresh.score(X_test)
        assert np.allclose(original, reloaded, atol=1e-6), det.get_model_name()


def test_score_idempotent():
    X = benign_data()
    X_test = benign_data(n=100, seed=3)
    for det in make_detectors():
        det.train(X)
        s1, s2 = det.score(X_test), det.score(X_test)
        assert np.array_equal(s1, s2), det.get_model_name()


def test_sweep_fpr_monotone():
    rng = np.random.RandomState(0)
    scores = rng.rand(2000)
    labels = np.zeros(2000, dtype=int)
    tuner = ThresholdTuner()
    tuner.tune(scores, labels, criterion="fpr<=0.01")
    sweep = sorted(tuner.get_sweep_curve(), key=lambda e: e["threshold"])
    fprs = [e["fpr"] for e in sweep]
    assert all(fprs[i] >= fprs[i + 1] for i in range(len(fprs) - 1))


def test_selected_threshold_satisfies_criterion():
    rng = np.random.RandomState(1)
    scores = rng.rand(5000)
    labels = np.zeros(5000, dtype=int)
    tuner = ThresholdTuner()
    thr = tuner.tune(scores, labels, criterion="fpr<=0.01")
    achieved_fpr = np.mean(scores > thr)
    assert achieved_fpr <= 0.01


def test_boundary_inputs_fail_loudly():
    tuner = ThresholdTuner()
    with pytest.raises(ValueError):
        tuner.tune(np.array([]), np.array([]))
    # all-attack validation: no benign flows -> FPR undefined -> loud failure
    with pytest.raises(ValueError):
        tuner.tune(np.random.rand(50), np.ones(50, dtype=int))
    # scoring before training must raise
    det = AutoencoderDetector(epochs=1)
    with pytest.raises(RuntimeError):
        det.score(benign_data(n=10))
    # LSTM: input shorter than sequence length must raise
    lstm = LSTMDetector(seq_len=50, epochs=1)
    with pytest.raises(ValueError):
        lstm.train(benign_data(n=10))


# Integration: train -> threshold -> persist -> reload full chain.
def test_p2_full_chain(tmp_path):
    X_train = benign_data(n=500)
    X_val = benign_data(n=300, seed=9)
    det = IsolationForestDetector(n_estimators=30, random_state=0)
    det.train(X_train)
    tuner = ThresholdTuner()
    thr = tuner.tune(det.score(X_val), np.zeros(len(X_val)), "fpr<=0.01")
    path = tmp_path / "if.bin"
    det.save(str(path))
    reloaded = IsolationForestDetector()
    reloaded.load(str(path))
    assert reloaded.is_model_trained()
    assert np.isfinite(thr)
    assert np.mean(reloaded.score(X_val) > thr) <= 0.01 + 1e-9
