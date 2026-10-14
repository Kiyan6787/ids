"""Tests for evaluation and cross-model analysis.

test_metrics_match_sklearn is the critical one: the hand-implemented
metrics in MetricsCalculator are cross-checked against scikit-learn's
reference implementations. sklearn appears only here, never in the
metric code itself.
"""

import numpy as np
import pytest
from sklearn import metrics as skm

from hyp.p2_models.isolation_forest_detector import IsolationForestDetector
from hyp.p2_models.threshold_tuner import ThresholdTuner
from hyp.p3_evaluation.metrics_calculator import MetricsCalculator as MC
from hyp.p3_evaluation.model_evaluator import ModelEvaluator
from hyp.p3_evaluation.results_analyser import ResultsAnalyser


def labelled_scores(seed=0, n=1000, attack_frac=0.3):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < attack_frac).astype(int)
    scores = rng.normal(0, 1, n) + 2.5 * y   # attacks score higher
    return y, scores


def make_evaluation(model_name, manifest_hash, n_test=500):
    y, scores = labelled_scores(n=n_test)
    cm = MC.compute_confusion_matrix(y, (scores > 1.0).astype(int))
    return {"model_name": model_name, "manifest_hash": manifest_hash,
            "n_test_rows": n_test, "threshold": 1.0, "metrics": MC.compute_metrics(cm),
            "roc_auc": 0.9, "scores": scores, "labels": y}


def test_metrics_match_sklearn():
    y, scores = labelled_scores()
    preds = (scores > 1.0).astype(int)
    cm = MC.compute_confusion_matrix(y, preds)
    m = MC.compute_metrics(cm)
    assert m["precision"] == pytest.approx(skm.precision_score(y, preds))
    assert m["recall"] == pytest.approx(skm.recall_score(y, preds))
    assert m["f1"] == pytest.approx(skm.f1_score(y, preds))
    tn, fp, fn, tp = skm.confusion_matrix(y, preds).ravel()
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (tn, fp, fn, tp)
    roc = MC.compute_roc_curve(y, scores)
    assert MC.compute_auc(roc["fpr"], roc["tpr"]) == pytest.approx(
        skm.roc_auc_score(y, scores), abs=1e-9)


def test_invariant_refusal():
    with pytest.raises(ValueError):     # manifest hash mismatch
        ResultsAnalyser([make_evaluation("A", "h1"), make_evaluation("B", "h2")]).compare_models()
    with pytest.raises(ValueError):     # test-size mismatch
        ResultsAnalyser([make_evaluation("A", "h"),
                         make_evaluation("B", "h", n_test=400)]).compare_models()
    assert len(ResultsAnalyser([make_evaluation("A", "h"),
                                make_evaluation("B", "h")]).compare_models()) == 2


def test_per_category_reconciliation():
    y, scores = labelled_scores(n=800, seed=5)
    cats = np.where(y == 0, "BENIGN",
                    np.random.RandomState(4).choice(["DoS", "DDoS"], size=len(y)))
    df = ResultsAnalyser([]).analyse_by_category(scores, y, cats, 1.0, "M")
    assert set(df["category"]) == {"BENIGN", "DoS", "DDoS"}
    overall = MC.compute_confusion_matrix(y, (scores > 1.0).astype(int))
    for k in ("tp", "fp", "tn", "fn"):
        assert df[k].sum() == overall[k]


# Integration: Isolation Forest training -> threshold tuning -> evaluation.
def test_full_chain_if_to_metrics():
    rng = np.random.RandomState(0)
    X_benign = rng.normal(0, 1, (600, 6))
    X_attack = rng.normal(4, 1, (150, 6))
    X_train, X_val = X_benign[:400], X_benign[400:500]
    X_test = np.vstack([X_benign[500:], X_attack])
    y_test = np.array([0] * 100 + [1] * 150)

    det = IsolationForestDetector(n_estimators=50, seed=0)
    det.train(X_train)
    thr = ThresholdTuner.tune(det.score(X_val), np.zeros(100), 0.01)
    ev = ModelEvaluator(det, thr, "hash").evaluate(X_test, y_test)
    assert ev["metrics"]["recall"] > 0.9          # separable synthetic data
    assert ev["roc_auc"] > 0.9
