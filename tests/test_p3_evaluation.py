"""Tests for P3 - Evaluation & Analysis, mapped to test plan 3.1-3.10.

Test 3.3 is the guideline-critical one: the hand-implemented metrics in
MetricsCalculator are cross-checked against scikit-learn's reference
implementations. The library appears ONLY here, never in the metric code.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm

import config
from hyp.p2_models.isolation_forest_detector import IsolationForestDetector
from hyp.p2_models.threshold_tuner import ThresholdTuner
from hyp.p3_evaluation.metrics_calculator import MetricsCalculator as MC
from hyp.p3_evaluation.model_evaluator import ModelEvaluator
from hyp.p3_evaluation.results_analyser import ResultsAnalyser, InvariantViolation


def labelled_scores(seed=0, n=1000, attack_frac=0.3):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < attack_frac).astype(int)
    scores = rng.normal(0, 1, n) + 2.5 * y   # attacks score higher
    return y, scores


# 3.1 - Exactly one score per flow; alignment preserved under reordering
def test_one_score_per_flow_alignment():
    X_train = np.random.RandomState(0).normal(0, 1, (300, 6))
    X_test = np.random.RandomState(1).normal(0, 1, (150, 6))
    det = IsolationForestDetector(n_estimators=30, random_state=0)
    det.train(X_train)
    scores = det.score(X_test)
    assert scores.shape == (150,)
    perm = np.random.RandomState(2).permutation(150)
    assert np.allclose(det.score(X_test[perm]), scores[perm])


# 3.2 - Metrics correct to three decimal places on known confusion matrix
def test_metrics_known_values():
    cm = {"tp": 80, "fp": 20, "tn": 880, "fn": 20}
    m = MC.compute_metrics(cm)
    assert round(m["precision"], 3) == 0.800
    assert round(m["recall"], 3) == 0.800
    assert round(m["f1"], 3) == 0.800
    assert round(m["fpr"], 3) == 0.022


# 3.3 - Custom metrics match scikit-learn exactly
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
    # ROC-AUC cross-check
    roc = MC.compute_roc_curve(y, scores)
    auc = MC.compute_auc(np.array(roc["fpr"]), np.array(roc["tpr"]))
    assert auc == pytest.approx(skm.roc_auc_score(y, scores), abs=1e-9)


# 3.4 - Edge cases do not crash
def test_metric_edge_cases():
    assert MC.compute_metrics({"tp": 0, "fp": 0, "tn": 10, "fn": 0})["precision"] == 0.0
    assert MC.compute_metrics({"tp": 0, "fp": 0, "tn": 0, "fn": 10})["recall"] == 0.0
    assert MC.compute_metrics({"tp": 0, "fp": 0, "tn": 0, "fn": 0})["f1"] == 0.0


# 3.5 - ROC AUC exceeds 0.5 for a genuinely informative scorer
def test_roc_auc_above_half():
    y, scores = labelled_scores()
    roc = MC.compute_roc_curve(y, scores)
    assert MC.compute_auc(np.array(roc["fpr"]), np.array(roc["tpr"])) > 0.5


def make_evaluation(model_name, manifest_hash, n_test=500, seed=0):
    y, scores = labelled_scores(seed=seed, n=n_test)
    cm = MC.compute_confusion_matrix(y, (scores > 1.0).astype(int))
    roc = MC.compute_roc_curve(y, scores)
    return {
        "model_name": model_name, "manifest_hash": manifest_hash,
        "n_test_rows": n_test, "threshold": 1.0,
        "confusion_matrix": cm, "metrics": MC.compute_metrics(cm),
        "roc_curve": roc, "pr_curve": MC.compute_pr_curve(y, scores),
        "roc_auc": MC.compute_auc(np.array(roc["fpr"]), np.array(roc["tpr"])),
        "scores": scores, "labels": y,
    }


# 3.6 - Contract: comparator refuses on manifest-hash mismatch
def test_invariant_refusal():
    evs = [make_evaluation("A", "hash1"), make_evaluation("B", "hash2")]
    analyser = ResultsAnalyser(evs)
    assert not analyser.verify_invariant()
    with pytest.raises(InvariantViolation):
        analyser.compare_models()
    # size mismatch also refused
    evs2 = [make_evaluation("A", "h"), make_evaluation("B", "h", n_test=400)]
    with pytest.raises(InvariantViolation):
        ResultsAnalyser(evs2).compare_models()


# 3.7 - Comparison table: one row per model, all metrics, no NaNs
def test_comparison_table_complete():
    evs = [make_evaluation(m, "h", seed=i) for i, m in
           enumerate(["Autoencoder", "LSTM", "IsolationForest"])]
    df = ResultsAnalyser(evs).compare_models()
    assert len(df) == 3
    for col in ("precision", "recall", "f1", "fpr"):
        assert col in df.columns
    assert not df.isna().any().any()


# 3.8 - Per-category coverage: every category appears in the output
def test_per_category_coverage():
    y, scores = labelled_scores(n=700)
    cats = np.where(y == 0, "BENIGN",
                    np.random.RandomState(3).choice(
                        ["DoS", "DDoS", "Port Scan", "Botnet"], size=len(y)))
    ev = make_evaluation("M", "h", n_test=700)
    ev["scores"], ev["labels"] = scores, y
    df = ResultsAnalyser([ev]).analyse_by_category(scores, y, cats, 1.0, "M")
    assert set(df["category"]) == set(np.unique(cats))


# 3.9 - Per-category counts reconcile to the overall confusion matrix
def test_per_category_reconciliation():
    y, scores = labelled_scores(n=800, seed=5)
    cats = np.where(y == 0, "BENIGN",
                    np.random.RandomState(4).choice(["DoS", "DDoS"], size=len(y)))
    df = ResultsAnalyser([make_evaluation("M", "h")]).analyse_by_category(
        scores, y, cats, 1.0, "M")
    overall = MC.compute_confusion_matrix(y, (scores > 1.0).astype(int))
    for k in ("tp", "fp", "tn", "fn"):
        assert df[k].sum() == overall[k]


# 3.10 - Integration: raw data -> IF training -> persisted metrics
def test_full_chain_if_to_metrics(tmp_path):
    rng = np.random.RandomState(0)
    X_benign = rng.normal(0, 1, (600, 6))
    X_attack = rng.normal(4, 1, (150, 6))
    X_train, X_val = X_benign[:400], X_benign[400:500]
    X_test = np.vstack([X_benign[500:], X_attack])
    y_test = np.r_[np.zeros(100), np.ones(150)].astype(int)

    det = IsolationForestDetector(n_estimators=50, random_state=0)
    det.train(X_train)
    thr = ThresholdTuner().tune(det.score(X_val), np.zeros(100), "fpr<=0.01")
    ev = ModelEvaluator(det, thr, "hash").evaluate(X_test, y_test)
    assert ev["metrics"]["recall"] > 0.9          # separable synthetic data
    assert ev["roc_auc"] > 0.9
    out = tmp_path / "metrics.json"
    import json
    out.write_text(json.dumps(ev["metrics"]))
    assert out.exists()
