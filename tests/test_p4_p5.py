"""Tests for reporting (export artefacts) and operational monitoring
(alerting, FPR tracking)."""

import numpy as np
import pandas as pd
import pytest

from hyp.p3_evaluation.metrics_calculator import MetricsCalculator as MC
from hyp.p4_reporting.results_exporter import ResultsExporter
from hyp.p5_monitoring.alert_manager import AlertManager
from hyp.p5_monitoring.alert_record import AlertRecord
from hyp.p2_models.isolation_forest_detector import IsolationForestDetector


def make_evaluation(model_name, seed=0):
    rng = np.random.RandomState(seed)
    y = (rng.rand(400) < 0.3).astype(int)
    scores = rng.normal(0, 1, 400) + 2.5 * y
    cm = MC.compute_confusion_matrix(y, (scores > 1.0).astype(int))
    return {
        "model_name": model_name, "threshold": 1.0, "confusion_matrix": cm,
        "metrics": MC.compute_metrics(cm),
    }


# ------------------------------------------------------------- reporting
def test_exporter_creates_all_artefacts(tmp_path):
    evs = [make_evaluation(m, i) for i, m in enumerate(["AE", "LSTM"])]
    comparison = pd.DataFrame([{"model": e["model_name"], **e["metrics"]} for e in evs])
    ex = ResultsExporter(str(tmp_path))
    ex.export_comparison_table(comparison)
    ex.export_confusion_matrices(evs)
    assert {p.name for p in tmp_path.iterdir()} == {
        "comparison_table.csv", "confusion_AE.png", "confusion_LSTM.png",
    }


# --------------------------------------------------------------- monitoring
def trained_detector():
    det = IsolationForestDetector(n_estimators=20, seed=0)
    det.train(np.random.RandomState(0).normal(0, 1, (300, 5)))
    return det


def test_alert_record_fields_populated():
    flow = {"features": [0.0] * 5, "source_ip": "10.0.0.1"}
    record = AlertRecord(flow, score=0.9, threshold=0.5)
    d = record.to_dict()
    for key in ("alert_id", "flow", "score", "threshold", "timestamp"):
        assert d[key] is not None
    assert not record.is_reviewed()
    record.set_verdict("false_positive", reviewer="analyst1", notes="benign burst")
    assert record.is_reviewed()
    assert AlertRecord.from_dict(record.to_dict()).verdict == "false_positive"
    with pytest.raises(ValueError):
        record.set_verdict("maybe", "analyst1")


def test_alert_only_above_threshold():
    flow = {"features": [0.0] * 5}
    quiet = AlertManager(trained_detector(), threshold=10.0)    # unreachable thr
    assert quiet.create_alert(flow, quiet.score_flow(flow)) is None
    assert quiet.get_alert_history() == []
    loud = AlertManager(trained_detector(), threshold=-10.0)    # always fires
    assert loud.create_alert(flow, loud.score_flow(flow)) is not None
    assert len(loud.get_alert_history()) == 1


def test_fpr_window_and_re_tune():
    mgr = AlertManager(trained_detector(), threshold=0.5, fpr_window_size=4)
    assert mgr.update_fpr("false_positive") == 1.0          # 1/1
    assert not mgr.should_re_tune()                         # window not full yet
    assert mgr.update_fpr("true_positive") == 0.5           # 1/2
    assert mgr.update_fpr("true_positive") == pytest.approx(1 / 3)
    assert mgr.update_fpr("false_positive") == 0.5          # 2/4
    # window slides: oldest (FP) drops out -> T,T,F + new T = 1/4
    assert mgr.update_fpr("true_positive") == pytest.approx(1 / 4)
    assert mgr.should_re_tune()                             # 0.25 > default 0.05
    with pytest.raises(ValueError):
        mgr.update_fpr("inconclusive")
