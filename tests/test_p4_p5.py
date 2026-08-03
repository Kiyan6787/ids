"""Tests for P4 (Reporting) and P5 (Operational Monitoring).

P4: file-existence checks, manifest completeness, headless Matplotlib
smoke test (preliminary test plan, P4 row).
P5: AlertRecord field population, sub-threshold non-alert, FPR update
arithmetic (test plan, P5 row).
"""

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
    roc = MC.compute_roc_curve(y, scores)
    return {
        "model_name": model_name, "manifest_hash": "h", "n_test_rows": 400,
        "threshold": 1.0, "confusion_matrix": cm,
        "metrics": MC.compute_metrics(cm), "roc_curve": roc,
        "pr_curve": MC.compute_pr_curve(y, scores),
        "roc_auc": MC.compute_auc(np.array(roc["fpr"]), np.array(roc["tpr"])),
        "scores": scores, "labels": y,
    }


# ---------------------------------------------------------------------- P4
def test_exporter_creates_all_artefacts(tmp_path):
    evs = [make_evaluation(m, i) for i, m in enumerate(["AE", "LSTM", "IF"])]
    comparison = pd.DataFrame([{"model": e["model_name"], **e["metrics"]}
                               for e in evs])
    category = pd.DataFrame({
        "model": ["AE", "AE"], "category": ["DoS", "BENIGN"],
        "recall": [0.9, None], "n_flows": [10, 20],
        "tp": [9, 0], "fp": [0, 1], "tn": [0, 19], "fn": [1, 0],
        "fpr": [None, 0.05],
    })
    ex = ResultsExporter(str(tmp_path))
    ex.export_tables(comparison, category)
    ex.export_confusion_matrices(evs)
    ex.export_roc_curves(evs)
    ex.export_pr_curves(evs)
    ex.export_category_charts(category)
    ex.write_manifest()

    for f in ex.get_exported_files():
        assert (tmp_path / f).exists(), f
    assert (tmp_path / "export_manifest.json").exists()
    import json
    manifest = json.loads((tmp_path / "export_manifest.json").read_text())
    # manifest completeness: every file maps back to a source UC
    assert all("source_uc" in entry for entry in manifest["files"])
    assert len(manifest["files"]) == len(ex.get_exported_files())


# ---------------------------------------------------------------------- P5
def trained_detector():
    det = IsolationForestDetector(n_estimators=20, random_state=0)
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
    round_trip = AlertRecord.from_dict(record.to_dict())
    assert round_trip.verdict == "false_positive"
    with pytest.raises(ValueError):
        record.set_verdict("maybe", "analyst1")


def test_sub_threshold_flow_produces_no_alert():
    mgr = AlertManager(trained_detector(), threshold=10.0)  # unreachable thr
    flow = {"features": [0.0] * 5}
    score = mgr.score_flow(flow)
    assert mgr.create_alert(flow, score) is None
    assert mgr.get_alert_history() == []


def test_above_threshold_flow_produces_alert():
    mgr = AlertManager(trained_detector(), threshold=-10.0)  # always fires
    flow = {"features": [5.0] * 5}
    score = mgr.score_flow(flow)
    record = mgr.create_alert(flow, score)
    assert record is not None
    assert len(mgr.get_alert_history()) == 1


def test_fpr_update_arithmetic():
    mgr = AlertManager(trained_detector(), threshold=0.5, fpr_window_size=4)
    assert mgr.update_fpr("false_positive") == 1.0          # 1/1
    assert mgr.update_fpr("true_positive") == 0.5           # 1/2
    assert mgr.update_fpr("true_positive") == pytest.approx(1 / 3)
    assert mgr.update_fpr("false_positive") == 0.5          # 2/4
    # window slides: oldest (FP) drops out -> 1/4... window now T,T,F + new
    assert mgr.update_fpr("true_positive") == pytest.approx(1 / 4)
    with pytest.raises(ValueError):
        mgr.update_fpr("inconclusive")


def test_should_re_tune_triggers_on_drift():
    mgr = AlertManager(trained_detector(), threshold=0.5,
                       fpr_window_size=5, re_tune_trigger=0.4)
    for verdict in ["false_positive"] * 3 + ["true_positive"] * 2:
        mgr.update_fpr(verdict)
    assert mgr.get_running_fpr() == 0.6
    assert mgr.should_re_tune()
