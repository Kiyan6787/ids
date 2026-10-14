import json
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from hyp.p5_monitoring.alert_manager import AlertManager

DASHBOARD_FILE = config.RESULTS_DIR / "dashboard.json"
DEPLOYMENT_FILE = config.RESULTS_DIR / "deployment.pkl"

if not DASHBOARD_FILE.exists() or not DEPLOYMENT_FILE.exists():
    raise SystemExit("No pipeline output found - run `python run_pipeline.py` first.")

dashboard = json.loads(DASHBOARD_FILE.read_text())
with open(DEPLOYMENT_FILE, "rb") as f:
    deployment = pickle.load(f)


def replay_flows() -> dict[str, AlertManager]:
    features = deployment["replay_features"]
    categories = deployment["replay_categories"]
    managers = {}
    for name, detector in deployment["detectors"].items():
        manager = AlertManager(detector, deployment["thresholds"][name])
        scores = detector.score(features)
        for flow_features, category, score in zip(features, categories, scores):
            flow = {"features": flow_features.tolist(), "category": category}
            manager.create_alert(flow, float(score))
        managers[name] = manager
    return managers


managers = replay_flows()

app = FastAPI(title="HYP anomaly detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


class VerdictRequest(BaseModel):
    verdict: str     
    reviewer: str
    notes: str = ""


def find_alert(alert_id: str):
    for manager in managers.values():
        for record in manager.alert_history:
            if record.alert_id == alert_id:
                return manager, record
    raise HTTPException(status_code=404, detail=f"No alert with ID {alert_id}")


@app.get("/api/results")
def get_results():
    return dashboard


@app.get("/api/alerts")
def get_alerts():
    alerts = []
    for name, manager in managers.items():
        for record in manager.get_alert_history():
            alerts.append({
                "id": record.alert_id,
                "model": name,
                "time": record.timestamp.isoformat(),
                "score": record.score,
                "threshold": record.threshold,
                "category": record.flow["category"],
                "features": record.flow["features"],
                "verdict": record.verdict,
                "reviewer": record.reviewer,
            })
    models = {}
    for name, manager in managers.items():
        models[name] = {
            "threshold": manager.threshold,
            "running_fpr": manager.get_running_fpr(),
            "should_re_tune": manager.should_re_tune(),
        }
    return {
        "alerts": alerts,
        "models": models,
        "replayed_flows": len(deployment["replay_categories"]),
        "feature_names": deployment["feature_names"],
        "manifest_hash": deployment["manifest_hash"],
    }


@app.post("/api/alerts/{alert_id}/verdict")
def set_verdict(alert_id: str, request: VerdictRequest):
    manager, record = find_alert(alert_id)
    if record.is_reviewed():
        raise HTTPException(status_code=409, detail="Alert already has a verdict")
    try:
        record.set_verdict(request.verdict, request.reviewer, request.notes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    if request.verdict != "inconclusive":
        manager.update_fpr(request.verdict)
    return {"id": record.alert_id, "verdict": record.verdict,
            "running_fpr": manager.get_running_fpr(),
            "should_re_tune": manager.should_re_tune()}
