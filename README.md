# HYP — Unsupervised Anomaly Detection for Network Intrusion Detection

Backend prototype for the HYP honours project: an unsupervised, deep-learning-powered
intrusion detection system evaluated on CICIDS 2017. Implements the five-package
architecture from the D5 detailed design (P1 Data Management, P2 Model Training &
Tuning, P3 Evaluation, P4 Reporting, P5 Operational Monitoring) and the multi-seed
experimental protocol from the preliminary design.

## Guideline compliance (AI project — two pipelines)

| Guideline requirement | Where it is satisfied |
|---|---|
| Pipeline 1 — existing pipeline, libraries permitted | `IsolationForestDetector` (scikit-learn), `AutoencoderDetector` and `LSTMDetector` (PyTorch). Isolation Forest doubles as the classical benchmark. |
| Pipeline 2 — developed from scratch by the student | `ScratchIsolationForestDetector` (`hyp/p2_models/scratch_isolation_forest.py`): pure NumPy — hand-built isolation trees, vectorised path-length scoring and the c(n) normalisation from Liu et al. (2008), cross-checked against scikit-learn's `IsolationForest` in the tests. No ML library. Additionally, `MetricsCalculator` (confusion matrix, precision/recall/F1/FPR, ROC/PR curves, trapezoidal AUC) and `ThresholdTuner` are implemented from first principles; scikit-learn appears only in the test suite as an independent cross-check. |
| Publicly available dataset | CICIDS 2017 (Canadian Institute for Cybersecurity). |
| Success metrics | Precision, Recall, F1, FPR, Accuracy, ROC-AUC per model; per-attack-category recall; multi-seed mean±std; Wilcoxon significance tests; triviality baselines. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11+ recommended. CPU-only PyTorch is sufficient.

## Dataset

Download the CICIDS 2017 "MachineLearningCVE" CSV files from
https://www.unb.ca/cic/datasets/ids-2017.html and place them in `data/raw/`.
The loader concatenates every `*.csv` in that folder, normalises the whitespace
in column names, and maps the raw labels to seven attack categories plus BENIGN
(see `config.ATTACK_CATEGORY_MAP`). By default a stratified sample of
`config.SAMPLE_ROWS` (400 000) flows is used so the experiment fits on a laptop;
set it to `None` for the full ~2.8M flows.

## Running

```bash
# Full experiment: 5 seeds, all four models, CICIDS 2017 in data/raw/
python run_pipeline.py

# Subsets
python run_pipeline.py --models autoencoder isolation_forest --seeds 0 1 2
```

The pipeline: loads and validates the dataset (FR-DM-01 schema gate) → repairs
NaN/Inf, encodes, normalises → benign-only 60/20 train/validation split with all
attacks reserved for the mixed test split → trains each model on each seed →
tunes the alert threshold to FPR ≤ 1% on benign validation scores → evaluates on
the fixed test split → verifies the FR-EV-03 invariant (identical preprocessing
manifest hash across models; the comparator refuses to produce output on
mismatch) → aggregates across seeds, runs pairwise Wilcoxon tests and triviality
baselines → exports all tables, curves and heatmaps.

Outputs land in `data/results/`: `comparison_table.csv`,
`multi_seed_summary.csv`, `significance_tests.csv`, `per_category_table.csv`,
`triviality_baselines.json`, ROC/PR overlays, confusion-matrix heatmaps,
and per-category recall chart.

## API (for the Angular dashboard)

`run_pipeline.py` also writes `dashboard.json` (headline results) and
`deployment.pkl` (the models listed in `config.DEPLOYED_MODELS` plus a sample of
test flows) to `data/results/`. Start the API after a pipeline run:

```bash
uvicorn api:app --reload        # http://localhost:8000, docs at /docs
```

| Endpoint | Purpose |
|---|---|
| `GET /api/results` | Model comparison, confusion matrices, ROC/PR curves, per-category results |
| `GET /api/alerts` | Alerts raised by each deployed model + its running FPR (P5) |
| `POST /api/alerts/{id}/verdict` | Analyst verdict (`true_positive` / `false_positive` / `inconclusive`) |

On startup the API replays `config.REPLAY_FLOWS` held-out test flows through each
deployed model's `AlertManager`, which stands in for live traffic. Alerts are kept in
memory, so they reset when the API restarts.

## Tests

```bash
python -m pytest tests/ -q
```

A compact suite covering the core behaviour of each package, including an
sklearn cross-check of the hand-written metrics.

## Structure

```
config.py                     # all paths, hyperparameters, seeds, thresholds
run_pipeline.py               # top-level orchestrator (UC1→UC8)
api.py                        # FastAPI service for the dashboard (results + P5 alerts)
hyp/
  p1_data/                    # DatasetLoader, Preprocessor, DataSplitter
  p2_models/                  # AnomalyDetector ABC, AE, LSTM, IF, ScratchIF, ThresholdTuner
  p3_evaluation/              # MetricsCalculator, ModelEvaluator, ResultsAnalyser (FR-EV-03)
  p4_reporting/               # ResultsExporter (headless matplotlib)
  p5_monitoring/              # AlertManager, AlertRecord (in-memory, UC9–UC11)
tests/                        # pytest suite
```

## Notes

- Hyperparameters in `config.py` are sensible defaults (the design documents do
  not fix them); adjust freely — everything is centralised there.
- The P5 monitoring layer is modelled in-memory per the D5 scope and exposed
  through `api.py`; the 3NF operational schema exists in the design docs but a
  database is not required for the honours prototype.
