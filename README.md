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
| Pipeline 2 — developed from scratch by the student | `ScratchAutoencoderDetector` (`hyp/p2_models/scratch_autoencoder.py`): pure NumPy — manual forward pass, manual backpropagation, hand-implemented Adam optimiser. No ML library. Additionally, `MetricsCalculator` (confusion matrix, precision/recall/F1/FPR, ROC/PR curves, trapezoidal AUC) and `ThresholdTuner` are implemented from first principles; scikit-learn appears only in the test suite as an independent cross-check. |
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
# Smoke test — no dataset needed, synthetic data with injected NaN/Inf
python run_pipeline.py --synthetic --seeds 0 1

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
baselines → exports all tables, curves and heatmaps with a UC-traceability
manifest.

Outputs land in `data/results/`: `comparison_table.csv`,
`multi_seed_summary.csv`, `significance_tests.csv`, `per_category_table.csv`,
`triviality_baselines.json`, ROC/PR overlays, confusion-matrix heatmaps,
per-category recall chart, and `export_manifest.json`.

## Tests

```bash
python -m pytest tests/ -q
```

35 tests mirroring the preliminary-design test plan (items 1.1–1.8, 2.1–2.9,
3.1–3.10, plus P4/P5 checks), including a Hypothesis property test for the
preprocessor and an sklearn cross-check of the hand-written metrics.

## Structure

```
config.py                     # all paths, hyperparameters, seeds, thresholds
run_pipeline.py               # top-level orchestrator (UC1→UC8)
hyp/
  p1_data/                    # DatasetLoader, Preprocessor, DataSplitter
  p2_models/                  # AnomalyDetector ABC, AE, LSTM, IF, ScratchAE, ThresholdTuner
  p3_evaluation/              # MetricsCalculator, ModelEvaluator, ResultsAnalyser (FR-EV-03)
  p4_reporting/               # ResultsExporter (headless matplotlib, manifest)
  p5_monitoring/              # AlertManager, AlertRecord (in-memory, UC9–UC11)
tests/                        # pytest suite (35 tests)
```

## Notes

- Hyperparameters in `config.py` are sensible defaults (the design documents do
  not fix them); adjust freely — everything is centralised there.
- The P5 monitoring layer is modelled in-memory per the D5 scope; the 3NF
  operational schema exists in the design docs but a database is not required
  for the honours prototype.
