"""
HYP - single top-level entry point that re-runs the entire experimental
pipeline (proposal: "a single top-level script that re-runs the entire
pipeline"), implementing the full-system activity diagram:

  P1 (UC1)  load -> validate -> repair -> encode -> normalise -> manifest
            -> split -> persist
  P2 (UC2+3) for each model: train on benign -> save -> score validation
            -> tune threshold to FPR <= 1 %
  P3 (UC4-7) evaluate each model on the SAME test split -> FR-EV-03 gate
            -> comparison table -> per-category disaggregation
  P4 (UC8)  export tables / confusion matrices / ROC / PR / bar charts
            + traceability manifest

Multi-seed protocol (proposal 6.3.2): the outer loop repeats P2+P3 for
each seed with the TEST SPLIT HELD FIXED, then reports mean +/- std per
model and a paired Wilcoxon signed-rank test between model pairs, plus
the triviality baselines from regression test 5.3.

Usage:
    python run_pipeline.py                 # full run, seeds from config
    python run_pipeline.py --seeds 0 1     # subset of seeds
    python run_pipeline.py --models autoencoder isolation_forest
    python run_pipeline.py --synthetic     # smoke-run on synthetic data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd
from scipy import stats

import config
from hyp.p1_data.dataset_loader import DatasetLoader
from hyp.p1_data.preprocessor import Preprocessor
from hyp.p1_data.data_splitter import DataSplitter
from hyp.p2_models.autoencoder_detector import AutoencoderDetector
from hyp.p2_models.lstm_detector import LSTMDetector
from hyp.p2_models.isolation_forest_detector import IsolationForestDetector
from hyp.p2_models.scratch_autoencoder import ScratchAutoencoderDetector
from hyp.p2_models.threshold_tuner import ThresholdTuner
from hyp.p3_evaluation.model_evaluator import ModelEvaluator
from hyp.p3_evaluation.results_analyser import ResultsAnalyser
from hyp.p4_reporting.results_exporter import ResultsExporter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_pipeline")

MODEL_FACTORIES = {
    "autoencoder": lambda seed: AutoencoderDetector(
        latent_dim=config.AUTOENCODER["latent_dim"],
        hidden_dims=config.AUTOENCODER["hidden_dims"],
        lr=config.AUTOENCODER["learning_rate"],
        epochs=config.AUTOENCODER["epochs"],
        batch_size=config.AUTOENCODER["batch_size"],
        seed=seed,
    ),
    "lstm": lambda seed: LSTMDetector(
        seq_len=config.LSTM["sequence_length"],
        hidden_dim=config.LSTM["hidden_dim"],
        num_layers=config.LSTM["num_layers"],
        lr=config.LSTM["learning_rate"],
        epochs=config.LSTM["epochs"],
        batch_size=config.LSTM["batch_size"],
        seed=seed,
    ),
    "isolation_forest": lambda seed: IsolationForestDetector(
        n_estimators=config.ISOLATION_FOREST["n_estimators"],
        contamination=config.ISOLATION_FOREST["contamination"],
        max_samples=config.ISOLATION_FOREST["max_samples"],
        random_state=seed,
    ),
    "scratch_autoencoder": lambda seed: ScratchAutoencoderDetector(
        latent_dim=config.SCRATCH_AUTOENCODER["latent_dim"],
        hidden_dim=config.SCRATCH_AUTOENCODER["hidden_dim"],
        lr=config.SCRATCH_AUTOENCODER["learning_rate"],
        epochs=config.SCRATCH_AUTOENCODER["epochs"],
        batch_size=config.SCRATCH_AUTOENCODER["batch_size"],
        seed=seed,
    ),
}


# =========================================================================
# P1 - Data Management (UC1)
# =========================================================================
def run_p1(synthetic: bool = False):
    if synthetic:
        df = _make_synthetic_dataset()
    else:
        loader = DatasetLoader(str(config.RAW_CSV_DIR))
        df = loader.load_all_csvs()
        if not loader.validate_schema([config.LABEL_COLUMN]):
            logger.error("Schema validation failed - stopping (activity diagram gate).")
            sys.exit(1)
        if config.SAMPLE_ROWS and len(df) > config.SAMPLE_ROWS:
            df = df.sample(n=config.SAMPLE_ROWS, random_state=0).reset_index(drop=True)
            logger.info("Sampled %d rows for laptop-scale run", config.SAMPLE_ROWS)

    pre = Preprocessor()
    df = pre.repair_nan_inf(df)
    df = pre.encode_categorical(df, config.LABEL_COLUMN)
    df = pre.normalise(df, exclude=[config.LABEL_COLUMN])
    manifest = pre.get_manifest()

    # The split seed is FIXED (seed 0) across the whole experiment: the
    # multi-seed loop varies MODEL initialisation only, holding the test
    # split constant (proposal 6.3.2 "holding the test split fixed").
    splitter = DataSplitter(random_seed=0)
    train, val, test = splitter.split(df, config.LABEL_COLUMN)
    splitter.persist_to_disk(str(config.ARTEFACT_DIR))
    (config.ARTEFACT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    feature_cols = pre.get_feature_columns()
    return train, val, test, feature_cols, manifest


# =========================================================================
# P2 + P3 for one (model, seed) pair
# =========================================================================
def run_model_once(model_key: str, seed: int, train, val, test,
                   feature_cols, manifest) -> dict:
    X_train = train[feature_cols].to_numpy()
    X_val = val[feature_cols].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_val = (val["Label_str"] != config.BENIGN_LABEL).astype(int).to_numpy()
    y_test = (test["Label_str"] != config.BENIGN_LABEL).astype(int).to_numpy()

    detector = MODEL_FACTORIES[model_key](seed)
    logger.info("=== %s (seed %d): training on %d benign flows ===",
                detector.get_model_name(), seed, len(X_train))
    training_metrics = detector.train(X_train)
    model_path = config.MODEL_DIR / f"{model_key}_seed{seed}.bin"
    detector.save(str(model_path))

    tuner = ThresholdTuner()
    val_scores = detector.score(X_val)
    threshold = tuner.tune(val_scores, y_val, criterion=config.THRESHOLD_CRITERION)

    evaluator = ModelEvaluator(detector, threshold, manifest["preprocessing_hash"])
    results = evaluator.evaluate(X_test, y_test)
    results["seed"] = seed
    results["training_metrics"] = training_metrics
    return results


# =========================================================================
# Triviality baselines (regression test 5.3)
# =========================================================================
def triviality_baselines(y_test: np.ndarray, seed: int = 0) -> dict:
    from hyp.p3_evaluation.metrics_calculator import MetricsCalculator as MC
    rng = np.random.RandomState(seed)
    out = {}
    for name, preds in {
        "predict_all_benign": np.zeros_like(y_test),
        "predict_all_attack": np.ones_like(y_test),
        "random": rng.randint(0, 2, size=len(y_test)),
    }.items():
        out[name] = MC.compute_metrics(MC.compute_confusion_matrix(y_test, preds))["f1"]
    return out


# =========================================================================
# Multi-seed aggregation + significance (proposal 6.3.2)
# =========================================================================
def aggregate_seeds(all_results: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for model, results in all_results.items():
        f1s = [r["metrics"]["f1"] for r in results]
        rows.append({
            "model": model,
            "n_seeds": len(f1s),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
            "precision_mean": float(np.mean([r["metrics"]["precision"] for r in results])),
            "recall_mean": float(np.mean([r["metrics"]["recall"] for r in results])),
            "fpr_mean": float(np.mean([r["metrics"]["fpr"] for r in results])),
            "roc_auc_mean": float(np.mean([r["roc_auc"] for r in results])),
        })
    return pd.DataFrame(rows)


def pairwise_significance(all_results: dict[str, list[dict]]) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test on per-seed F1 between model pairs
    (non-parametric per proposal: 'avoids normality assumption')."""
    models = list(all_results)
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = [r["metrics"]["f1"] for r in all_results[models[i]]]
            b = [r["metrics"]["f1"] for r in all_results[models[j]]]
            if len(a) != len(b) or len(a) < 2:
                continue
            diff = np.array(a) - np.array(b)
            if np.allclose(diff, 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = stats.wilcoxon(a, b)
            rows.append({"model_a": models[i], "model_b": models[j],
                         "mean_f1_diff": float(np.mean(diff)),
                         "wilcoxon_stat": float(stat), "p_value": float(p)})
    return pd.DataFrame(rows)


# =========================================================================
# Synthetic dataset for smoke runs (no CICIDS download required)
# =========================================================================
def _make_synthetic_dataset(n_benign: int = 6000, n_attack: int = 1200,
                            n_features: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    benign = rng.normal(0, 1, size=(n_benign, n_features))
    categories = ["DoS Hulk", "DDoS", "PortScan", "FTP-Patator"]
    attack_frames, labels = [benign], [np.array(["BENIGN"] * n_benign)]
    per_cat = n_attack // len(categories)
    for k, cat in enumerate(categories):
        shift = 3.0 + k
        attack_frames.append(rng.normal(shift, 1.5, size=(per_cat, n_features)))
        labels.append(np.array([cat] * per_cat))
    X = np.vstack(attack_frames)
    y = np.concatenate(labels)
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df[config.LABEL_COLUMN] = y
    # Inject a few NaN/Inf artefacts so repair_nan_inf has real work to do.
    df.loc[rng.choice(len(df), 25, replace=False), cols[0]] = np.inf
    df.loc[rng.choice(len(df), 25, replace=False), cols[1]] = np.nan
    return df


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="HYP experimental pipeline")
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--models", nargs="+",
                        default=["autoencoder", "lstm", "isolation_forest",
                                 "scratch_autoencoder"],
                        choices=list(MODEL_FACTORIES))
    parser.add_argument("--synthetic", action="store_true",
                        help="run on synthetic data (smoke test, no CICIDS needed)")
    args = parser.parse_args()

    for d in (config.ARTEFACT_DIR, config.MODEL_DIR, config.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- P1 -------------------------------------------------------------
    train, val, test, feature_cols, manifest = run_p1(synthetic=args.synthetic)
    y_test = (test["Label_str"] != config.BENIGN_LABEL).astype(int).to_numpy()
    categories = np.where(y_test == 0, "BENIGN",
                          test["Label_str"].map(
                              lambda s: config.ATTACK_CATEGORY_MAP.get(s, s)))

    # ---- P2 + P3, multi-seed loop ---------------------------------------
    all_results: dict[str, list[dict]] = {m: [] for m in args.models}
    for seed in args.seeds:
        for model_key in args.models:
            all_results[model_key].append(
                run_model_once(model_key, seed, train, val, test,
                               feature_cols, manifest))

    # ---- FR-EV-03 gate + comparison on the FIRST seed (headline run) ----
    headline = [all_results[m][0] for m in args.models]
    analyser = ResultsAnalyser(headline)
    if not analyser.verify_invariant():
        logger.error("FR-EV-03 invariant violated - refusing comparative output.")
        sys.exit(1)
    comparison_df = analyser.compare_models()
    category_df = analyser.analyse_all_models_by_category(categories)

    # ---- Multi-seed statistics ------------------------------------------
    seed_summary = aggregate_seeds(all_results)
    significance = pairwise_significance(all_results)
    baselines = triviality_baselines(y_test)

    # Regression gates 5.2 / 5.3
    if (seed_summary["f1_std"] > config.MAX_F1_STD).any():
        logger.warning("Regression 5.2: max per-model F1 std exceeds %.2f",
                       config.MAX_F1_STD)
    for _, row in seed_summary.iterrows():
        best_trivial = max(baselines.values())
        if row["f1_mean"] <= best_trivial + 0.05:
            logger.warning(
                "Regression 5.3: %s (F1=%.3f) does not beat trivial "
                "baselines (best=%.3f) by > 0.05", row["model"],
                row["f1_mean"], best_trivial)

    # ---- P4 export -------------------------------------------------------
    exporter = ResultsExporter(str(config.RESULTS_DIR))
    exporter.export_tables(comparison_df, category_df)
    exporter.export_confusion_matrices(headline)
    exporter.export_roc_curves(headline)
    exporter.export_pr_curves(headline)
    exporter.export_category_charts(category_df)
    seed_summary.to_csv(config.RESULTS_DIR / "multi_seed_summary.csv", index=False)
    significance.to_csv(config.RESULTS_DIR / "significance_tests.csv", index=False)
    (config.RESULTS_DIR / "triviality_baselines.json").write_text(
        json.dumps(baselines, indent=2))
    exporter.manifest.extend([
        {"file": "multi_seed_summary.csv", "source_uc": "UC6",
         "description": "Mean +/- std across seeds per model"},
        {"file": "significance_tests.csv", "source_uc": "UC6",
         "description": "Paired Wilcoxon signed-rank tests on per-seed F1"},
        {"file": "triviality_baselines.json", "source_uc": "UC6",
         "description": "Trivial-classifier F1 baselines (regression 5.3)"},
    ])
    exporter.write_manifest()

    print("\n=== Comparison (seed", args.seeds[0], ") ===")
    print(comparison_df.to_string(index=False))
    print("\n=== Multi-seed summary ===")
    print(seed_summary.to_string(index=False))
    if not significance.empty:
        print("\n=== Pairwise significance (Wilcoxon on F1) ===")
        print(significance.to_string(index=False))
    print(f"\nArtefacts written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
