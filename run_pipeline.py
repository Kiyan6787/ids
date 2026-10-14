import argparse
import json
import logging
import pickle
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
from hyp.p2_models.scratch_isolation_forest import ScratchIsolationForestDetector
from hyp.p2_models.threshold_tuner import ThresholdTuner
from hyp.p3_evaluation.metrics_calculator import MetricsCalculator as MC
from hyp.p3_evaluation.model_evaluator import ModelEvaluator
from hyp.p3_evaluation.results_analyser import ResultsAnalyser
from hyp.p4_reporting.results_exporter import ResultsExporter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_pipeline")

MODEL_NAMES = ["autoencoder", "lstm", "isolation_forest", "scratch_isolation_forest"]


def build_model(model_key: str, seed: int):
    """Create a fresh, untrained detector using the hyperparameters in config.py."""
    if model_key == "autoencoder":
        c = config.AUTOENCODER
        return AutoencoderDetector(latent_dim=c["latent_dim"], hidden_dims=c["hidden_dims"],
                                   lr=c["lr"], epochs=c["epochs"],
                                   batch_size=c["batch_size"], seed=seed)
    elif model_key == "lstm":
        c = config.LSTM
        return LSTMDetector(seq_len=c["seq_len"], hidden_dim=c["hidden_dim"],
                            num_layers=c["num_layers"], lr=c["lr"], epochs=c["epochs"],
                            batch_size=c["batch_size"], seed=seed)
    elif model_key == "isolation_forest":
        c = config.ISOLATION_FOREST
        return IsolationForestDetector(n_estimators=c["n_estimators"],
                                       contamination=c["contamination"],
                                       max_samples=c["max_samples"], seed=seed)
    elif model_key == "scratch_isolation_forest":
        c = config.SCRATCH_ISOLATION_FOREST
        return ScratchIsolationForestDetector(n_estimators=c["n_estimators"],
                                              max_samples=c["max_samples"], seed=seed)
    else:
        raise ValueError(f"Unknown model: {model_key}")


def run_p1():
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

    train, val, test = DataSplitter(random_seed=0).split(df)
    return train, val, test, pre


# Train, tune, and evaluate one (model, seed) pair
def run_model_once(model_key: str, seed: int, train, val, test,
                   feature_cols, manifest_hash):
    """Returns the evaluation results and the trained detector."""
    X_train = train[feature_cols].to_numpy()
    X_val = val[feature_cols].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_val = (val["Label_str"] != config.BENIGN_LABEL).astype(int).to_numpy()
    y_test = (test["Label_str"] != config.BENIGN_LABEL).astype(int).to_numpy()

    detector = build_model(model_key, seed)
    logger.info("=== %s (seed %d): training on %d benign flows ===",
                detector.model_name, seed, len(X_train))
    detector.train(X_train)
    threshold = ThresholdTuner.tune(detector.score(X_val), y_val, config.TARGET_FPR)
    results = ModelEvaluator(detector, threshold, manifest_hash).evaluate(X_test, y_test)
    return results, detector


# Save the deployed models for the API
def save_deployment(detectors, headline, test, feature_cols, manifest_hash):
    """Pickle the deployed detectors, their tuned thresholds and a sample of
    test flows. api.py replays those flows through AlertManager to raise alerts."""
    replay = test.sample(n=min(config.REPLAY_FLOWS, len(test)), random_state=0)
    thresholds = {ev["model_name"]: ev["threshold"] for ev in headline}
    deployment = {
        "detectors": {d.model_name: d for d in detectors},
        "thresholds": {d.model_name: thresholds[d.model_name] for d in detectors},
        "manifest_hash": manifest_hash,
        "feature_names": feature_cols,
        "replay_features": replay[feature_cols].to_numpy(),
        "replay_categories": [config.ATTACK_CATEGORY_MAP.get(s, s) for s in replay["Label_str"]],
    }
    with open(config.RESULTS_DIR / "deployment.pkl", "wb") as f:
        pickle.dump(deployment, f)
    logger.info("Saved %d deployed model(s) for the API", len(detectors))


def triviality_baselines(y_test: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    out = {}
    for name, preds in {
        "predict_all_benign": np.zeros_like(y_test),
        "predict_all_attack": np.ones_like(y_test),
        "random": rng.randint(0, 2, size=len(y_test)),
    }.items():
        out[name] = MC.compute_metrics(MC.compute_confusion_matrix(y_test, preds))["f1"]
    return out


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
    (non-parametric, so it doesn't assume normally distributed F1 scores)."""
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


def main():
    parser = argparse.ArgumentParser(description="HYP experimental pipeline")
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--models", nargs="+",
                        default=MODEL_NAMES, choices=MODEL_NAMES)
    args = parser.parse_args()

    train, val, test, pre = run_p1()
    feature_cols, manifest_hash = pre.feature_columns, pre.get_manifest_hash()
    y_test = (test["Label_str"] != config.BENIGN_LABEL).astype(int).to_numpy()
    categories = np.where(y_test == 0, "BENIGN",
                          test["Label_str"].map(
                              lambda s: config.ATTACK_CATEGORY_MAP.get(s, s)))


    all_results: dict[str, list[dict]] = {m: [] for m in args.models}
    deployed_detectors = []
    for seed in args.seeds:
        for model_key in args.models:
            results, detector = run_model_once(model_key, seed, train, val, test,
                                               feature_cols, manifest_hash)
            all_results[model_key].append(results)
            if seed == args.seeds[0] and model_key in config.DEPLOYED_MODELS:
                deployed_detectors.append(detector)


    headline = [all_results[m][0] for m in args.models]
    analyser = ResultsAnalyser(headline)
    comparison_df = analyser.compare_models()
    category_df = analyser.analyse_all_models_by_category(categories)

    seed_summary = aggregate_seeds(all_results)
    significance = pairwise_significance(all_results)
    baselines = triviality_baselines(y_test)


    if (seed_summary["f1_std"] > config.MAX_F1_STD).any():
        logger.warning("Max per-model F1 std exceeds %.2f", config.MAX_F1_STD)
    best_trivial = max(baselines.values())
    for _, row in seed_summary.iterrows():
        if row["f1_mean"] <= best_trivial + 0.05:
            logger.warning(
                "%s (F1=%.3f) does not beat trivial baselines "
                "(best=%.3f) by > 0.05", row["model"],
                row["f1_mean"], best_trivial)

    # Export tables and figures 
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

    # Dashboard results
    summary = {
        "source": "CICIDS 2017",
        "manifest_hash": manifest_hash,
        "rows_dropped": pre.rows_dropped,
        "train": len(train),
        "validation": len(val),
        "test": len(test),
        "test_attack": int(y_test.sum()),
        "test_benign": int(len(y_test) - y_test.sum()),
    }
    exporter.export_dashboard_json(headline, category_df, summary)
    save_deployment(deployed_detectors, headline, test, feature_cols, manifest_hash)

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
