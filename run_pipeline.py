import argparse
import logging
import pickle
import sys

import numpy as np

import config
from hyp.p1_data.dataset_loader import DatasetLoader
from hyp.p1_data.preprocessor import Preprocessor
from hyp.p1_data.data_splitter import DataSplitter
from hyp.p2_models.autoencoder_detector import AutoencoderDetector
from hyp.p2_models.lstm_detector import LSTMDetector
from hyp.p2_models.isolation_forest_detector import IsolationForestDetector
from hyp.p2_models.scratch_isolation_forest import ScratchIsolationForestDetector
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
    y_test = (test["Label_str"] != config.BENIGN_LABEL).astype(int).to_numpy()

    detector = build_model(model_key, seed)
    logger.info("=== %s (seed %d): training on %d benign flows ===",
                detector.model_name, seed, len(X_train))
    detector.train(X_train)
    # Validation is benign-only, so this quantile gives FPR <= TARGET_FPR on it
    threshold = float(np.quantile(detector.score(X_val), 1 - config.TARGET_FPR))
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


def main():
    parser = argparse.ArgumentParser(description="HYP experimental pipeline")
    parser.add_argument("--seeds", type=int, nargs="+", default=config.SEEDS)
    parser.add_argument("--models", nargs="+",
                        default=MODEL_NAMES, choices=MODEL_NAMES)
    args = parser.parse_args()

    train, val, test, pre = run_p1()
    feature_cols, manifest_hash = pre.feature_columns, pre.get_manifest_hash()
    y_test = (test["Label_str"] != config.BENIGN_LABEL).astype(int).to_numpy()

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
    comparison_df = ResultsAnalyser(headline).compare_models()
    # Spread across seeds, to show the seed-0 numbers aren't a lucky/unlucky draw
    if len(args.seeds) > 1:
        for metric in ("precision", "recall", "f1", "fpr", "accuracy"):
            comparison_df[f"{metric}_std"] = [
                round(float(np.std([r["metrics"][metric] for r in all_results[m]], ddof=1)), 4)
                for m in args.models]

    # Export comparison table and confusion matrices
    exporter = ResultsExporter(str(config.RESULTS_DIR))
    exporter.export_comparison_table(comparison_df)
    exporter.export_confusion_matrices(headline)

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
    exporter.export_dashboard_json(headline, summary)
    save_deployment(deployed_detectors, headline, test, feature_cols, manifest_hash)

    print("\n=== Comparison (seed", args.seeds[0], ") ===")
    print(comparison_df.to_string(index=False))
    print(f"\nArtefacts written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
