"""Central configuration: paths, dataset settings, model hyperparameters,
and experiment protocol shared by the whole pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_DIR = DATA_DIR / "raw"          # place the CICIDS 2017 CSVs here
RESULTS_DIR = DATA_DIR / "results"      # metrics, tables, figures

# Dataset / preprocessing
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"

# CICIDS 2017 attack-label -> category mapping, used to label alerts in the API.
# Keys are labels as they appear in the CSVs after whitespace stripping.
ATTACK_CATEGORY_MAP = {
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "DoS",
    "DDoS": "DDoS",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "PortScan": "Port Scan",
    "Web Attack – Brute Force": "Web Attack",
    "Web Attack – XSS": "Web Attack",
    "Web Attack – Sql Injection": "Web Attack",
    # Some CSV releases use a plain hyphen instead of an en-dash:
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Infiltration": "Infiltration",
    "Bot": "Botnet",
}

# Split ratios of the BENIGN records: train / validation. The remaining
# benign records join all attack records to form the mixed test split.
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

# Row cap for laptop-scale runs; None uses the full dataset (~2.8M rows).
SAMPLE_ROWS = 400_000

# Models
AUTOENCODER = {
    "latent_dim": 16,
    "hidden_dims": [64, 32],      # encoder: input -> 64 -> 32 -> latent
    "lr": 1e-3,
    "epochs": 30,
    "batch_size": 512,
}

LSTM = {
    "seq_len": 10,
    "hidden_dim": 64,
    "num_layers": 2,
    "lr": 1e-3,
    "epochs": 20,
    "batch_size": 256,
}

ISOLATION_FOREST = {
    "n_estimators": 100,
    "contamination": "auto",
    "max_samples": 256,
}

# Hand-implemented (NumPy-only, no ML library) Isolation Forest.
SCRATCH_ISOLATION_FOREST = {
    "n_estimators": 100,
    "max_samples": 256,
}

# Models the API deploys for live alerting (P5). Any of the run_pipeline
# model names can go here, e.g. ["isolation_forest"] to deploy only IF.
DEPLOYED_MODELS = ["isolation_forest", "autoencoder", "lstm"]
REPLAY_FLOWS = 200   # test flows the API replays through each deployed model

# Threshold tuning and experiment protocol
TARGET_FPR = 0.01                   # target operating point: FPR <= 1%
SEEDS = [0, 1, 2, 3, 4]             # seeds for the multi-seed run
