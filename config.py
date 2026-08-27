"""Central configuration: paths, dataset settings, model hyperparameters,
and experiment protocol shared by the whole pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_DIR = DATA_DIR / "raw"          # place the CICIDS 2017 CSVs here
ARTEFACT_DIR = DATA_DIR / "artefacts"   # persisted splits + preprocessing manifest
MODEL_DIR = DATA_DIR / "models"         # serialised model artefacts
RESULTS_DIR = DATA_DIR / "results"      # metrics, tables, figures

# Dataset / preprocessing
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"

# CICIDS 2017 attack-label -> category mapping, for per-category reporting.
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
    "Web Attack \u2013 Brute Force": "Web Attack",
    "Web Attack \u2013 XSS": "Web Attack",
    "Web Attack \u2013 Sql Injection": "Web Attack",
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
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 512,
}

LSTM = {
    "sequence_length": 10,
    "hidden_dim": 64,
    "num_layers": 2,
    "learning_rate": 1e-3,
    "epochs": 20,
    "batch_size": 256,
}

ISOLATION_FOREST = {
    "n_estimators": 100,
    "contamination": "auto",
    "max_samples": 256,
}

# Hand-implemented (NumPy-only, no ML library) autoencoder.
SCRATCH_AUTOENCODER = {
    "latent_dim": 16,
    "hidden_dim": 32,
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 512,
}

# Threshold tuning and experiment protocol
THRESHOLD_CRITERION = "fpr<=0.01"   # target operating point: FPR <= 1%
SEEDS = [0, 1, 2, 3, 4]             # seeds for the multi-seed run

# Max tolerated F1 std across seeds before a stability warning is logged.
MAX_F1_STD = 0.20
