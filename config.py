"""
HYP - Unsupervised Anomaly Detection for Network Intrusion
Central configuration.

All tunable values live here so that every experiment run is fully
described by (this file + the preprocessing manifest + the seed list).
This is part of the FR-EV-03 reproducibility story: no hidden defaults
scattered through the codebase.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_DIR = DATA_DIR / "raw"            # place the 8 CICIDS 2017 CSVs here
ARTEFACT_DIR = DATA_DIR / "artefacts"     # persisted splits + manifest (UC1)
MODEL_DIR = DATA_DIR / "models"           # serialised model artefacts (UC2)
RESULTS_DIR = DATA_DIR / "results"        # metrics, tables, figures (UC8)

# ---------------------------------------------------------------------------
# Dataset / preprocessing (P1)
# ---------------------------------------------------------------------------
LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"

# CICIDS 2017 attack-label -> category mapping used for UC7 per-category
# disaggregation. Keys are the labels as they appear in the CSVs (after
# whitespace stripping); values are the seven categories in the proposal.
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

# Split ratios (of the BENIGN records): train / validation. The remaining
# benign records join all attack records to form the mixed test split.
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

# Optional row-sampling for laptop-scale runs (D6: "project uses a sampled
# subset"). Set to None to use the full ~2.8M rows.
SAMPLE_ROWS = 400_000

# ---------------------------------------------------------------------------
# Models (P2)
# ---------------------------------------------------------------------------
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

# From-scratch NumPy autoencoder (the "developed by the student from
# scratch" pipeline required by the AI project guidelines).
SCRATCH_AUTOENCODER = {
    "latent_dim": 16,
    "hidden_dim": 32,
    "learning_rate": 1e-3,
    "epochs": 30,
    "batch_size": 512,
}

# ---------------------------------------------------------------------------
# Threshold tuning (UC3) and experiment protocol
# ---------------------------------------------------------------------------
THRESHOLD_CRITERION = "fpr<=0.01"   # FPR <= 1 % operating point
SEEDS = [0, 1, 2, 3, 4]             # N = 5 seeds per model (proposal 6.3.2)

# Regression gate 5.2 from the preliminary test plan.
MAX_F1_STD = 0.20
