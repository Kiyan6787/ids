"""Central configuration: paths, dataset settings, model hyperparameters,
and experiment protocol shared by the whole pipeline."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_DIR = DATA_DIR / "raw"          
RESULTS_DIR = DATA_DIR / "results"     

CSV_FILE_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"

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
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
    "Infiltration": "Infiltration",
    "Bot": "Botnet",
}

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
SAMPLE_ROWS = 400_000

# Models
AUTOENCODER = {
    "latent_dim": 16,
    "hidden_dims": [64, 32],   
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

SCRATCH_ISOLATION_FOREST = {
    "n_estimators": 100,
    "max_samples": 256,
}

DEPLOYED_MODELS = ["isolation_forest", "autoencoder", "lstm"]
REPLAY_FLOWS = 200 

TARGET_FPR = 0.01                
SEEDS = [0, 1, 2, 3, 4]          
