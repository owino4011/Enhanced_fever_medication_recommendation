# src/utils.py
import os
import yaml
import logging
from logging.handlers import RotatingFileHandler

DEFAULT_CONFIG = {
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.0,
    "models_path": "models",
    "model_filename": "fever_model.pkl",
    "processed_csv": "data/processed/processed_data.csv",
    "do_hpo": False
}

def load_config(path="config/config.yaml"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return {**DEFAULT_CONFIG, **(cfg or {})}
    return DEFAULT_CONFIG.copy()

def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

def get_logger(name, logfile):
    ensure_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
