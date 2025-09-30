# src/data_preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import get_logger

logger = get_logger("pipeline", "logs/training_log.log")

DROP_COLS = ["Previous_Medication"]  # as requested

def load_raw_csv(raw_dir="data/raw"):
    # find first CSV in raw_dir
    raw_dir = os.path.abspath(raw_dir)
    csvs = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {raw_dir}")
    path = os.path.join(raw_dir, csvs[0])
    logger.info(f"Loading raw CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    return df

def preprocess_and_split(df, target_col="Recommended_Medication", test_size=0.2, seed=42):
    # Drop requested columns
    df = df.copy()
    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            logger.info(f"Dropped column: {c}")

    # Basic cleaning: strip strings
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": pd.NA})

    # Ensure target exists
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    # Split features / target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str)

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    logger.info(f"Performed train/test split: train={len(X_train)}, test={len(X_test)}")
    # save processed (concatenate for reproducibility)
    processed_df = pd.concat([X_train, y_train.rename(target_col)], axis=1)
    os.makedirs(os.path.dirname("data/processed/processed_data.csv"), exist_ok=True)
    processed_df.to_csv("data/processed/processed_data.csv", index=False)
    logger.info("Saved processed sample to data/processed/processed_data.csv (train portion only)")
    return X_train, X_test, y_train, y_test

