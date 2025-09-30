# src/model_training.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from src.utils import get_logger, load_config

logger = get_logger("training", "logs/training_log.log")

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Imputers and transformers
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ], remainder="drop")
    return preprocessor, num_cols, cat_cols

def upsample_train(X_train, y_train, seed=42):
    # Combine then resample minority class(es) to match majority
    df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    target_col = y_train.name
    counts = df_train[target_col].value_counts()
    majority = counts.idxmax()
    majority_count = counts.max()

    dfs = []
    for cls, cnt in counts.items():
        df_cls = df_train[df_train[target_col] == cls]
        if cnt < majority_count:
            df_up = resample(df_cls, replace=True, n_samples=majority_count, random_state=seed)
            dfs.append(df_up)
        else:
            dfs.append(df_cls)
    df_balanced = pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    X_bal = df_balanced.drop(columns=[target_col])
    y_bal = df_balanced[target_col]
    logger.info(f"Balanced training set by upsampling to {majority_count} examples per class")
    return X_bal, y_bal

def train_and_save(X_train, y_train, X_test, y_test, model_path="models/fever_model.pkl", seed=42, do_hpo=False):
    # Build preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    # Optionally balance
    X_train_bal, y_train_bal = upsample_train(X_train, y_train, seed=seed)

    # Build pipeline
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Fit
    logger.info("Starting model training")
    pipeline.fit(X_train_bal, y_train_bal)
    logger.info("Model training completed")

    # cross-val on balanced train (quick)
    try:
        cv_scores = cross_val_score(pipeline, X_train_bal, y_train_bal, cv=3, scoring="f1_macro", n_jobs=-1)
        logger.info(f"CV f1_macro scores: {cv_scores}, mean: {cv_scores.mean():.4f}")
    except Exception as e:
        logger.warning(f"Cross-val failed: {e}")

    # Save pipeline (includes preprocessor + classifier)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved pipeline to {model_path}")

    return pipeline





