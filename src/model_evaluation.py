# src/model_evaluation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import get_logger

logger = get_logger("eval", "logs/training_log.log")

def evaluate_and_report(pipeline, X_test, y_test, report_dir="reports"):
    os.makedirs(report_dir, exist_ok=True)
    preds = pipeline.predict(X_test)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)

    # Classification report
    clf_rep = classification_report(y_test, preds, output_dict=True)
    clf_rep_df = pd.DataFrame(clf_rep).transpose()
    clf_rep_df.to_csv(os.path.join(report_dir, "classification_report.csv"))
    logger.info("Saved classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=pipeline.classes_)
    cm_df = pd.DataFrame(cm, index=pipeline.classes_, columns=pipeline.classes_)
    cm_df.to_csv(os.path.join(report_dir, "confusion_matrix.csv"))

    # Save a small plot for confusion matrix
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(pipeline.classes_)))
    ax.set_xticklabels(pipeline.classes_, rotation=45)
    ax.set_yticks(range(len(pipeline.classes_)))
    ax.set_yticklabels(pipeline.classes_)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(os.path.join(report_dir, "confusion_matrix.png"))
    plt.close(fig)
    logger.info("Saved confusion_matrix.png")

    return clf_rep_df, cm_df

