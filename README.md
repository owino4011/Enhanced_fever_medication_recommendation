**Enhanced Fever Medication Recommendation System**

**Overview**
The Enhanced Fever Medication Recommendation System is an end-to-end machine learning pipeline that trains a classifier to recommend fever medication (e.g., Ibuprofen or Paracetamol) from patient data. The repository includes data ingestion, preprocessing, balancing, model training, evaluation, and a Streamlit UI for inference. This README explains project structure, how to run each stage, troubleshooting tips, and best practices for reproducibility.

**Project layout**
Enhanced_fever_medication_recommendation/
├── data/
│   ├── raw/                         # Put raw dataset (CSV or zipped dataset to be extracted)
│   ├── processed/                   # Processed CSVs (train portion saved by preprocessing)
├── src/                             # Core library code
│   ├── __init__.py
│   ├── data_preprocessing.py        # load_raw_csv(), preprocess_and_split()
│   ├── model_training.py            # build_preprocessor(), upsample_train(), train_and_save()
│   ├── model_evaluation.py          # evaluate_and_report()
│   ├── utils.py                     # logging, config helpers
├── scripts/                         # CLI wrappers/orchestration scripts
│   ├── data_ingestion.py
│   ├── split_data.py
│   ├── balance_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
├── models/
│   ├── fever_model.pkl              # trained pipeline (preprocessor + model)
├── notebooks/
│   ├── 01_raw_data_eda.ipynb
│   ├── 02_processed_data_eda.ipynb
├── logs/
│   ├── training_log.log
│   ├── streamlit_output.log
│   ├── evaluation_log.log
├── reports/                         # saved evaluation artifacts (classification_report.csv, confusion_matrix.png)
├── app.py                           # Streamlit app (UI)
├── main.py                          # Orchestrator (run stages / run all)
├── requirements.txt                 # Dependencies
├── setup.py                         # Optional package installer
├── .gitignore
└── README.md

Quick start (recommended)

Python environment

Use Python 3.8+ (3.11/3.12 are common). Create & activate a virtual environment:

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate


**Install dependencies**

pip install --upgrade pip
pip install -r requirements.txt


(If reproducibility is critical, pin versions in requirements.txt or create requirements-lock.txt.)

Place your dataset

If your dataset is zipped (e.g. enhanced_fever_medicine_recommendation.zip) put it somewhere (for example C:\Users\<Oscar's>\Desktop\MyDatasets\...) and run the data_ingestion.py script (or the CLI wrapper) to extract the CSV into data/raw/data.csv. The ingestion script expects the raw CSV to end up in data/raw/ with a .csv extension.

**Run the full pipeline**

python main.py --run all


This executes ingestion → preprocessing → training → evaluation (depending on your main.py options). The trained pipeline will be saved to models/fever_model.pkl and evaluation reports saved to reports/.

**Launch the app (after training)**

streamlit run app.py


The Streamlit UI will load the saved models/fever_model.pkl and allow interactive predictions.

Running individual stages

main.py supports flags to run either the whole pipeline or individual stages (depending on your version of main.py). Examples:

Full pipeline (default)

python main.py --run all


**Train only**

python main.py --run train


**Evaluate only**

python main.py --run eval


(Optional) Add more granular flags in main.py if you want ingest, preprocess etc.

**What each stage does (concise)**

Data ingestion (scripts/data_ingestion.py): extracts zipped dataset and moves CSV into data/raw/data.csv.

Preprocessing (src/data_preprocessing.py): drops Previous_Medication, basic cleaning (strip strings), checks target presence, performs stratified train/test split, and writes a processed sample to data/processed/processed_data.csv. Returns X_train, X_test, y_train, y_test.

Balancing (src/model_training.py → upsample_train): performs upsampling of minority classes in training set to match the majority class count (so the pipeline learns on a balanced training set).

Model training (src/model_training.py): builds a ColumnTransformer preprocessor (numeric: median imputer + StandardScaler; categorical: most frequent imputer + OneHotEncoder), fits a RandomForestClassifier, runs quick cross-val (f1_macro) on the balanced training set, and saves the trained pipeline using joblib to models/fever_model.pkl.

Evaluation (src/model_evaluation.py): generates classification_report.csv, confusion_matrix.csv and confusion_matrix.png in reports/. Evaluation logs are written to logs/evaluation_log.log.

Deployment / Streamlit (app.py): loads models/fever_model.pkl, presents inputs (in the same column order as the raw CSV), allows selection via dropdowns (categorical) and numeric inputs (min/max/default derived from training stats), performs prediction, shows confidence, lets user download the result CSV, and includes a Clear Inputs button to reset UI to defaults.

**Important implementation notes & tips**

Column order & UI: The Streamlit UI reads the raw CSV column order and exposes inputs in the same order (omitting Previous_Medication and the target Recommended_Medication).

Defaults for numeric inputs in Streamlit: defaults are set to the median value from the raw data; min and max are the observed min and max from the dataset. The Clear Inputs button resets fields to these defaults via streamlit.session_state.

OneHotEncoder compatibility: OneHotEncoder parameter names evolved across scikit-learn versions. If you see TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse', ensure you have scikit-learn >= 1.2.0. This project has been tested with scikit-learn 1.6.1. To guarantee compatibility you can pin:

scikit-learn==1.6.1


Reproducibility: src/utils.py provides load_config() and ensure_dirs() — store hyperparams and paths in config/config.yaml if you want to modify split ratios, random seed, or HPO flags without changing code.

**Logs:**

logs/training_log.log — ingestion/preprocessing/training logs.

logs/evaluation_log.log — evaluation-specific logs (classification report saved, confusion matrix saved).

logs/streamlit_output.log — streamlit app activity.

Model artifact: a single joblib pipeline (models/fever_model.pkl) contains both preprocessor and classifier. This ensures preprocessor + model consistency at inference.

**Quick verification & utilities**

Confirm the processed file exists:

import os
os.path.exists("data/processed/processed_data.csv")


Check target distribution (value counts & proportions):

import pandas as pd
df = pd.read_csv("data/processed/processed_data.csv")
print(df["Recommended_Medication"].value_counts())
print(df["Recommended_Medication"].value_counts(normalize=True))


If evaluation looks too perfect (e.g., CV scores of 1.0), inspect data leakage (features containing label info) or mistakes in preprocessing. Perfect scores are suspicious—double-check training/test separation and feature leakage.

Troubleshooting (common issues)

streamlit not recognized: ensure the virtual environment is activated where streamlit is installed, or run **python -m streamlit run app.py**.

Permission error removing .git: use PowerShell **Remove-Item -Recurse -Force .git** from the directory where the misplaced .git lives.

Model loading fails due to sklearn version: either reinstall scikit-learn to the version used during save (pip install scikit-learn==1.6.1) or retrain after upgrading.





