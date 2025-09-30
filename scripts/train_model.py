# scripts/train_model.py
from src.data_preprocessing import load_raw_csv, preprocess_and_split
from src.model_training import train_and_save
from src.utils import load_config, get_logger

logger = get_logger("script_train", "logs/training_log.log")

def main():
    cfg = load_config()
    seed = cfg.get("random_seed", 42)
    model_path = cfg.get("models_path", "models") + "/" + cfg.get("model_filename", "fever_model.pkl")

    df = load_raw_csv()
    X_train, X_test, y_train, y_test = preprocess_and_split(df, test_size=cfg.get("test_size", 0.2), seed=seed)

    pipeline = train_and_save(X_train, y_train, X_test, y_test, model_path=model_path, seed=seed, do_hpo=cfg.get("do_hpo", False))
    logger.info("Training script finished.")

if __name__ == "__main__":
    main()
