# main.py
import argparse
from src.utils import load_config, get_logger, ensure_dirs
from scripts.train_model import main as train_main
from scripts.evaluate_model import main as eval_main

logger = get_logger("main", "logs/pipeline.log")

def run_all():
    logger.info("Starting full pipeline run")
    print("1) Training model...")
    train_main()
    print("2) Evaluating model...")
    eval_main()
    print("Pipeline finished. Reports in /reports and model in /models")
    logger.info("Pipeline finished successfully")

if __name__ == "__main__":
    ensure_dirs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["all","train","eval"], default="all")
    args = parser.parse_args()
    if args.run == "all":
        run_all()
    elif args.run == "train":
        train_main()
    elif args.run == "eval":
        eval_main()
