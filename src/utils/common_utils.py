import pandas as pd
import os
import yaml
import argparse


def load_config(config_path="config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_processed_data(train_path, test_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed train/test files not found. Run preprocessing first.")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df


def parse_app_args():
    parser = argparse.ArgumentParser(description="ML Project Utility")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params file")
    parser.add_argument("--all", action="store_true", help="Use all data")
    parser.add_argument("--selected-segments", "-s", action="append", default=None, help="Segments to process")
    args = parser.parse_args()
    return args

