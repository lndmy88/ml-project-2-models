import os
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import numpy as np
from datetime import datetime
from pathlib import Path
import joblib
import json


def load_params(params_path="params.yaml"):
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Params file not found at {params_path}")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred)
    }

def regression_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }

def save_model_package(model, metrics, params, output_dir, model_name):
    """
    Saves model, metrics, and parameters to a timestamped versioned folder.
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    version_dir = Path(output_dir) / f"{model_name}_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, version_dir / f"{model_name}.joblib")

    # Save metrics & params
    with open(version_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(version_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"📦 Model package exported to: {output_dir}")
    return version_dir