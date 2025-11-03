from datetime import datetime
from unittest import result
import mlflow

def start_mlflow_run(experiment_name, run_name):
    mlflow.set_tracking_uri("file:./mlruns")  # local tracking folder
    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return mlflow.start_run(run_name=f"{run_name}_{timestamp}")


def register_model(run, model_name):
    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/{model_name}",
        name=model_name
    )

    print(f"Registered model: {result.name}, version: {result.version}")