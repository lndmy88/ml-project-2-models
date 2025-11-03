import json
from pdb import run
import joblib
from pathlib import Path
import mlflow
from sklearn.ensemble import RandomForestClassifier
from src.utils.common_utils import load_processed_data
from src.utils.mlflow_utils import register_model, start_mlflow_run


def train_churn_models(config, params, selected_segments=None):

    #  get config
    model_name = config['artifacts']['churn']['model_name']
    output_dir = config['artifacts']['churn']['model_dir']
    experiment_name = config['mlflow']['experiment_name']
    target_column = config['data']['churn_target_column']
    not_feature_columns = config['data']['not_feature_columns']

    # get params
    model_type = params['churn']['model_type']
    params = params['churn']['params']

    for segment in selected_segments:

        # prepare data paths for the segment
        train_path = f"{config['data']['processed_path']}/{config['data']['train_file_name']}_day_{segment}.parquet"
        test_path = f"{config['data']['processed_path']}/{config['data']['test_file_name']}_day_{segment}.parquet"

        train_df, _ = load_processed_data(train_path, test_path)

        # prepare training data
        X_train = train_df.drop(columns=not_feature_columns)
        y_train = train_df[target_column]

        model = RandomForestClassifier(**params)
        with start_mlflow_run(experiment_name, f"{model_name}_day_{segment}_training") as run:
            model.fit(X_train, y_train)

            mlflow.log_param("model_type", model_type)
            mlflow.log_params(params)

            signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, name=f"{model_name}_day_{segment}", signature=signature, input_example=X_train.head(5))

            # Save temporary trained model (for evaluation step)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            joblib.dump(model, f"{output_dir}/{model_name}_day_{segment}_temp.joblib")

            print(f"✅ Churn model training complete for day {segment}. Ready for evaluation.")

            register_model(run, f"{model_name}_day_{segment}")
            
if __name__ == "__main__":
    train_churn_models()
