import joblib
import mlflow
from src.utils.common_utils import load_processed_data
from src.utils.mlflow_utils import register_model, start_mlflow_run
from src.utils.model_utils import classification_metrics, save_model_package

def evaluate_churn_models(config, params, selected_segments=None):

    #  get config
    model_name = config['artifacts']['churn']['model_name']
    model_dir = config['artifacts']['churn']['model_dir']
    experiment_name = config['mlflow']['experiment_name']
    target_column = config['data']['churn_target_column']
    not_feature_columns = config['data']['not_feature_columns']

    # get params
    params = params['churn']

    for segment in selected_segments:

        # prepare data paths for the segment
        train_path = f"{config['data']['processed_path']}/{config['data']['train_file_name']}_day_{segment}.parquet"
        test_path = f"{config['data']['processed_path']}/{config['data']['test_file_name']}_day_{segment}.parquet"

        _, test_df = load_processed_data(train_path, test_path)
        model = joblib.load(f"{model_dir}/{model_name}_day_{segment}_temp.joblib")

        X_test = test_df.drop(columns=not_feature_columns)
        y_test = test_df[target_column]

        preds = model.predict(X_test)
        metrics = classification_metrics(y_test, preds)

        with start_mlflow_run(experiment_name, f"{model_name}_day_{segment}_evaluation") as run:
            mlflow.log_metrics(metrics)

            save_model_package(
                model=model,
                metrics=metrics,
                params=params,  # can reload from JSON if needed
                output_dir=model_dir,
                model_name=f"{model_name}_day_{segment}"
            )

            print(f"✅ Churn evaluation done for day {segment}. Accuracy={metrics['accuracy']:.3f}, AUC={metrics['roc_auc']:.3f}")

            #register_model(run, f"{model_name}_day_{segment}")