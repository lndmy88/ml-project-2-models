import joblib
import mlflow
from src.utils.common_utils import load_processed_data
from src.utils.mlflow_utils import register_model, start_mlflow_run
from src.utils.model_utils import regression_metrics, save_model_package

def evaluate_ltv_models(config, params, selected_segments=None):

    #  get config
    model_name = config['artifacts']['ltv']['model_name']
    model_dir = config['artifacts']['ltv']['model_dir']
    experiment_name = config['mlflow']['experiment_name']
    target_column = config['data']['ltv_target_column']
    not_feature_columns = config['data']['not_feature_columns']

    # get params
    params = params['ltv']

    for segment in selected_segments:

        # prepare data paths for the segment
        train_path = f"{config['data']['processed_path']}/{config['data']['train_file_name']}_day_{segment}.parquet"
        test_path = f"{config['data']['processed_path']}/{config['data']['test_file_name']}_day_{segment}.parquet"
    
        _, test_df = load_processed_data(train_path, test_path)
        model = joblib.load(f"{model_dir}/{model_name}_day_{segment}_temp.joblib")

        X_test = test_df.drop(columns=not_feature_columns)
        y_test = test_df[target_column]

        preds = model.predict(X_test)
        metrics = regression_metrics(y_test, preds)

        with start_mlflow_run(experiment_name, f"{model_name}_day_{segment}_evaluation") as run:
            mlflow.log_metrics(metrics)

            save_model_package(
                model=model,
                metrics=metrics,
                params=params,
                output_dir=model_dir,
                model_name=f"{model_name}_day_{segment}"
            )

            print(f"✅ LTV evaluation done for day {segment}. RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")

            #register_model(run, f"{model_name}_day_{segment}")