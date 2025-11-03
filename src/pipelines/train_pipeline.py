from logging import config
from src.models.churn.train_model import train_churn_models
from src.models.churn.evaluate_model import evaluate_churn_models
from src.models.ltv.train_model import train_ltv_models
from src.models.ltv.evaluate_model import evaluate_ltv_models
from src.process_data import process_data

def train_pipeline_definition(config, params, selected_segments=None):
    process_data(config, selected_segments)

    # Churn
    train_churn_models(config, params, selected_segments)
    evaluate_churn_models(config, params, selected_segments)

    # LTV
    train_ltv_models(config, params, selected_segments)
    evaluate_ltv_models(config, params, selected_segments)
