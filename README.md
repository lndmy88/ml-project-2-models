# ml-project-2-models
This is to demo a ML project where there are 2 different models sharing the same dataset 

# Business goal

A telecom company wants to:

- Predict which customers are likely to churn (classification problem).

- Estimate each customer’s future lifetime value (LTV) (regression problem).

Both models use the same customer dataset, but train on different target variables.


# DESCRIPTION
- preprocess data (data_preprocessing.py)
    Reads & cleans the raw dataset.
    Encodes & scales features.
    Creates two targets:
        churn_target (binary classification)
        ltv_target (continuous regression).
    Splits into train/test sets (80/20).
    Saves both sets to data/processed/ as .parquet files — ready for model training.

- train model: (pipelines/)
    Load processed train/test data into two pipelines (classification + regression).
    Track experiments with MLflow (local tracking server).
    (No DVC involved yet.)
    Save the trained model object as model.joblib inside a dedicated folder.
    Optionally compress metadata (like metrics, params) alongside it.
    Store each run under a versioned subfolder

