import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pathlib import Path

'''
This code will:
    Reads & cleans the raw dataset.
    Encodes & scales features.
    Creates two targets:
        churn_target (binary classification)
        ltv_target (continuous regression).
    Splits into train/test sets (80/20).
    Saves both sets to data/processed/ as .parquet files — ready for model training.
'''


def load_data(folder_path: str, segment) -> pd.DataFrame:
    """
    Load and concatenate all CSV or Parquet files in a given folder.
    
    Args:
        folder_path (str): Path to the directory containing data files.
        segment (str): Segment to filter the data.

    Returns:
        pd.DataFrame: Combined DataFrame of all files.
    """
    folder = Path(folder_path)
    all_files = list(folder.glob("*"))

    if not all_files:
        raise FileNotFoundError(f"No files found in {folder_path}")
    
    # Filter files by selected segments if provided
    if segment:
        all_files = [f for f in all_files if segment in f.name]
        print(f"🔍 Loading selected segment: {segment}")

    dfs = []

    for file in all_files:
        if file.suffix == ".csv":
            df = pd.read_csv(file)
        elif file.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(file)
        else:
            print(f"Skipping unsupported file type: {file.name}")
            continue

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No supported data files found in {folder_path}")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and feature type corrections."""
    df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Convert TotalCharges to numeric and handle missing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Drop ID column
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    print(f"🧹 Cleaned data. Nulls remaining: {df.isnull().sum().sum()}")
    return df


def preprocess_features(df: pd.DataFrame):
    """Preprocess categorical and numeric features."""
    # Separate targets
    y_churn = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.drop(columns=['Churn'])

    # Create synthetic LTV target
    df['ltv_target'] = df['MonthlyCharges'] * df['tenure']

    # Identify feature types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Remove synthetic target from preprocessing
    numeric_features = [col for col in numeric_features if col != 'ltv_target']

    # Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform
    X = preprocessor.fit_transform(df)

    # Get feature names
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + cat_feature_names.tolist()

    # Build DataFrame
    X_processed = pd.DataFrame(X, columns=feature_names)
    X_processed['ltv_target'] = df['ltv_target'].values
    X_processed['churn_target'] = y_churn.values

    print(f"✅ Processed data shape: {X_processed.shape}")
    return X_processed


def split_and_save_data(df: pd.DataFrame, output_dir: str, segment, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test sets and save as parquet."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['churn_target'])

    train_path = Path(output_dir) / f"train_day_{segment}.parquet"
    test_path = Path(output_dir) / f"test_day_{segment}.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"💾 Saved train set to {train_path} ({train_df.shape})")
    print(f"💾 Saved test set to {test_path} ({test_df.shape})")


def process_data(config, selected_segments=None):

    # get paths from config
    raw_path = config['data']['raw_path']
    output_dir = config['data']['processed_path']
    if selected_segments is None:
        segments = config['data']['segments']
    else:
        segments = selected_segments
    for segment in segments:
        df = load_data(raw_path, segment)
        df = clean_data(df)
        processed = preprocess_features(df)
        split_and_save_data(processed, output_dir, segment)

