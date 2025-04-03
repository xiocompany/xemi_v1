import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def load_model_and_data(model_path, data_path):
    """
    Load the model and the validation data from the given paths.
    """
    # Load the model
    model = joblib.load(model_path)
    # Load validation data
    data = pd.read_csv(data_path)
    return model, data

def calculate_t_direction(data, N):
    """
    Calculate the direction ('Up' or 'Down') based on the percentage change between the current point and N periods ahead.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least the 'Close' column.
        N (int): Number of periods ahead to calculate the direction.

    Returns:
        pd.Series: The direction as 'Up' or 'Down'.
    """
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain 'Close' column for calculate_t_direction.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    pct_change = (data['Close'].shift(-N) - data['Close']) / data['Close'] * 100
    direction = pct_change.apply(lambda x: 1 if x > 0 else 0)
    return direction

def dataset(data, target_column):
    """
    Prepares the feature set (X) and target set (y) for evaluation.
    """
    # Ensure the target column is calculated and match the target labels from the original data
    y = data[target_column]
    X = data.drop(columns=[target_column])
    return X, y

def validate_for_timeframes(timeframes, model_base_path, data_base_path):
    """
    Validate model performance for multiple timeframes.
    """
    for N in timeframes:
        print(f"\nValidating model for target_{N}...")

        # Load model and data for the current timeframe
        model_path = os.path.join(model_base_path, f"xgb_classifier_target_{N}.pkl")
        data_path = os.path.join(data_base_path, f"xgb_classifier_target_{N}_validation_data.csv")

        model, df = load_model_and_data(model_path, data_path)

        # Prepare the validation dataset
        target_column = f'target_{N}'
        X, y = dataset(df, target_column)
        print(y[-1800:0])
        # Ensure that the target labels are encoded similarly to the training set
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)  # This assumes that the labels in `y` are categorical strings

        # Make predictions
        pred = model.predict(X)

        # Calculate accuracy and other metrics
        accuracy = accuracy_score(y_encoded, pred)
        precision = precision_score(y_encoded, pred, average='weighted', zero_division=0)
        recall = recall_score(y_encoded, pred, average='weighted', zero_division=0)
        f1 = f1_score(y_encoded, pred, average='weighted', zero_division=0)

        # Print results for current timeframe
        print(f"Results for target_{N}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")

# Define the timeframes you want to validate
timeframes = [3, 5, 15, 30, 45, 60, 75, 90, 120]   # Adjust the timeframes as needed

# Base paths for models and validation data
model_base_path = "models_xgb_classification"
data_base_path = "models_xgb_classification"

# Validate for each timeframe
validate_for_timeframes(timeframes, model_base_path, data_base_path)
