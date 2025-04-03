import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

########################
#  1) DEFINE FUNCTIONS
########################

def calculate_t_direction(data, N):
    """
    Calculate the direction ('Up' or 'Down') based on the percentage change between the current point and N periods ahead.
    Parameters:
        data (pd.DataFrame): DataFrame containing at least the 'Close' column.
        N (int): Number of periods ahead to calculate the direction.
    Returns:
        pd.Series: The direction as 0 (Down) or 1 (Up).
    """
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain 'Close' column for calculate_t_direction.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    pct_change = (data['Close'].shift(-N) - data['Close']) / data['Close'] * 100
    direction = pct_change.apply(lambda x: 1 if x > 0 else 0)
    return direction


def train_with_best_params_classification(
    X_train, y_train, X_val, y_val, X_test, y_test,
    feature_cols,
    target_col,
    model_name,
    best_params,
    output_dir="models_xgb_classification"
):
    """
    Train an XGBoost classifier with specific best parameters. It also saves the model, and validation data.
    """
    num_classes = len(np.unique(y_train))
    assert num_classes == 2, "Number of classes must be 2 for binary classification."

    print(f"[{model_name}] Training model...")

    try:
        final_model = XGBClassifier(
            tree_method='hist',
            device='cuda',
            use_label_encoder=False,
            eval_metric='logloss',
            objective='binary:logistic',  # Use binary classification objective
            random_state=42,
            **best_params
        )
        final_model.fit(X_train, y_train, verbose=True)
        print(f"Trained [{model_name}] on GPU successfully.")
    except XGBoostError as gpu_error:
        print(f"GPU training failed for [{model_name}] with error:\n{gpu_error}")
        print("Falling back to CPU training...")
        final_model = XGBClassifier(
            tree_method='hist',
            device='cpu',
            use_label_encoder=False,
            eval_metric='logloss',
            objective='binary:logistic',  # Use binary classification objective
            random_state=42,
            **best_params
        )
        final_model.fit(X_train, y_train, verbose=True)

    # Predict on train, validation, and test sets
    train_preds = final_model.predict(X_train)
    val_preds = final_model.predict(X_val)
    test_preds = final_model.predict(X_test)

    # Ensure predictions are of integer type (0 or 1 for binary classification)
    train_preds = train_preds.astype(int)
    val_preds = val_preds.astype(int)
    test_preds = test_preds.astype(int)

    # Make sure y_train, y_val, y_test are also integers (binary labels)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    # Calculate metrics for each dataset
    train_accuracy = accuracy_score(y_train, train_preds)
    val_accuracy = accuracy_score(y_val, val_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    train_precision = precision_score(y_train, train_preds, average='binary', zero_division=0)
    val_precision = precision_score(y_val, val_preds, average='binary', zero_division=0)
    test_precision = precision_score(y_test, test_preds, average='binary', zero_division=0)

    train_recall = recall_score(y_train, train_preds, average='binary', zero_division=0)
    val_recall = recall_score(y_val, val_preds, average='binary', zero_division=0)
    test_recall = recall_score(y_test, test_preds, average='binary', zero_division=0)

    train_f1 = f1_score(y_train, train_preds, average='binary', zero_division=0)
    val_f1 = f1_score(y_val, val_preds, average='binary', zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average='binary', zero_division=0)

    # Print results for train, validation, and test
    print(f"[{model_name}] Metrics:")
    print(f"Train Accuracy = {train_accuracy:.4f}, Train Precision = {train_precision:.4f}, Train Recall = {train_recall:.4f}, Train F1-Score = {train_f1:.4f}")
    print(f"Validation Accuracy = {val_accuracy:.4f}, Validation Precision = {val_precision:.4f}, Validation Recall = {val_recall:.4f}, Validation F1-Score = {val_f1:.4f}")
    print(f"Test Accuracy = {test_accuracy:.4f}, Test Precision = {test_precision:.4f}, Test Recall = {test_recall:.4f}, Test F1-Score = {test_f1:.4f}")

    # Save model and validation data
    os.makedirs(output_dir, exist_ok=True)

    # Save trained model
    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(final_model, model_path)

    # Save validation data (features and target)
    val_feature_df = pd.DataFrame(X_val, columns=feature_cols)
    val_target_df = pd.DataFrame(y_val, columns=[target_col])
    val_data_path = os.path.join(output_dir, f"{model_name}_validation_data.csv")
    val_feature_target_df = pd.concat([val_feature_df, val_target_df], axis=1)
    val_feature_target_df.to_csv(val_data_path, index=False)

    # Confusion matrix for test data
    cm = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Confusion Matrix (Test Set)")
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Return metrics for all sets
    return {
        'train': (train_accuracy, train_precision, train_recall, train_f1),
        'validation': (val_accuracy, val_precision, val_recall, val_f1),
        'test': (test_accuracy, test_precision, test_recall, test_f1)
    }

########################
#  2) MAIN FUNCTION
########################

def main():
    # Load data
    csv_path = "BTCUSDT_ohlc_data_1m_features_with_qx_open_volume.csv"
    df = pd.read_csv(csv_path)

    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by='timestamp', inplace=True)

    # Generate target column
    N_values = [3, 5, 15, 30, 45, 60, 75, 90, 120] 
    for N in N_values:
        df[f"target_{N}"] = calculate_t_direction(df, N)

    # Select features
    exclude_cols = ['timestamp']
    target_cols = [f"target_{N}" for N in N_values]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and c not in target_cols
    ]

    # Define best params for the model
    BEST_PARAMS = BEST_PARAMS = {
        3: {
            'n_estimators': 1411,
            'learning_rate': 0.097,
            'max_depth': 12,
            'subsample': 0.635,
            'colsample_bytree': 0.505,
            'gamma': 0.00137,
            'reg_alpha': 0.0000921,
            'reg_lambda': 37.436,
        },
        5: {
            'n_estimators': 1184,
            'learning_rate': 0.0319,
            'max_depth': 10,
            'subsample': 0.902,
            'colsample_bytree': 0.501,
            'gamma': 3.762e-07,
            'reg_alpha': 0.0390,
            'reg_lambda': 1.670,
        },
        15: {
            'n_estimators': 1604,
            'learning_rate': 0.0251,
            'max_depth': 12,
            'subsample': 0.821,
            'colsample_bytree': 0.785,
            'gamma': 2.545e-05,
            'reg_alpha': 0.00180,
            'reg_lambda': 3.109,
        },
        30: {
            'n_estimators': 1228,
            'learning_rate': 0.0275,
            'max_depth': 9,  
            'subsample': 0.785,
            'colsample_bytree': 0.893,
            'gamma': 5.198e-08,
            'reg_alpha': 2.942e-06,
            'reg_lambda': 1.409,
        },
        45: {
            'n_estimators': 1790,
            'learning_rate': 0.0380,
            'max_depth': 11,
            'subsample': 0.753,
            'colsample_bytree': 0.559,
            'gamma': 6.318e-08,
            'reg_alpha': 0.000151,
            'reg_lambda': 4.400e-07,
        },
        60: {
            'n_estimators': 1964,
            'learning_rate': 0.0329,
            'max_depth': 12,
            'subsample': 0.649,
            'colsample_bytree': 0.987,
            'gamma': 0.003126,
            'reg_alpha': 0.0433,
            'reg_lambda': 53.698,
        },
        75: {
            'n_estimators': 1807,
            'learning_rate': 0.0150,
            'max_depth': 11,
            'subsample': 0.934,
            'colsample_bytree': 0.679,
            'gamma': 3.853e-06,
            'reg_alpha': 9.932e-07,
            'reg_lambda': 0.00110,
        },
        90: {
            'n_estimators': 1164,
            'learning_rate': 0.0501,
            'max_depth': 11,
            'subsample': 0.828,
            'colsample_bytree': 0.814,
            'gamma': 0.000347,
            'reg_alpha': 4.345e-07,
            'reg_lambda': 0.01549,
        },
        120: {
            'n_estimators': 966,
            'learning_rate': 0.0289,
            'max_depth': 10,
            'subsample': 0.861,
            'colsample_bytree': 0.721,
            'gamma': 1.522e-06,
            'reg_alpha': 0.04797,
            'reg_lambda': 0.2767,
        },
    }

    # Train and evaluate model for each target
    results = {}
    for N in N_values:
        target_col = f"target_{N}"
        model_name = f"xgb_classifier_target_{N}"

        if N not in BEST_PARAMS:
            print(f"\nNo best params found for {model_name}, skipping...")
            continue

        print(f"\nTraining model for [{model_name}] using provided best params...")
        best_params = BEST_PARAMS[N]

        # Prepare data for current target
        temp_df = df[feature_cols + [target_col]].dropna()
        X = temp_df[feature_cols]
        y = temp_df[target_col]

        # Split data into train, validation (test), and final test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # Train the model and get evaluation metrics for train, validation, and test sets
        metrics = train_with_best_params_classification(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
            feature_cols=feature_cols,
            target_col=target_col,
            model_name=model_name,
            best_params=best_params,
            output_dir="models_xgb_classification"
        )

        results[target_col] = metrics

    # Final summary of the results
    print("\n--- Final Results ---")
    for tgt, metrics in results.items():
        print(f"\n{tgt}:")
        print(f"  Train Accuracy = {metrics['train'][0]:.4f}, Train Precision = {metrics['train'][1]:.4f}, Train Recall = {metrics['train'][2]:.4f}, Train F1-Score = {metrics['train'][3]:.4f}")
        print(f"  Validation Accuracy = {metrics['validation'][0]:.4f}, Validation Precision = {metrics['validation'][1]:.4f}, Validation Recall = {metrics['validation'][2]:.4f}, Validation F1-Score = {metrics['validation'][3]:.4f}")
        print(f"  Test Accuracy = {metrics['test'][0]:.4f}, Test Precision = {metrics['test'][1]:.4f}, Test Recall = {metrics['test'][2]:.4f}, Test F1-Score = {metrics['test'][3]:.4f}")


if __name__ == "__main__":
    main()
