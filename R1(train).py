import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost.core import XGBoostError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

########################
#  1) DEFINE FUNCTIONS
########################

def calculate_t_pct(data, N):
    """
    Calculate the percentage change between the current point and the point N periods ahead.

    Parameters:
        data (pd.DataFrame): DataFrame containing at least the 'Close' column.
        N (int): Number of periods ahead to calculate the percentage change.

    Returns:
        pd.Series: The percentage change between current point and N periods ahead.
    """
    if 'Close' not in data.columns:
        raise ValueError("DataFrame must contain 'Close' column for calculate_t_pct.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")

    return (data['Close'].shift(-N) - data['Close']) / data['Close'] * 100


def train_with_best_params(
    df,
    feature_cols,
    target_col,
    model_name,
    best_params,
    output_dir="models_xgb"
):
    """
    Train an XGBoost model with a specific set of best parameters.
    1) Splits data into train/test
    2) Attempts GPU first (device='cuda').
       Falls back to CPU if GPU is not available / fails.
    3) Evaluates MSE & R^2
    4) Saves scatter plot & model
    """
    # Drop rows with NaN in target or features
    temp_df = df[feature_cols + [target_col]].dropna()

    # Separate features & target
    X = temp_df[feature_cols]
    y = temp_df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Attempt GPU first
    try:
        model = XGBRegressor(
            tree_method='hist',
            device='cuda',       # GPU training with device='cuda'
            random_state=42,
            **best_params
        )
        model.fit(X_train, y_train)
        print(f"Trained [{model_name}] on GPU successfully with provided best hyperparams.")
    except XGBoostError as gpu_error:
        print(f"GPU training failed for [{model_name}] with error:\n{gpu_error}")
        print("Falling back to CPU training...")
        model = XGBRegressor(
            tree_method='hist',
            device='cpu',       # CPU training
            random_state=42,
            **best_params
        )
        model.fit(X_train, y_train)

    # Predict on test set
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"[{model_name}] Final MSE = {mse:.4f}, R^2 = {r2:.4f}")

    # Plot actual vs. predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.title(f"{model_name}\nMSE={mse:.4f}, R^2={r2:.4f}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)

    # Save plot & model
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{model_name}_scatter.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    model_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)

    # Save validation data (features and target)
    val_feature_df = pd.DataFrame(X_test, columns=feature_cols)  # X_test used as validation data
    val_target_df = pd.DataFrame(y_test, columns=[target_col])  # y_test used as validation target
    val_data_path = os.path.join(output_dir, f"{model_name}_validation_data.csv")
    val_feature_target_df = pd.concat([val_feature_df, val_target_df], axis=1)
    val_feature_target_df.to_csv(val_data_path, index=False)

    return mse, r2


########################
#  2) MAIN EXAMPLE
########################

def main():
    # -----------------------------
    # A. Load or create the dataset
    # -----------------------------
    csv_path = r"D:\project\XAUUSD_ohlc_data_1m_features_with_qx_open_volume.csv" # update to your file
    df = pd.read_csv(csv_path)

    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column.")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by='timestamp', inplace=True)

    # -----------------------
    # B. Create multiple targets
    # -----------------------
    N_values = [60] 
    for N in N_values:
        df[f"target_{N}"] = calculate_t_pct(df, N)

    # -----------------------
    # C. Select features
    # -----------------------
    exclude_cols = ['timestamp']
    target_cols = [f"target_{N}" for N in N_values]
    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and c not in target_cols
    ]

    print("Using the following feature columns:")
    print(feature_cols)

    # ---------------------------------------------
    # D. Best parameters for each target
    # ---------------------------------------------
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
    # -----------------------
    # E. Train & Evaluate
    # -----------------------
    results = {}

    for N in N_values:
        target_col = f"target_{N}"
        model_name = f"xgb_target_{N}"

        if N not in BEST_PARAMS:
            print(f"\nNo best params found for target_{N}, skipping...")
            continue

        print(f"\nTraining model for [{model_name}] using known best params...")
        best_params = BEST_PARAMS[N]
        mse, r2 = train_with_best_params(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            model_name=model_name,
            best_params=best_params,
            output_dir="models_xgb"
        )
        results[target_col] = (mse, r2)

    # -----------------------
    # F. Final summary
    # -----------------------
    print("\n--- Final Results ---")
    for tgt, (mse_val, r2_val) in results.items():
        print(f"{tgt}: MSE={mse_val:.4f}, R^2={r2_val:.4f}")


if __name__ == "__main__":
    main()
