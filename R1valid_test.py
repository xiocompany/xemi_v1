# Import necessary libraries
import pandas as pd
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define the list of timeframes (N values) to validate
timeframes = [3, 5, 15, 30, 45, 60, 75, 90, 120] 

# Load the validation dataset once
data_file = r"D:\project\models_xgb\xgb_target_{N}_validation_data.csv"

# Loop through each timeframe and validate the corresponding model
for N in timeframes:
    # Construct the specific file path for each N
    current_data_file = data_file.replace("{N}", str(N))  # Replace {N} with the actual value

    try:
        # Load the validation dataset
        data = pd.read_csv(current_data_file)
    except FileNotFoundError:
        print(f"Skipping timeframe {N}: Validation data file '{current_data_file}' not found.")
        continue

    # Define the target column dynamically
    target_col = f'target_{N}'

    if target_col not in data.columns:
        print(f"Skipping timeframe {N}: Target column '{target_col}' not found in dataset.")
        continue

    # Split features (X_val) and target (y_val)
    y_val = data[target_col]  
    X_val = data.drop(columns=[target_col])  

    # Load the trained XGBRegressor model for this timeframe
    model_file = rf"D:\project\models_xgb\xgb_target_{N}.pkl"
    
    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        print(f"Skipping timeframe {N}: Model file '{model_file}' not found.")
        continue

    # Make predictions on the validation features
    y_pred = model.predict(X_val)

    # Calculate regression metrics
    mse = mean_squared_error(y_val, y_pred)
    r2  = r2_score(y_val, y_pred)

    # Print the results for this timeframe in a structured format
    print(f"Timeframe {N}: MSE = {mse:.4f}, R² = {r2:.4f}")

    # Create a scatter plot of Actual vs Predicted values
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_pred, alpha=0.6, label='Data points')

    # Plot a diagonal line representing perfect prediction (Predicted = Actual)
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')

    # Add plot titles and labels
    plt.title(f'Actual vs Predicted (Timeframe {N})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()  # Display the plot for this timeframe
