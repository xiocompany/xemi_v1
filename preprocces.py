import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------
# Load CSV
# -----------------------------------------------------------------------
timeframes = ['1m']  # You can add more if needed

def load_data(timeframe):
    csv_file = r"D:\project\Data_test\BTCUSDT_ohlc_data_1m.csv"
    data = pd.read_csv(csv_file, parse_dates=['timestamp'])
    return data

# -----------------------------------------------------------------------
# Basic Features
# -----------------------------------------------------------------------
def calculate_hl_N(data, N):
    high = data['High'].rolling(window=N, min_periods=1).max()
    low = data['Low'].rolling(window=N, min_periods=1).min()
    return ((high - low) / low) * 100

def calculate_p_N(data, N):
    return ((data['Close'] - data['Close'].shift(N)) / data['Close'].shift(N)) * 100

def calculate_v_N(data, N):
    return data['Volume'].rolling(window=N, min_periods=1).sum()

# -----------------------------------------------------------------------
# Enhanced Feature Engineering with Open and Volume
# -----------------------------------------------------------------------
def calculate_q_n_open_volume(data, N):
    # Example: Q_n as the ratio of Open to Volume over N periods
    open_n = data['Open'].rolling(window=N, min_periods=1).mean()  # Rolling mean of Open
    volume_n = data['Volume'].rolling(window=N, min_periods=1).mean()  # Rolling mean of Volume
    return open_n / volume_n  # Adjust logic as needed

def calculate_x_n_open_volume(data, N):
    # Example: X_n as the product of Open and Volume over N periods
    open_n = data['Open'].rolling(window=N, min_periods=1).mean()
    volume_n = data['Volume'].rolling(window=N, min_periods=1).mean()
    return open_n * volume_n  # Adjust logic as needed

# -----------------------------------------------------------------------
# Main Worker with Q_n and X_n based on Open and Volume
# -----------------------------------------------------------------------
def process_timeframe_with_qx_open_volume(timeframe):
    data = load_data(timeframe)

    # Define periods for your existing basic features
    periods_for_basic_features =  [
        30, 45, 60, 90
    ]

    # Create a dictionary for new columns
    new_columns = {}

    for period in tqdm(periods_for_basic_features, desc=f"Processing {timeframe} - Features with Open and Volume"):
        new_columns[f'p__{period}'] = calculate_p_N(data, period)
        new_columns[f'hl_{period}'] = calculate_hl_N(data, period)
        new_columns[f'v_{period}'] = calculate_v_N(data, period)

        # Add Q_n and X_n based on Open and Volume
        new_columns[f'Q_{period}'] = calculate_q_n_open_volume(data, period)
        new_columns[f'X_{period}'] = calculate_x_n_open_volume(data, period)

    # Combine original data + new basic features
    new_df = pd.DataFrame(new_columns)
    output = pd.concat([data.reset_index(drop=True), new_df], axis=1)

    # Save to CSV
    out_csv = f"BTCUSDT(1year).csv"
    output.to_csv(out_csv, index=False)
    print(f"Data with features and Q_n, X_n saved to {out_csv}")

# -----------------------------------------------------------------------
# Run with Enhanced Features based on Open and Volume
# -----------------------------------------------------------------------
if __name__ == "__main__":
    process_timeframe_with_qx_open_volume("1m")
