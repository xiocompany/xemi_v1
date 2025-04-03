import ccxt
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # added tqdm import

# Initialize the exchange
exchange = ccxt.binance()

# Define the symbol
symbol = 'BTC/USDT'

# Define the timeframes and corresponding lookback periods
timeframes = {
    '1m': 1850,  # Fetch 90 days of data
}

# Function to fetch OHLCV data with tqdm progress bar
def fetch_ohlcv(symbol, timeframe, days, limit=50000):
    try:
        now = datetime.utcnow()
        since_date = now - timedelta(days=days)
        since = int(since_date.timestamp() * 1000)  # Convert to milliseconds

        # Fetch OHLCV data in batches with progress bar
        ohlcv = []
        with tqdm(desc=f"Fetching {timeframe} data", unit="batch") as pbar:
            while True:
                data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not data:
                    break
                ohlcv.extend(data)
                pbar.update(1)
                since = data[-1][0] + 1  # Move to the next batch
                if since >= int(now.timestamp() * 1000):
                    break

        return ohlcv
    except Exception as e:
        print(f"Error fetching data for {timeframe}: {e}")
        return []

# Function to save data to a CSV
def save_to_csv(ohlcv, timeframe):
    if ohlcv:
        # Convert the data to a pandas DataFrame
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')  # Convert timestamp

        # Save the data to a CSV file
        csv_file = f"BTCUSDT_ohlc_data_{timeframe}(3year).csv"
        data.to_csv(csv_file, index=False)
        print(f"Data saved to {csv_file}")
    else:
        print(f"No data fetched for timeframe {timeframe}")

# Fetch and save data for each timeframe in parallel
with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(fetch_ohlcv, symbol, timeframe, days): timeframe
        for timeframe, days in timeframes.items()
    }
    for future in futures:
        timeframe = futures[future]
        ohlcv = future.result()
        save_to_csv(ohlcv, timeframe)