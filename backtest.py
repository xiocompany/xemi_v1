import pandas as pd

def calculate_t_direction(data, N, threshold=0.3):
    """
    محاسبه جهت ('Up'، 'Down' یا 'Stable') بر اساس تغییر درصد بین نقطه فعلی و N دوره بعد.
    """
    pct_change = (data['Close'].shift(-N) - data['Close']) / data['Close'] * 100
    direction = pct_change.apply(
        lambda x: 2 if x > threshold else (0 if x < -threshold else 1)
    )
    return direction

def preprocess_data(file_path, N_values, threshold=0.3, keep_last=100):
    """
    بارگذاری داده، محاسبه جهت هدف، انتخاب ویژگی‌ها و نگه داشتن تنها آخرین 'keep_last' ردیف.

    Args:
        file_path (str): مسیر فایل CSV.
        N_values (list): لیستی از مقادیر N برای محاسبه جهت هدف.
        threshold (float): آستانه برای تعیین جهت.
        keep_last (int): تعداد ردیف‌های آخر که باید نگه داشته شوند (اینجا 100).

    Returns:
        tuple: شامل DataFrame پردازش‌شده و لیست ستون‌های ویژگی.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None

    for N in N_values:
        df[f"target_{N}"] = calculate_t_direction(df, N, threshold)

    # انتخاب ویژگی‌ها (حذف ستون 'Datetime' و ستون‌های هدف)
    exclude_cols = ['Datetime']
    target_cols = [f"target_{N}" for N in N_values]
    feature_cols = [c for c in df.columns if c not in exclude_cols and c not in target_cols]

    # نگه داشتن تنها آخرین 'keep_last' ردیف
    if len(df) > keep_last:
        df = df.iloc[-keep_last:]
        print(f"Kept only the last {keep_last} rows.")
    else:
        print("DataFrame has fewer than 'keep_last' rows. Keeping all rows.")

    return df, feature_cols

# مثال استفاده:
if __name__ == "__main__":
    file_path = r"D:\project\BTCbacktest.csv"  # مسیر فایل CSV خود را جایگزین کنید
    N_values = [60]
    threshold = 0.3
    keep_last = 100  # فقط 100 سکانس آخر نگه داشته شود

    df, feature_cols = preprocess_data(file_path, N_values, threshold, keep_last)

    if df is not None:
        print("Data loaded and preprocessed successfully.")
        print("Feature columns:", feature_cols)
        print(df.head())
        print(df.tail())

        # در صورت وجود ستون 'timestamp'، آن را حذف می‌کنیم
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
            print("\n'timestamp' column dropped.")
            print(df.head())
            print(df.tail())
        else:
            print("\n'timestamp' column not found.")

        # ذخیره CSV با تنها 100 سکانس
        df.to_csv("processed_data.csv", index=False)
        print("CSV saved with only 100 sequences.")
    else:
        print("Data loading or preprocessing failed.")
