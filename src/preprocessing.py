import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

MAX_CAPACITY = 7280.0
LOOKBACK = 24
HORIZON = 1

def find_csv(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv",
        project_root / "Metro_Interstate_Traffic_Volume.csv",
        Path.cwd() / "Metro_Interstate_Traffic_Volume.csv",
        Path.home() / "Downloads" / "Metro_Interstate_Traffic_Volume.csv",
    ]
    for path in candidates:
        if path.exists():
            print(f"Found dataset at: {path}")
            return path
    raise FileNotFoundError(
        "Could not find 'Metro_Interstate_Traffic_Volume.csv'. "
        f"Place it in '{project_root / 'data' / 'raw'}' and re-run."
    )

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date_time'] = pd.to_datetime(df['date_time'])

    holiday_dates = set(df.loc[df['holiday'].notna(), 'date_time'].dt.date)
    df['is_holiday'] = df['date_time'].dt.date.isin(holiday_dates).astype(int)

    df.loc[df['temp'] < 200, 'temp'] = np.nan
    df.loc[df['rain_1h'] > 150, 'rain_1h'] = np.nan

    df = df.sort_values('date_time')
    df_clean = df.groupby('date_time', as_index=False).agg({
        'traffic_volume': 'first',
        'temp': 'mean',
        'rain_1h': 'max',
        'snow_1h': 'max',
        'clouds_all': 'mean',
        'weather_main': 'first',
        'is_holiday': 'max'
    })

    df_clean = df_clean.set_index('date_time').asfreq(pd.Timedelta(hours=1))
    num_cols = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all']
    df_clean[num_cols] = df_clean[num_cols].interpolate(method='linear', limit=2)
    df_clean['is_holiday'] = df_clean['is_holiday'].ffill(limit=2)
    df_clean['weather_main'] = df_clean['weather_main'].ffill(limit=2)
    return df_clean

def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    df = df_clean.copy()
    hour = df.index.hour
    dow = df.index.dayofweek
    month = df.index.month

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    df['is_weekend'] = (dow >= 5).astype(int)
    df['is_rush_hour'] = (
        (dow < 5) & (df['is_holiday'] == 0) &
        (((hour >= 7) & (hour <= 9)) | ((hour >= 16) & (hour <= 18)))
    ).astype(int)
    df['vc_ratio'] = df['traffic_volume'] / MAX_CAPACITY

    top_weather = ['Clear', 'Clouds', 'Mist', 'Rain', 'Snow', 'Drizzle', 'Haze', 'Thunderstorm', 'Fog']
    df['weather_grouped'] = np.where(df['weather_main'].isin(top_weather), df['weather_main'], 'Other')
    dummies = pd.get_dummies(df['weather_grouped'], prefix='weather').astype(int)
    df = pd.concat([df, dummies], axis=1)

    df = df.drop(columns=['weather_main', 'weather_grouped'])
    feature_cols = ['traffic_volume'] + [c for c in df.columns if c != 'traffic_volume']
    return df[feature_cols]

def create_contiguous_sequences(scaled_df: pd.DataFrame, lookback=LOOKBACK, horizon=HORIZON):
    data = scaled_df.values
    X, y, timestamps = [], [], []
    total_window = lookback + horizon
    for i in range(len(data) - total_window + 1):
        window = data[i : i + total_window]
        if np.isnan(window).any():
            continue
        X.append(window[:lookback])
        y.append(window[lookback : lookback + horizon, 0])
        timestamps.append(str(scaled_df.index[i + lookback]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(timestamps)

def run_pipeline():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = find_csv(project_root)

    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df_clean = load_and_clean(csv_path)
    df_feat = engineer_features(df_clean)
    feature_cols = list(df_feat.columns)

    n = len(df_feat)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[train_end:val_end]
    test_df = df_feat.iloc[val_end:]

    scaler = MinMaxScaler()
    scaler.fit(train_df.dropna())

    train_scaled = pd.DataFrame(scaler.transform(train_df), index=train_df.index, columns=feature_cols)
    val_scaled = pd.DataFrame(scaler.transform(val_df), index=val_df.index, columns=feature_cols)
    test_scaled = pd.DataFrame(scaler.transform(test_df), index=test_df.index, columns=feature_cols)

    X_train, y_train, ts_train = create_contiguous_sequences(train_scaled)
    X_val, y_val, ts_val = create_contiguous_sequences(val_scaled)
    X_test, y_test, ts_test = create_contiguous_sequences(test_scaled)

    np.savez_compressed(
        processed_dir / "tensors.npz",
        X_train=X_train, y_train=y_train, ts_train=ts_train,
        X_val=X_val, y_val=y_val, ts_val=ts_val,
        X_test=X_test, y_test=y_test, ts_test=ts_test
    )
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump({'scaler': scaler, 'feature_cols': feature_cols, 'max_capacity': MAX_CAPACITY}, f)

    print(f"SUCCESS -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == '__main__':
    run_pipeline()