import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Maximum road traffic capacity
MAX_CAPACITY = 7280.0

# Number of previous hours used as input
LOOKBACK = 24

# Predict traffic volume for the next hour
HORIZON = 1


# ==================================================
# 1. FIND THE DATASET
# ==================================================

def find_csv(project_root: Path) -> Path:

    # Check different possible locations for the dataset
    candidates = [
        project_root / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv",
        project_root / "Metro_Interstate_Traffic_Volume.csv",
        Path.cwd() / "Metro_Interstate_Traffic_Volume.csv",
        Path.home() / "Downloads" / "Metro_Interstate_Traffic_Volume.csv",
    ]

    # Return the dataset path if the file exists
    for path in candidates:
        if path.exists():
            print(f"Found dataset at: {path}")
            return path

    # Raise an error if the dataset cannot be located
    raise FileNotFoundError(
        "Could not find 'Metro_Interstate_Traffic_Volume.csv'. "
        f"Place it in '{project_root / 'data' / 'raw'}' and re-run."
    )


# ==================================================
# 2. DATA CLEANING AND PREPROCESSING
# ==================================================

def load_and_clean(csv_path: Path) -> pd.DataFrame:

    # Load the original traffic dataset
    df = pd.read_csv(csv_path)

    # Convert date_time into datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])

    # HOLIDAY HANDLING:
    # Holiday labels are recorded at midnight for applicable dates.
    # Extract holiday dates and mark every observation on those dates.
    # Dates without a holiday label are treated as non-holiday.
    holiday_dates = set(
        df.loc[df['holiday'].notna(), 'date_time'].dt.date
    )

    df['is_holiday'] = (
        df['date_time'].dt.date.isin(holiday_dates).astype(int)
    )

    # SENSOR ANOMALY HANDLING:
    # Replace unrealistic temperature readings with NaN.
    df.loc[df['temp'] < 200, 'temp'] = np.nan

    # Replace unusually high rainfall readings with NaN.
    df.loc[df['rain_1h'] > 150, 'rain_1h'] = np.nan

    # Sort observations chronologically
    df = df.sort_values('date_time')

    # DUPLICATE TIMESTAMP HANDLING:
    # Combine duplicate timestamps into one hourly observation.
    # Traffic volume: retain the first available observation.
    # Temperature and cloud coverage: calculate the mean.
    # Rainfall and snowfall: retain the maximum value.
    # Weather and holiday information: retain the first/max value.
    df_clean = df.groupby('date_time', as_index=False).agg({
        'traffic_volume': 'first',
        'temp': 'mean',
        'rain_1h': 'max',
        'snow_1h': 'max',
        'clouds_all': 'mean',
        'weather_main': 'first',
        'is_holiday': 'max'
    })

    # MISSING HOURLY OBSERVATIONS:
    # Reindex the dataset to ensure continuous hourly timestamps.
    # Missing hourly observations are introduced as NaN.
    df_clean = df_clean.set_index('date_time').asfreq(
        pd.Timedelta(hours=1)
    )

    # Select numerical weather features for missing-value handling
    exogenous_num_cols = [
        'temp',
        'rain_1h',
        'snow_1h',
        'clouds_all'
    ]

    # Fill short gaps in weather features using forward filling.
    # Maximum of two consecutive missing hours are filled.
    df_clean[exogenous_num_cols] = (
        df_clean[exogenous_num_cols].ffill(limit=2)
    )

    # Carry forward holiday information for short gaps
    df_clean['is_holiday'] = (
        df_clean['is_holiday'].ffill(limit=2)
    )

    # Carry forward weather categories for short gaps
    df_clean['weather_main'] = (
        df_clean['weather_main'].ffill(limit=2)
    )

    # IMPORTANT:
    # Traffic volume is NOT interpolated or forward-filled.
    # This prevents artificial target values and preserves
    # missing traffic observations.

    # Long missing periods remain unchanged.
    return df_clean


# ==================================================
# 3. TEMPORAL FEATURE ENGINEERING
# ==================================================

def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:

    # Create a copy of the cleaned dataset
    df = df_clean.copy()

    # Extract hour, day of week and month from timestamps
    hour = df.index.hour
    dow = df.index.day_of_week
    month = df.index.month

    # CYCLICAL TIME FEATURES:
    # Represent time variables using sine and cosine.
    # This helps the model understand repeating time patterns.

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

    df['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    # WEEKEND FEATURE:
    # Mark Saturday and Sunday as weekends.
    df['is_weekend'] = (dow >= 5).astype(int)

    # RUSH-HOUR FEATURE:
    # Identify morning and evening rush hours on weekdays.
    # Exclude holidays from rush-hour identification.
    df['is_rush_hour'] = (
        (dow < 5) & (df['is_holiday'] == 0) &
        (((hour >= 7) & (hour <= 9)) |
         ((hour >= 16) & (hour <= 18)))
    ).astype(int)

    # VOLUME-TO-CAPACITY RATIO:
    # Calculate traffic volume relative to maximum road capacity.
    df['vc_ratio'] = df['traffic_volume'] / MAX_CAPACITY

    # WEATHER FEATURE ENGINEERING:
    # Group common weather conditions into categories.
    top_weather = [
        'Clear', 'Clouds', 'Mist', 'Rain',
        'Snow', 'Drizzle', 'Haze',
        'Thunderstorm', 'Fog'
    ]

    df['weather_grouped'] = np.where(
        df['weather_main'].isin(top_weather),
        df['weather_main'],
        'Other'
    )

    # Convert categorical weather conditions into numerical columns.
    # This process is called One-Hot Encoding.
    dummies = pd.get_dummies(
        df['weather_grouped'],
        prefix='weather'
    ).astype(int)

    # Combine newly created weather features with the dataset
    df = pd.concat([df, dummies], axis=1)

    # Remove original categorical weather columns
    df = df.drop(
        columns=['weather_main', 'weather_grouped']
    )

    # Keep traffic_volume as the first column.
    # It is the target variable that we want to predict.
    feature_cols = ['traffic_volume'] + [
        c for c in df.columns if c != 'traffic_volume'
    ]

    return df[feature_cols]


# ==================================================
# 4. CREATE SEQUENCES FOR LSTM
# ==================================================

def create_contiguous_sequences(
    scaled_df: pd.DataFrame,
    lookback=LOOKBACK,
    horizon=HORIZON
):

    # Convert the processed DataFrame into a NumPy array
    data = scaled_df.values

    # Initialize lists for inputs, targets and timestamps
    X, y, timestamps = [], [], []

    # Total observations required for one sequence
    total_window = lookback + horizon

    # Generate continuous time-based sequences
    for i in range(len(data) - total_window + 1):

        # Extract 25 hours of data (24 input + 1 target)
        window = data[i : i + total_window]

        # Skip sequences containing missing values.
        # This also excludes sequences with unavailable
        # target traffic volume or missing input features.
        if np.isnan(window).any():
            continue

        # First 24 hours are used as input features
        X.append(window[:lookback])

        # Next hour's traffic volume is the prediction target
        # Traffic volume is stored in column index 0.
        y.append(
            window[lookback : lookback + horizon, 0]
        )

        # Store the timestamp corresponding to the prediction
        timestamps.append(
            str(scaled_df.index[i + lookback])
        )

    # Convert lists into NumPy arrays for LSTM processing
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(timestamps)
    )


# ==================================================
# 5. MAIN PREPROCESSING PIPELINE
# ==================================================

def run_pipeline():

    # Locate the main project directory
    project_root = Path(__file__).resolve().parents[1]

    # Find the original CSV dataset
    csv_path = find_csv(project_root)

    # Create folders for processed data and saved models
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Execute data cleaning
    df_clean = load_and_clean(csv_path)

    # Generate temporal and weather-based features
    df_feat = engineer_features(df_clean)

    # Store feature names for future model predictions
    feature_cols = list(df_feat.columns)

    # ==================================================
    # 6. CHRONOLOGICAL DATA SPLITTING
    # ==================================================

    # Calculate total number of observations
    n = len(df_feat)

    # Split data chronologically:
    # 70% Training, 15% Validation, 15% Testing.
    # Random splitting is avoided to preserve time order.
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[train_end:val_end]
    test_df = df_feat.iloc[val_end:]

    # ==================================================
    # 7. FEATURE SCALING
    # ==================================================

    # Initialize MinMaxScaler to normalize feature values
    scaler = MinMaxScaler()

    # Fit the scaler using only valid training observations.
    # Validation and testing data are not used during fitting.
    # This prevents information leakage from future datasets.
    scaler.fit(train_df.dropna())

    # Apply the same training-fitted scaler to all datasets
    train_scaled = pd.DataFrame(
        scaler.transform(train_df),
        index=train_df.index,
        columns=feature_cols
    )

    val_scaled = pd.DataFrame(
        scaler.transform(val_df),
        index=val_df.index,
        columns=feature_cols
    )

    test_scaled = pd.DataFrame(
        scaler.transform(test_df),
        index=test_df.index,
        columns=feature_cols
    )

    # ==================================================
    # 8. GENERATE TRAINING, VALIDATION AND TEST SEQUENCES
    # ==================================================

    # Create sequences from training data
    X_train, y_train, ts_train = create_contiguous_sequences(
        train_scaled
    )

    # Create sequences from validation data
    X_val, y_val, ts_val = create_contiguous_sequences(
        val_scaled
    )

    # Create sequences from testing data
    X_test, y_test, ts_test = create_contiguous_sequences(
        test_scaled
    )

    # ==================================================
    # 9. SAVE PROCESSED DATA AND SCALER
    # ==================================================

    # Save all generated sequences in a compressed NPZ file
    np.savez_compressed(
        processed_dir / "tensors.npz",

        X_train=X_train,
        y_train=y_train,
        ts_train=ts_train,

        X_val=X_val,
        y_val=y_val,
        ts_val=ts_val,

        X_test=X_test,
        y_test=y_test,
        ts_test=ts_test
    )

    # Save the fitted scaler, feature names and capacity.
    # These will be required during future predictions.
    with open(models_dir / "scaler.pkl", "wb") as f:

        pickle.dump({
            'scaler': scaler,
            'feature_cols': feature_cols,
            'max_capacity': MAX_CAPACITY
        }, f)

    # Display the final shapes of generated datasets
    print(
        f"SUCCESS -> Train: {X_train.shape}, "
        f"Val: {X_val.shape}, Test: {X_test.shape}"
    )


# ==================================================
# 10. EXECUTE THE PIPELINE
# ==================================================

# Start preprocessing when this Python file is executed directly
if __name__ == '__main__':
    run_pipeline()
