import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from models import build_vanilla_lstm, build_bilstm, build_champion_model
from sklearn.metrics import root_mean_squared_error


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def train_all():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "processed" / "tensors.npz"
    models_dir = project_root / "models"

    data = np.load(data_path, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    ts_test = data["ts_test"]

    with open(models_dir / "scaler.pkl", "rb") as f:
        meta = pickle.load(f)
    scaler = meta["scaler"]
    feature_cols = meta["feature_cols"]
    vol_min = scaler.data_min_[0]
    vol_range = scaler.data_max_[0] - scaler.data_min_[0]

    def inv(y_scaled):
        return y_scaled.ravel() * vol_range + vol_min

    predictions = {"y_true": inv(y_test)}
    val_predictions = {"y_true": inv(y_val)}
    val_predictions["Seasonal_Naive_24h"] = inv(X_val[:, 0, 0])
    val_predictions["Persistence_1h"] = inv(X_val[:, -1, 0])

    # 1. Naive & Persistence Baselines
    predictions["Seasonal_Naive_24h"] = inv(X_test[:, 0, 0])
    predictions["Persistence_1h"] = inv(X_test[:, -1, 0])

    # 2. Tabular ML Baseline (HistGradientBoosting)
    print("Training HistGradientBoosting baseline...")
    hgb = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_depth=6, random_state=SEED)
    
    hgb.fit(X_train.reshape(len(X_train), -1), y_train.ravel())
    with open(models_dir / "HistGradientBoosting.pkl", "wb") as f:
        pickle.dump(hgb, f)

    print("Saved HistGradientBoosting model successfully.")
    val_predictions["HistGradientBoosting"] = inv(
        hgb.predict(X_val.reshape(len(X_val), -1))
    )
    predictions["HistGradientBoosting"] = inv(
        hgb.predict(X_test.reshape(len(X_test), -1))
    )

    # 3. Deep Learning Models
    input_shape = (X_train.shape[1], X_train.shape[2])
    dl_builders = {
        "Vanilla_LSTM": build_vanilla_lstm(input_shape),
        "Stacked_BiLSTM": build_bilstm(input_shape),
        "Champion_Conv1D_BiLSTM_Attn": build_champion_model(input_shape, name="Champion_Conv1D_BiLSTM_Attn"),
    }

    histories = {}
    for name, model in dl_builders.items():
        print(f"\n--- Training {name} ---")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        ]
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=45,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        histories[name] = hist.history
        model.save(models_dir / f"{name}.keras")
        predictions[name] = inv(
            model.predict(X_test, verbose=0)
        )
        val_predictions[name] = inv(
            model.predict(X_val, verbose=0)
        )
    

    # 4. Controlled Ablation Study: Mask out Weather & Holiday channels
    print("\n--- Training Ablation Model (Traffic + Time Only, No Weather/Holiday) ---")
    weather_holiday_idx = [
        i for i, col in enumerate(feature_cols)
        if col in ["temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday"] or col.startswith("weather_")
    ]
    X_train_abl = X_train.copy()
    X_val_abl = X_val.copy()
    X_test_abl = X_test.copy()
    X_train_abl[:, :, weather_holiday_idx] = 0.0
    X_val_abl[:, :, weather_holiday_idx] = 0.0
    X_test_abl[:, :, weather_holiday_idx] = 0.0

    abl_model = build_champion_model(input_shape, name="Ablation_No_Weather")
    abl_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    abl_hist = abl_model.fit(
        X_train_abl, y_train,
        validation_data=(X_val_abl, y_val),
        epochs=45,
        batch_size=64,
        callbacks=abl_callbacks,
        verbose=1
    )
    histories["Ablation_No_Weather"] = abl_hist.history
    abl_model.save(models_dir / "Ablation_No_Weather.keras")
    predictions["Ablation_No_Weather"] = inv(abl_model.predict(X_test_abl, verbose=0))
    val_predictions["Ablation_No_Weather"] = inv(
    abl_model.predict(X_val_abl, verbose=0)
    )

    validation_results = {}

    for name, y_val_pred in val_predictions.items():
        if name == "y_true":
            continue

        validation_results[name] = root_mean_squared_error(
            val_predictions["y_true"],
            y_val_pred
        )

    best_model = min(validation_results, key=validation_results.get)

    print("\n=== VALIDATION MODEL SELECTION ===")
    for name, rmse in sorted(
        validation_results.items(),
        key=lambda item: item[1]
    ):
        print(f"{name}: RMSE = {rmse:.2f} veh/hr")

    print(f"\nSelected champion: {best_model}")

    with open(models_dir / "selected_model.txt", "w") as f:
        f.write(best_model)

    np.savez_compressed(models_dir / "test_predictions.npz", ts_test=ts_test, **predictions)
    with open(models_dir / "training_histories.pkl", "wb") as f:
        pickle.dump(histories, f)

    print("\nSUCCESS -> All models trained and saved to models/!")

if __name__ == "__main__":
    train_all()