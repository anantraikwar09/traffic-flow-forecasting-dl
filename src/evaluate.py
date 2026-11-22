import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

MAX_CAPACITY = 7280.0
TIER_NAMES = ["Low (0-35%)", "Moderate (35-65%)", "Heavy (65-85%)", "Severe (>85%)"]

def volume_to_tier(vol: np.ndarray) -> np.ndarray:
    vc = np.clip(vol, 0, None) / MAX_CAPACITY
    tiers = np.zeros_like(vc, dtype=int)
    tiers[(vc >= 0.35) & (vc < 0.65)] = 1
    tiers[(vc >= 0.65) & (vc < 0.85)] = 2
    tiers[vc >= 0.85] = 3
    return tiers

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true >= 100.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def evaluate_all():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    fig_dir = project_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    preds = np.load(models_dir / "test_predictions.npz", allow_pickle=True)
    with open(models_dir / "training_histories.pkl", "rb") as f:
        histories = pickle.load(f)

    y_true = preds["y_true"]
    ts_test = pd.to_datetime(preds["ts_test"])
    true_tiers = volume_to_tier(y_true)

    model_order = [
        "Seasonal_Naive_24h",
        "Persistence_1h",
        "HistGradientBoosting",
        "Vanilla_LSTM",
        "Stacked_BiLSTM",
        "Ablation_No_Weather",
        "Champion_Conv1D_BiLSTM_Attn",
    ]

    rows = []
    for name in model_order:
        y_pred = np.clip(preds[name], 0, None)
        pred_tiers = volume_to_tier(y_pred)
        rows.append({
            "Model": name,
            "MAE (veh/hr)": round(mean_absolute_error(y_true, y_pred), 2),
            "RMSE (veh/hr)": round(root_mean_squared_error(y_true, y_pred), 2),
            "MAPE (%)": round(safe_mape(y_true, y_pred), 2),
            "R2 Score": round(r2_score(y_true, y_pred), 4),
            "Congestion Acc (%)": round(accuracy_score(true_tiers, pred_tiers) * 100, 2),
            "Congestion Macro-F1": round(f1_score(true_tiers, pred_tiers, average="macro"), 4),
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(models_dir / "experiment_results.csv", index=False)
    print("\n=== TEST SET BENCHMARK & ABLATION RESULTS ===")
    print(results_df.to_string(index=False))

    # Regime breakdown for Champion model
    champ_pred = np.clip(preds["Champion_Conv1D_BiLSTM_Attn"], 0, None)
    is_rush = (ts_test.dayofweek < 5) & (ts_test.hour.isin([7, 8, 9, 16, 17, 18]))
    print("\n=== CHAMPION REGIME ERROR ANALYSIS ===")
    print(f"Peak Rush-Hour RMSE: {root_mean_squared_error(y_true[is_rush], champ_pred[is_rush]):.2f} veh/hr")
    print(f"Off-Peak RMSE:       {root_mean_squared_error(y_true[~is_rush], champ_pred[~is_rush]):.2f} veh/hr")

    # Figure 1: Training & Validation Loss Curves
    plt.figure(figsize=(10, 4.5))
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], linewidth=2, label=f"{name} (Val MSE)")
    plt.title("Validation MSE Convergence Across Deep Learning Experiments", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE (Scaled)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "01_training_curves.png", dpi=150)
    plt.close()

    # Figure 2: Model Benchmark Bar Chart (MAE & RMSE)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(results_df))
    w = 0.38
    ax.bar(x - w/2, results_df["MAE (veh/hr)"], width=w, label="MAE (veh/hr)", color="#3498db")
    ax.bar(x + w/2, results_df["RMSE (veh/hr)"], width=w, label="RMSE (veh/hr)", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=22, ha="right")
    ax.set_ylabel("Error (Vehicles / Hour)")
    ax.set_title("Test Set Error Progression & Ablation Study (Lower is Better)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "02_model_benchmark.png", dpi=150)
    plt.close()

    # Figure 3: 7-Day Test Forecast Overlay with Congestion Bands
    plt.figure(figsize=(13, 4.8))
    sl = slice(240, 408)  # 168 consecutive test hours (1 full week)
    t_week = ts_test[sl]
    plt.axhspan(0, 0.35 * MAX_CAPACITY, color="#2ecc71", alpha=0.08, label="Low Congestion")
    plt.axhspan(0.35 * MAX_CAPACITY, 0.65 * MAX_CAPACITY, color="#f1c40f", alpha=0.08, label="Moderate")
    plt.axhspan(0.65 * MAX_CAPACITY, 0.85 * MAX_CAPACITY, color="#e67e22", alpha=0.08, label="Heavy")
    plt.axhspan(0.85 * MAX_CAPACITY, MAX_CAPACITY, color="#e74c3c", alpha=0.10, label="Severe")
    plt.plot(t_week, y_true[sl], color="#111111", linewidth=2, label="Actual Traffic Volume")
    plt.plot(t_week, champ_pred[sl], color="#2980b9", linestyle="--", linewidth=1.8, label="Champion Forecast (+1h)")
    plt.title("7-Day Out-of-Sample Forecast vs. Actual Traffic Volume & Congestion Zones", fontweight="bold")
    plt.ylabel("Traffic Volume (veh/hr)")
    plt.legend(loc="upper right", ncol=3, fontsize=8.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "03_actual_vs_predicted_week.png", dpi=150)
    plt.close()

    # Figure 4: Confusion Matrix & Residual Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm = confusion_matrix(true_tiers, volume_to_tier(champ_pred))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=TIER_NAMES, yticklabels=TIER_NAMES, ax=axes[0])
    axes[0].set_title("Congestion Level Confusion Matrix (Champion)", fontweight="bold")
    axes[0].set_xlabel("Predicted Congestion Tier")
    axes[0].set_ylabel("Actual Congestion Tier")

    residuals = y_true - champ_pred
    axes[1].hist(residuals, bins=60, color="#8e44ad", edgecolor="black", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title(f"Residual Error Distribution (Mean={residuals.mean():.1f}, Std={residuals.std():.1f})", fontweight="bold")
    axes[1].set_xlabel("Prediction Error: Actual - Predicted (veh/hr)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "04_confusion_matrix_residuals.png", dpi=150)
    plt.close()

    print(f"\nSUCCESS -> Saved experiment_results.csv and 4 charts to {fig_dir}")

if __name__ == "__main__":
    evaluate_all()