import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="TrafficDL")
class TemporalAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * alpha, axis=1)

    def get_config(self):
        return super().get_config()


st.set_page_config(page_title="I-94 Deep Learning Traffic Forecaster", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tensors.npz"


@st.cache_resource
def load_assets():
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        meta = pickle.load(f)

    custom_objs = {"TemporalAttention": TemporalAttention}
    best_core_model = tf.keras.models.load_model(
        MODELS_DIR / "Ablation_No_Weather.keras", custom_objects=custom_objs
    )
    multimodal_model = tf.keras.models.load_model(
        MODELS_DIR / "Champion_Conv1D_BiLSTM_Attn.keras", custom_objects=custom_objs
    )

    data = np.load(DATA_PATH, allow_pickle=True)
    results_df = pd.read_csv(MODELS_DIR / "experiment_results.csv")
    return meta, best_core_model, multimodal_model, data, results_df


def classify_congestion(vol: float, max_cap: float):
    vc = max(vol, 0.0) / max_cap
    if vc < 0.35:
        return "LOW (Free-Flow)", "#2ecc71", vc * 100
    elif vc < 0.65:
        return "MODERATE", "#f1c40f", vc * 100
    elif vc < 0.85:
        return "HEAVY CONGESTION", "#e67e22", vc * 100
    else:
        return "SEVERE JAM ALERT", "#e74c3c", vc * 100


def forecast_multistep(
    window_scaled, start_ts, horizon, model, scaler, feature_cols,
    max_cap, mask_weather, weather_override, temp_c, rain_mm
):
    vol_min = scaler.data_min_[0]
    vol_range = scaler.data_max_[0] - scaler.data_min_[0]
    weather_holiday_idx = [
        i for i, col in enumerate(feature_cols)
        if col in ["temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday"] or col.startswith("weather_")
    ]

    curr_window = window_scaled.copy()
    preds_vol = []
    pred_times = []

    for step in range(horizon):
        target_ts = start_ts + pd.Timedelta(hours=step)
        x_in = curr_window.copy()

        if not mask_weather:
            unscaled_seq = scaler.inverse_transform(x_in)
            unscaled_seq[-1, feature_cols.index("temp")] = temp_c + 273.15
            unscaled_seq[-1, feature_cols.index("rain_1h")] = rain_mm
            for col in feature_cols:
                if col.startswith("weather_"):
                    unscaled_seq[-1, feature_cols.index(col)] = (
                        1.0 if col == f"weather_{weather_override}" else 0.0
                    )
            x_in = scaler.transform(unscaled_seq)
        else:
            x_in[:, weather_holiday_idx] = 0.0

        y_hat_scaled = float(model.predict(x_in[np.newaxis, :, :], verbose=0)[0, 0])
        y_hat_vol = float(np.clip(y_hat_scaled * vol_range + vol_min, 0.0, max_cap * 1.05))
        preds_vol.append(y_hat_vol)
        pred_times.append(target_ts)

        last_unscaled = scaler.inverse_transform(curr_window[-1:])[0]
        next_unscaled = last_unscaled.copy()
        next_unscaled[0] = y_hat_vol
        next_unscaled[feature_cols.index("vc_ratio")] = y_hat_vol / max_cap
        next_unscaled[feature_cols.index("hour_sin")] = np.sin(2 * np.pi * target_ts.hour / 24.0)
        next_unscaled[feature_cols.index("hour_cos")] = np.cos(2 * np.pi * target_ts.hour / 24.0)
        next_unscaled[feature_cols.index("dow_sin")] = np.sin(2 * np.pi * target_ts.dayofweek / 7.0)
        next_unscaled[feature_cols.index("dow_cos")] = np.cos(2 * np.pi * target_ts.dayofweek / 7.0)
        next_unscaled[feature_cols.index("month_sin")] = np.sin(2 * np.pi * target_ts.month / 12.0)
        next_unscaled[feature_cols.index("month_cos")] = np.cos(2 * np.pi * target_ts.month / 12.0)

        is_wknd = float(target_ts.dayofweek >= 5)
        is_hol = next_unscaled[feature_cols.index("is_holiday")]
        next_unscaled[feature_cols.index("is_weekend")] = is_wknd
        next_unscaled[feature_cols.index("is_rush_hour")] = float(
            (target_ts.dayofweek < 5)
            and (is_hol == 0)
            and ((7 <= target_ts.hour <= 9) or (16 <= target_ts.hour <= 18))
        )
        next_scaled = scaler.transform(next_unscaled.reshape(1, -1))[0]
        curr_window = np.vstack([curr_window[1:], next_scaled])

    return pred_times, preds_vol


meta, best_core_model, multimodal_model, data, results_df = load_assets()
scaler = meta["scaler"]
feature_cols = meta["feature_cols"]
max_cap = meta["max_capacity"]
X_test, y_test, ts_test = data["X_test"], data["y_test"], pd.to_datetime(data["ts_test"])

st.title("Spatio-Temporal Traffic Flow & Congestion Forecasting System")
st.caption("Interstate 94 Westbound (ATR Station 301) | Deep Learning Conv1D-BiLSTM-Attention Architecture")

st.sidebar.header("Simulation Controls")
model_choice = st.sidebar.radio(
    "Select Active Model",
    [
        "Core Conv1D-BiLSTM-Attn (Best: RMSE 225.01)",
        "Multimodal Conv1D-BiLSTM-Attn (+Weather: RMSE 253.58)",
    ],
)
use_core = model_choice.startswith("Core")
active_model = best_core_model if use_core else multimodal_model

sample_idx = st.sidebar.slider(
    "Test Set Window Index (2017-2018)", min_value=24, max_value=len(X_test) - 4, value=320
)
horizon = st.sidebar.selectbox(
    "Forecast Horizon", [1, 2, 3], index=2, format_func=lambda h: f"Next {h} Hour(s)"
)
surge_pct = st.sidebar.slider(
    "What-If Traffic Surge / Drop (%)", min_value=-30, max_value=30, value=0, step=5
)

st.sidebar.subheader("Weather Stress-Test (Multimodal Mode)")
weather_override = st.sidebar.selectbox(
    "Weather Condition", ["Clear", "Clouds", "Rain", "Snow", "Thunderstorm", "Fog"]
)
temp_c = st.sidebar.slider("Temperature (C)", min_value=-20, max_value=38, value=15)
rain_mm = st.sidebar.slider("Rainfall (mm/hr)", min_value=0.0, max_value=40.0, value=0.0, step=1.0)

window_scaled = X_test[sample_idx].copy()
if surge_pct != 0:
    unscaled_w = scaler.inverse_transform(window_scaled)
    unscaled_w[:, 0] = np.clip(unscaled_w[:, 0] * (1.0 + surge_pct / 100.0), 0, max_cap)
    unscaled_w[:, feature_cols.index("vc_ratio")] = unscaled_w[:, 0] / max_cap
    window_scaled = scaler.transform(unscaled_w)

target_ts = ts_test[sample_idx]
hist_unscaled = scaler.inverse_transform(window_scaled)
hist_vols = hist_unscaled[:, 0]
hist_times = [target_ts - pd.Timedelta(hours=24 - i) for i in range(24)]

pred_times, preds_vol = forecast_multistep(
    window_scaled, target_ts, horizon, active_model, scaler, feature_cols,
    max_cap, use_core, weather_override, temp_c, rain_mm
)

next_vol = preds_vol[0]
curr_vol = hist_vols[-1]
delta_vol = next_vol - curr_vol
vol_range = scaler.data_max_[0] - scaler.data_min_[0]
actual_t1 = float(y_test[sample_idx, 0] * vol_range + scaler.data_min_[0])
tier_label, tier_color, vc_pct = classify_congestion(next_vol, max_cap)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Forecast Timestamp (t+1h)", target_ts.strftime("%Y-%m-%d %H:%M"))
c2.metric("Predicted Volume (t+1h)", f"{next_vol:,.0f} veh/hr", f"{delta_vol:+,.0f} vs t")
c3.metric("Actual Ground Truth (t+1h)", f"{actual_t1:,.0f} veh/hr", f"Error: {next_vol - actual_t1:+,.0f}")
c4.metric("Road Capacity Utilization", f"{vc_pct:.1f}%", tier_label)

st.markdown(
    f"<div style='padding:12px;border-radius:8px;background-color:{tier_color}22;"
    f"border-left:6px solid {tier_color};margin-bottom:14px;'>"
    f"<strong>Live Traffic Operations Status ({target_ts.strftime('%H:%M')}):</strong> "
    f"<span style='color:{tier_color};font-weight:bold;'>{tier_label}</span> — "
    f"Projected {next_vol:,.0f} / {max_cap:,.0f} veh/hr corridor capacity.</div>",
    unsafe_allow_html=True,
)

fig, ax = plt.subplots(figsize=(12, 4.2))
ax.axhspan(0, 0.35 * max_cap, color="#2ecc71", alpha=0.08, label="Low (0-2,548)")
ax.axhspan(0.35 * max_cap, 0.65 * max_cap, color="#f1c40f", alpha=0.08, label="Moderate (2,548-4,732)")
ax.axhspan(0.65 * max_cap, 0.85 * max_cap, color="#e67e22", alpha=0.08, label="Heavy (4,732-6,188)")
ax.axhspan(0.85 * max_cap, max_cap * 1.05, color="#e74c3c", alpha=0.10, label="Severe (>6,188)")

ax.plot(hist_times, hist_vols, marker="o", markersize=4, linewidth=2, color="#2c3e50", label="Past 24h Observed")
ax.plot(
    [hist_times[-1]] + pred_times, [hist_vols[-1]] + preds_vol,
    marker="D", markersize=6, linestyle="--", linewidth=2.3, color="#e74c3c",
    label=f"DL Forecast (+{horizon}h)"
)

valid_gt_times, valid_gt_vols = [], []
for s in range(horizon):
    expected_ts = target_ts + pd.Timedelta(hours=s)
    if sample_idx + s < len(ts_test) and ts_test[sample_idx + s] == expected_ts:
        valid_gt_times.append(expected_ts)
        valid_gt_vols.append(float(y_test[sample_idx + s, 0] * vol_range + scaler.data_min_[0]))

if valid_gt_times:
    ax.scatter(valid_gt_times, valid_gt_vols, color="#27ae60", s=70, zorder=5, label="Actual Ground Truth")

ax.set_title(f"24-Hour Lookback & {horizon}-Hour Forecast Trajectory", fontweight="bold")
ax.set_ylabel("Traffic Volume (veh/hr)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", ncol=3, fontsize=8.5)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Controlled Benchmark & Ablation Results (7,781 Test Hours)")
st.dataframe(results_df, use_container_width=True, hide_index=True)