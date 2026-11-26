
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="I-94 Traffic Flow Forecaster",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "tensors.npz"


# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_assets():

    required_files = [
        MODELS_DIR / "HistGradientBoosting.pkl",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "selected_model.txt",
        MODELS_DIR / "experiment_results.csv",
        DATA_PATH,
    ]

    missing_files = [str(p) for p in required_files if not p.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Required project files are missing:\n"
            + "\n".join(missing_files)
        )

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        meta = pickle.load(f)

    with open(MODELS_DIR / "HistGradientBoosting.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODELS_DIR / "selected_model.txt", "r") as f:
        selected_model = f.read().strip()

    if selected_model != "HistGradientBoosting":
        raise ValueError(
            f"Expected HistGradientBoosting as selected model, "
            f"but found: {selected_model}"
        )

    results_df = pd.read_csv(MODELS_DIR / "experiment_results.csv")

    with np.load(DATA_PATH, allow_pickle=True) as data:
        X_test = data["X_test"].copy()
        y_test = data["y_test"].copy()
        ts_test = pd.to_datetime(data["ts_test"].copy())

    return meta, model, X_test, y_test, ts_test, results_df


# --------------------------------------------------
# CONGESTION CLASSIFICATION
# --------------------------------------------------

def classify_congestion(vol, max_cap):

    vc = max(vol, 0.0) / max_cap

    if vc < 0.35:
        return "LOW (Free-Flow)", "#2ecc71", vc * 100

    elif vc < 0.65:
        return "MODERATE", "#f1c40f", vc * 100

    elif vc < 0.85:
        return "HEAVY CONGESTION", "#e67e22", vc * 100

    else:
        return "SEVERE JAM ALERT", "#e74c3c", vc * 100


# --------------------------------------------------
# MULTI-STEP FORECASTING
# --------------------------------------------------

def forecast_multistep(
    window_scaled,
    start_ts,
    horizon,
    model,
    scaler,
    feature_cols,
    max_cap,
):

    vol_min = scaler.data_min_[0]
    vol_range = scaler.data_max_[0] - scaler.data_min_[0]

    curr_window = window_scaled.copy()

    preds_vol = []
    pred_times = []

    for step in range(horizon):

        target_ts = start_ts + pd.Timedelta(hours=step)

        x_in = curr_window.copy()

        # HistGradientBoosting requires a 2D flattened input.
        # Shape: (1, 24 * 25) = (1, 600)
        x_flat = x_in.reshape(1, -1)

        y_hat_scaled = float(model.predict(x_flat)[0])

        # Convert scaled prediction back to vehicles/hour.
        y_hat_vol = float(
            np.clip(
                y_hat_scaled * vol_range + vol_min,
                0.0,
                max_cap * 1.05
            )
        )

        preds_vol.append(y_hat_vol)
        pred_times.append(target_ts)

        # Prepare the next autoregressive input window.
        last_unscaled = scaler.inverse_transform(
            curr_window[-1:]
        )[0]

        next_unscaled = last_unscaled.copy()

        # Update predicted traffic volume.
        next_unscaled[0] = y_hat_vol

        # Update traffic volume / capacity ratio.
        next_unscaled[feature_cols.index("vc_ratio")] = (
            y_hat_vol / max_cap
        )

        # Update time-based features.
        next_unscaled[feature_cols.index("hour_sin")] = (
            np.sin(2 * np.pi * target_ts.hour / 24.0)
        )

        next_unscaled[feature_cols.index("hour_cos")] = (
            np.cos(2 * np.pi * target_ts.hour / 24.0)
        )

        next_unscaled[feature_cols.index("dow_sin")] = (
            np.sin(2 * np.pi * target_ts.day_of_week / 7.0)
        )

        next_unscaled[feature_cols.index("dow_cos")] = (
            np.cos(2 * np.pi * target_ts.day_of_week / 7.0)
        )

        next_unscaled[feature_cols.index("month_sin")] = (
            np.sin(2 * np.pi * target_ts.month / 12.0)
        )

        next_unscaled[feature_cols.index("month_cos")] = (
            np.cos(2 * np.pi * target_ts.month / 12.0)
        )

        # Update calendar-derived features.
        is_weekend = float(target_ts.day_of_week >= 5)

        is_holiday = next_unscaled[
            feature_cols.index("is_holiday")
        ]

        next_unscaled[
            feature_cols.index("is_weekend")
        ] = is_weekend

        next_unscaled[
            feature_cols.index("is_rush_hour")
        ] = float(
            (target_ts.day_of_week < 5)
            and (is_holiday == 0)
            and (
                (7 <= target_ts.hour <= 9)
                or (16 <= target_ts.hour <= 18)
            )
        )

        # Future weather/exogenous inputs are held at their
        # latest observed values during this short forecast.
        next_scaled = scaler.transform(
            next_unscaled.reshape(1, -1)
        )[0]

        curr_window = np.vstack([
            curr_window[1:],
            next_scaled
        ])

    return pred_times, preds_vol


# --------------------------------------------------
# LOAD PROJECT ASSETS
# --------------------------------------------------

try:
    (
        meta,
        active_model,
        X_test,
        y_test,
        ts_test,
        results_df,
    ) = load_assets()

except Exception as e:
    st.error(f"Could not load project assets: {e}")
    st.stop()


scaler = meta["scaler"]
feature_cols = meta["feature_cols"]
max_cap = meta["max_capacity"]


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title(
    "Spatio-Temporal Traffic Flow & Congestion Forecasting System"
)

st.caption(
    "Interstate 94 Westbound (ATR Station 301) | "
    "Validation-Selected HistGradientBoosting Model"
)


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.header("Simulation Controls")

sample_idx = st.sidebar.slider(
    "Test Set Window Index",
    min_value=0,
    max_value=len(X_test) - 1,
    value=min(320, len(X_test) - 1),
)

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [1, 2, 3],
    index=2,
    format_func=lambda h: f"Next {h} Hour(s)",
)

surge_pct = st.sidebar.slider(
    "What-If Traffic Surge / Drop (%)",
    min_value=-30,
    max_value=30,
    value=0,
    step=5,
)


# --------------------------------------------------
# PREPARE INPUT WINDOW
# --------------------------------------------------

window_scaled = X_test[sample_idx].copy()

if surge_pct != 0:

    unscaled_w = scaler.inverse_transform(window_scaled)

    unscaled_w[:, 0] = np.clip(
        unscaled_w[:, 0] * (1.0 + surge_pct / 100.0),
        0,
        max_cap
    )

    unscaled_w[:, feature_cols.index("vc_ratio")] = (
        unscaled_w[:, 0] / max_cap
    )

    window_scaled = scaler.transform(unscaled_w)


# The timestamp corresponds to the target being forecast.
target_ts = ts_test[sample_idx]

hist_unscaled = scaler.inverse_transform(window_scaled)

hist_vols = hist_unscaled[:, 0]

hist_times = [
    target_ts - pd.Timedelta(hours=24 - i)
    for i in range(24)
]


# --------------------------------------------------
# GENERATE FORECAST
# --------------------------------------------------

pred_times, preds_vol = forecast_multistep(
    window_scaled=window_scaled,
    start_ts=target_ts,
    horizon=horizon,
    model=active_model,
    scaler=scaler,
    feature_cols=feature_cols,
    max_cap=max_cap,
)


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

next_vol = preds_vol[0]

curr_vol = hist_vols[-1]

delta_vol = next_vol - curr_vol

vol_range = scaler.data_max_[0] - scaler.data_min_[0]

actual_t1 = float(
    y_test[sample_idx, 0] * vol_range + scaler.data_min_[0]
)

tier_label, tier_color, vc_pct = classify_congestion(
    next_vol,
    max_cap
)


# --------------------------------------------------
# TOP METRIC CARDS
# --------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Forecast Timestamp (t+1h)",
    target_ts.strftime("%Y-%m-%d %H:%M")
)

c2.metric(
    "Predicted Volume (t+1h)",
    f"{next_vol:,.0f} veh/hr",
    f"{delta_vol:+,.0f} vs t"
)

c3.metric(
    "Actual Ground Truth (t+1h)",
    f"{actual_t1:,.0f} veh/hr",
    f"Error: {next_vol - actual_t1:+,.0f}"
)

c4.metric(
    "Road Capacity Utilization",
    f"{vc_pct:.1f}%",
    tier_label
)


# --------------------------------------------------
# TRAFFIC STATUS
# --------------------------------------------------

st.markdown(
    f"""
    <div style="
        padding:12px;
        border-radius:8px;
        background-color:{tier_color}22;
        border-left:6px solid {tier_color};
        margin-bottom:14px;
    ">
        <strong>
            Traffic Operations Status ({target_ts.strftime('%H:%M')}):
        </strong>
        <span style="color:{tier_color};font-weight:bold;">
            {tier_label}
        </span>
        — Projected {next_vol:,.0f} / {max_cap:,.0f}
        veh/hr corridor capacity.
    </div>
    """,
    unsafe_allow_html=True,
)


st.caption(
    "For multi-hour forecasts, future weather and other "
    "exogenous inputs are held at their latest observed values. "
    "Predictions after the first hour are autoregressive."
)


# --------------------------------------------------
# FORECAST VISUALIZATION
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 4.2))

ax.axhspan(
    0,
    0.35 * max_cap,
    color="#2ecc71",
    alpha=0.08,
    label="Low (0-35%)"
)

ax.axhspan(
    0.35 * max_cap,
    0.65 * max_cap,
    color="#f1c40f",
    alpha=0.08,
    label="Moderate (35-65%)"
)

ax.axhspan(
    0.65 * max_cap,
    0.85 * max_cap,
    color="#e67e22",
    alpha=0.08,
    label="Heavy (65-85%)"
)

ax.axhspan(
    0.85 * max_cap,
    max_cap * 1.05,
    color="#e74c3c",
    alpha=0.10,
    label="Severe (85%+)"
)

# Historical traffic.
ax.plot(
    hist_times,
    hist_vols,
    marker="o",
    markersize=4,
    linewidth=2,
    color="#2c3e50",
    label="Past 24h Observed"
)

# Forecast trajectory.
ax.plot(
    [hist_times[-1]] + pred_times,
    [hist_vols[-1]] + preds_vol,
    marker="D",
    markersize=6,
    linestyle="--",
    linewidth=2.3,
    color="#e74c3c",
    label=f"HistGradientBoosting Forecast (+{horizon}h)"
)


# --------------------------------------------------
# ACTUAL GROUND TRUTH OVERLAY
# --------------------------------------------------

valid_gt_times = []
valid_gt_vols = []

for s in range(horizon):

    expected_ts = target_ts + pd.Timedelta(hours=s)

    if (
        sample_idx + s < len(ts_test)
        and ts_test[sample_idx + s] == expected_ts
    ):

        valid_gt_times.append(expected_ts)

        actual_vol = float(
            y_test[sample_idx + s, 0] * vol_range
            + scaler.data_min_[0]
        )

        valid_gt_vols.append(actual_vol)


if valid_gt_times:

    ax.scatter(
        valid_gt_times,
        valid_gt_vols,
        color="#27ae60",
        s=70,
        zorder=5,
        label="Actual Ground Truth"
    )


ax.set_title(
    f"24-Hour Lookback & {horizon}-Hour Forecast Trajectory",
    fontweight="bold"
)

ax.set_ylabel("Traffic Volume (veh/hr)")

ax.grid(True, alpha=0.3)

ax.legend(
    loc="upper left",
    ncol=3,
    fontsize=8.5
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# --------------------------------------------------
# MODEL BENCHMARK RESULTS
# --------------------------------------------------

st.subheader(
    f"Test Set Benchmark & Ablation Results ({len(y_test):,} Test Hours)"
)

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True
)
