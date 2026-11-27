# Traffic Flow & Congestion Forecasting Using Deep Learning

## 1. Problem Statement & Objectives

Urban highway corridors suffer from sharp congestion waves driven by commuter rush hours, day-of-week travel patterns, and weather disruptions. Accurate short-term traffic forecasting allows Intelligent Transportation Systems (ITS) to deploy proactive ramp metering, dynamic routing, and congestion warnings before bottlenecks lock up.

**Dual-Task Project Objectives:**

1. **Continuous Traffic Volume Forecasting:** Predict upcoming hourly vehicle counts (`veh/hr`) on Interstate 94 Westbound (ATR Station 301, Minneapolis–St. Paul; observed corridor capacity = `7,280 veh/hr`) using a 24-hour historical lookback window.
2. **Actionable Congestion Level Classification:** Translate predicted volumes into four operational Volume-to-Capacity ($V/C$) congestion tiers:
   - **Low (Free-Flow):** 0–35% capacity (`0–2,548 veh/hr`)
   - **Moderate:** 35–65% capacity (`2,548–4,732 veh/hr`)
   - **Heavy Congestion:** 65–85% capacity (`4,732–6,188 veh/hr`)
   - **Severe Jam Alert:** >85% capacity (`>6,188 veh/hr`)

---

## 2. Dataset Description & Leak-Free Preprocessing

### Dataset Overview

The project uses the **Metro Interstate Traffic Volume Dataset**, containing hourly traffic observations from Interstate 94 Westbound in the Minneapolis–St. Paul metropolitan area.

- **Raw observations:** 48,204 rows
- **Observation period:** October 2, 2012 – September 30, 2018
- **Target variable:** `traffic_volume`
- **Input features:** Weather conditions, temperature, humidity, wind speed, cloud coverage, holiday information, and engineered temporal and traffic features.

### Data Cleaning and Preprocessing

The preprocessing pipeline addresses missing observations, duplicate timestamps, sensor anomalies, and temporal data leakage.

- **Holiday handling:** The dataset records holiday labels only at midnight for applicable dates. Missing holiday values are treated as non-holiday observations.
- **Duplicate timestamps:** Duplicate hourly records are aggregated to produce one observation per timestamp. Weather measurements are aggregated appropriately, while traffic volume is retained using the first available observation.
- **Sensor anomalies:** Invalid temperature and precipitation readings are identified and handled as missing values before preprocessing.
- **Missing hourly observations:** The timeline is reindexed to hourly frequency. Short gaps in exogenous weather features are handled using limited forward-filling. Missing target traffic volumes are not interpolated, preventing future target information from influencing historical observations.
- **Long outages:** Extended missing periods are preserved rather than artificially reconstructed. Sequences containing unavailable target values are excluded.
- **Chronological splitting:** Data is divided into training, validation, and test sets in chronological order (70% / 15% / 15%), avoiding random temporal mixing.
- **Feature scaling:** The scaler is fitted using training data only and then applied to validation and test data.
- **Temporal feature engineering:** Hour, day of the week, and month are represented using cyclical transformations. Additional features include weekend indicators, rush-hour indicators, and historical Volume-to-Capacity (`vc_ratio`).

### Final Sequence Dataset

Each input sequence contains 24 consecutive hourly observations and 25 features.

| Dataset | Sequence Shape |
|---|---|
| Training | `(13955, 24, 25)` |
| Validation | `(7411, 24, 25)` |
| Testing | `(7403, 24, 25)` |

The prediction target is the traffic volume for the next hour.

---

## 3. Model Architecture & Controlled Experiments

The project evaluates traditional machine learning and deep learning approaches under the same chronological data-splitting strategy.

### Models Evaluated

1. **Persistence Baseline:** Uses the most recently observed traffic volume as the next-hour prediction.
2. **Seasonal Naive Baseline:** Uses traffic volume from the same hour on the previous day.
3. **HistGradientBoosting:** A tree-based machine learning model trained on flattened 24-hour input sequences.
4. **Vanilla LSTM:** A recurrent neural network designed to learn temporal dependencies.
5. **Stacked BiLSTM:** A bidirectional LSTM architecture that captures patterns across the historical input sequence.
6. **Conv1D-BiLSTM-Attention:** A hybrid architecture combining convolutional feature extraction, bidirectional recurrent layers, and temporal attention.
7. **No-Weather Ablation:** An experiment that excludes weather input features to examine their contribution to forecasting performance.

### Model Selection

The final model is selected using **validation RMSE**, rather than test-set performance. This helps prevent test-set information from influencing model selection.

The validation results were:

| Model | Validation RMSE (veh/hr) |
|---|---:|
| HistGradientBoosting | 238.95 |
| Stacked_BiLSTM | 268.94 |
| Ablation_No_Weather | 240.50 |
| Vanilla_LSTM | 270.81 |
| Champion_Conv1D_BiLSTM_Attn | 283.28 |

**Selected Model: HistGradientBoosting**

HistGradientBoosting achieved the lowest validation RMSE and was selected for final evaluation and dashboard deployment.

### Final Test-Set Benchmark

The following results are from the final evaluation run:

| Model | MAE (veh/hr) | RMSE (veh/hr) | MAPE (%) | $R^2$ Score | Congestion Accuracy (%) | Congestion Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal_Naive_24h | 568.60 | 1026.74 | 25.33 | 0.7281 | 74.79 | 0.7173 |
| Persistence_1h | 584.40 | 810.03 | 26.82 | 0.8307 | 70.96 | 0.5947 |
| HistGradientBoosting | **152.62** | **235.97** | **6.62** | **0.9856** | **91.49** | **0.8861** |
| Vanilla_LSTM | 182.17 | 272.40 | 9.12 | 0.9809 | 89.65 | 0.8662 |
| Stacked_BiLSTM | 170.03 | 258.17 | 8.23 | 0.9828 | 90.36 | 0.8764 |
| Ablation_No_Weather | 176.32 | 267.08 | 8.34 | 0.9816 | 90.23 | 0.8668 |
| Champion_Conv1D_BiLSTM_Attn | 191.06 | 284.06 | 8.54 | 0.9792 | 88.48 | 0.8030 |

**Evaluation metrics:**

- **MAE:** Mean Absolute Error, measuring average absolute prediction error.
- **RMSE:** Root Mean Squared Error, penalizing larger prediction errors.
- **MAPE:** Mean Absolute Percentage Error. Observations with actual traffic volume below 100 veh/hr are excluded to avoid unstable percentage errors.
- **$R^2$:** Coefficient of determination, measuring explained variance.
- **Congestion Accuracy:** Percentage of predictions assigned to the correct congestion tier.
- **Congestion Macro-F1:** Macro-averaged F1 score across the four congestion categories.

### Rush-Hour vs. Off-Peak Analysis

The selected HistGradientBoosting model was also evaluated across different traffic regimes.

| Traffic Regime | RMSE (veh/hr) |
|---|---:|
| Peak Rush-Hour | 293.92 |
| Off-Peak | 221.27 |

The higher rush-hour RMSE indicates that traffic patterns during peak periods are more difficult to forecast accurately.

### Weather Ablation Experiment

The no-weather ablation experiment evaluates the model after removing weather input features.

The comparison helps assess whether weather-related information contributes to forecasting performance. However, zeroing scaled weather features represents minimum scaled values rather than neutral or missing weather conditions, so the ablation should be interpreted with this limitation in mind.

---

## 4. Interactive Streamlit Dashboard

The project includes an interactive Streamlit dashboard for exploring traffic forecasts and hypothetical traffic scenarios.

### Dashboard Features

- **Traffic Forecasting:** Generates next-hour traffic volume predictions using the selected HistGradientBoosting model.
- **Congestion Classification:** Converts predicted traffic volume into one of four operational congestion tiers.
- **Historical Comparison:** Displays predicted and actual traffic volumes for the selected test-set timestamp.
- **Multi-Hour Forecasting:** Supports recursive multi-step forecasting using previous predictions as inputs.
- **Traffic Surge Simulation:** Allows users to simulate hypothetical increases in traffic volume.
- **Traffic Drop Simulation:** Allows users to simulate hypothetical reductions in traffic volume.
- **Capacity Utilization:** Displays predicted traffic volume as a percentage of the observed corridor capacity.

**Important forecasting limitation:** The model is trained for one-hour-ahead prediction. Multi-hour forecasts are generated recursively, so prediction errors may accumulate over subsequent forecast steps. Future weather and other exogenous inputs are held at their latest observed values during recursive forecasting.

Traffic surge and drop simulations are hypothetical scenarios and should not be interpreted as standard model evaluation results.

---

## 5. Limitations & Future Scope

Although the project demonstrates strong forecasting performance, several limitations remain:

1. **Single-Sensor Scope:** The model uses observations from one traffic monitoring station. Its performance may not generalize to other highway corridors or monitoring locations.
2. **Unexpected Events:** Accidents, road closures, construction, and special events are not explicitly modeled.
3. **Weather Forecast Availability:** Future weather conditions are not supplied by a live weather forecasting service. The dashboard holds the latest observed exogenous values constant during recursive predictions.
4. **Multi-Step Forecasting:** The model is optimized for one-hour-ahead forecasting. Recursive predictions over multiple hours may accumulate errors.
5. **Congestion Thresholds:** Congestion categories are defined using fixed capacity-based thresholds. Actual operational congestion can also depend on road geometry, traffic composition, and local conditions.

### Future Improvements

- Integrate real-time traffic and weather APIs.
- Extend the project to multiple monitoring stations.
- Incorporate accident, construction, and road-event information.
- Explore direct multi-horizon forecasting architectures.
- Evaluate adaptive congestion thresholds.
- Deploy the dashboard as a cloud-hosted application.

---

## 6. Repository Structure

```text
traffic-flow-forecasting-dl/
│
├── app.py
├── README.md
├── requirements.txt
├── CONTRIBUTION.md
│
├── data/
│   ├── raw/
│   │   └── Metro_Interstate_Traffic_Volume.csv
│   │
│   └── processed/
│       └── tensors.npz
│
├── models/
│   ├── HistGradientBoosting.pkl
│   ├── scaler.pkl
│   ├── selected_model.txt
│   ├── experiment_results.csv
│   ├── test_predictions.csv
│   └── training_histories.pkl
│
├── reports/
│
└── src/
    ├── preprocessing.py
    ├── models.py
    ├── train.py
    └── evaluate.py
```

---

## 7. Installation & Execution

### Prerequisites

- Python environment compatible with the installed project dependencies.
- `pip` package manager.
- The Metro Interstate Traffic Volume dataset.

### Step 1: Clone the Repository

```bash
git clone https://github.com/anantraikwar09/traffic-flow-forecasting-dl.git
cd traffic-flow-forecasting-dl
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

Place the original dataset in:

```text
data/raw/Metro_Interstate_Traffic_Volume.csv
```

### Step 4: Run Preprocessing

```bash
python src/preprocessing.py
```

This generates the processed sequence tensors and scaler artifacts.

### Step 5: Train the Models

```bash
python src/train.py
```

Training includes the baseline, machine learning, deep learning, and ablation experiments. TensorFlow-compatible Python and dependency versions are required for the deep learning experiments.

### Step 6: Evaluate the Models

```bash
python src/evaluate.py
```

This generates the test-set benchmark results and evaluation outputs.

### Step 7: Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard uses the saved HistGradientBoosting model and processed data artifacts. TensorFlow is not required to run the dashboard.

**Note:** The model is stored as a Python pickle file. Only load model files from trusted sources.

---

## 8. Technologies Used

- **Python:** Core programming language.
- **Pandas & NumPy:** Data manipulation and numerical computation.
- **Scikit-learn:** HistGradientBoosting, preprocessing, and evaluation metrics.
- **TensorFlow / Keras:** Deep learning model development and training.
- **Matplotlib & Seaborn:** Data visualization.
- **Streamlit:** Interactive forecasting dashboard.
- **Google Colab:** Cloud-based deep learning training environment.

---

## 9. Project Outcome

This project demonstrates an end-to-end traffic forecasting workflow, from data cleaning and leakage-aware preprocessing to model comparison, congestion classification, and interactive deployment.

HistGradientBoosting was selected using validation RMSE and achieved the following test-set results:

- **MAE:** 152.62 veh/hr
- **RMSE:** 235.97 veh/hr
- **$R^2$ Score:** 0.9856
- **Congestion Accuracy:** 91.49%
- **Congestion Macro-F1:** 0.8861

The results demonstrate that a carefully engineered tree-based model can outperform the evaluated deep learning architectures on this particular traffic forecasting dataset.
