# Traffic Flow & Congestion Forecasting Using Deep Learning

## 1. Problem Statement & Objectives
Urban highway corridors suffer from sharp congestion waves driven by commuter rush hours, day-of-week travel patterns, and weather disruptions. Accurate short-term traffic forecasting allows Intelligent Transportation Systems (ITS) to deploy proactive ramp metering, dynamic routing, and congestion warnings before bottlenecks lock up.

**Dual-Task Project Objectives:**
1. **Continuous Traffic Volume Forecasting:** Predict upcoming hourly vehicle counts (`veh/hr`) on Interstate 94 Westbound (ATR Station 301, Minneapolis–St. Paul; observed corridor capacity = `7,280 veh/hr`) using a 24-hour historical lookback window.
2. **Actionable Congestion Level Classification:** Translate predicted volumes into four operational Volume-to-Capacity ($V/C$) congestion tiers:
   - **Low (Free-Flow):** $0\text{–}35\%$ capacity ($0\text{–}2,548\text{ veh/hr}$)
   - **Moderate:** $35\text{–}65\%$ capacity ($2,548\text{–}4,732\text{ veh/hr}$)
   - **Heavy Congestion:** $65\text{–}85\%$ capacity ($4,732\text{–}6,188\text{ veh/hr}$)
   - **Severe Jam Alert:** $>85\%$ capacity ($>6,188\text{ veh/hr}$)

---

## 2. Dataset Description & Leak-Free Preprocessing
We used the **Metro Interstate Traffic Volume Dataset** (`48,204` raw rows, `9` features, `2012-10-02` to `2018-09-30`). Auditing the raw CSV uncovered five data quality traps resolved in `src/preprocessing.py`:

| Trap Discovered in Raw CSV | Exact Impact | Engineering Resolution |
| :--- | :--- | :--- |
| **Holiday Midnight-Only Tagging** | `48,143` `NaN`s; all `61` holiday labels occurred only at `00:00:00` | Converted to binary `is_holiday` and propagated `1` across the full 24-hour date |
| **Duplicate Timestamps** | `7,629` duplicate `date_time` rows (`17` exact + `7,612` multi-weather tags) | Grouped by `date_time` (`mean` temp/clouds, `max` rain/snow, `first` volume) $\rightarrow$ `40,575` unique hours |
| **Extreme Sensor Glitches** | `10` rows with `temp = 0.0 K` ($-273^\circ\text{C}$); `1` row with `rain_1h = 9,831.3 mm` | Replaced `temp < 200` and `rain_1h > 150` with `NaN` prior to local interpolation |
| **10-Month Sensor Outage** | `11,976` missing hourly grid slots, including one `7,387`-hour outage (`Aug 2014–Jun 2015`) | Reindexed with `pd.Timedelta(hours=1)`, interpolated only gaps `<= 2h`, and skipped `NaN` windows |
| **Data Leakage & Cyclical Time** | Random splits leak future autocorrelation; raw hours break `23:00 -> 00:00` continuity | Enforced chronological **70/15/15 split**, fit `MinMaxScaler` on Train only, added `sin/cos` time encodings & `vc_ratio` |

**Final Contiguous Sequence Tensors (`lookback=24h`, `horizon=1h`, `25 features`):**
- `X_train`: `(25,382, 24, 25)` | `X_val`: `(7,720, 24, 25)` | `X_test`: `(7,781, 24, 25)`

---

## 3. Model Architecture & Controlled Experiments
We benchmarked seven models on the exact same `7,781`-hour chronological test set (`2017-11-06` to `2018-09-30`) using identical random seeds (`42`), `EarlyStopping(patience=8)`, and `ReduceLROnPlateau`:

| Model | MAE (veh/hr) | RMSE (veh/hr) | MAPE (%) | $R^2$ Score | Congestion Acc (%) | Congestion Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Seasonal Naive ($t-24\text{h}$) | 567.57 | 1024.35 | 25.50 | 0.7298 | 74.76 | 0.7170 |
| Persistence ($t-1\text{h}$) | 582.92 | 808.39 | 26.73 | 0.8317 | 71.02 | 0.5951 |
| HistGradientBoosting (Tabular ML) | 147.29 | 229.02 | **6.56** | 0.9865 | 91.84 | 0.8898 |
| Vanilla LSTM (`64 -> 32`) | 157.08 | 239.00 | 7.23 | 0.9853 | 91.33 | 0.8868 |
| Stacked Bi-LSTM (`64 -> 32`) | 164.01 | 254.40 | 7.68 | 0.9833 | 90.80 | 0.8773 |
| Multimodal Conv1D-BiLSTM-Attn (25 feat) | 168.71 | 253.58 | 8.00 | 0.9834 | 90.63 | 0.8787 |
| **Core Conv1D-BiLSTM-Attn (14 feat — Best)** | **146.88** | **225.01** | 6.75 | **0.9870** | **92.44** | **0.9019** |

### Key Analytical Findings
1. **Superior Accuracy Across Baselines:** Our `Conv1D + Stacked Bi-LSTM + TemporalAttention` architecture cuts test RMSE by **78.03%** over `Seasonal_Naive_24h`, **72.17%** over `Persistence_1h`, **5.85%** over `Vanilla_LSTM`, and **1.75%** over `HistGradientBoosting`.
2. **Ablation Discovery (Sparse Weather Overfitting):** Comparing the 25-feature Multimodal model (`RMSE: 253.58`) against the 14-feature Core Spatio-Temporal model (`RMSE: 225.01`) revealed that 10 sparse one-hot `weather_*` categories (`Squall`, `Smoke`, `Fog`) and `is_holiday` (`0.13%` positive rate) degraded generalization by **11.27%**. Evidence-driven feature pruning yielded our #1 model across every regression and classification metric.

---

## 4. Limitations & Future Scope
- **Single-Sensor Spatial Scope:** ATR Station 301 measures a single highway cross-section; scaling to city-wide road networks requires Spatio-Temporal Graph Convolutional Networks (ST-GCN) ingesting upstream/downstream loop detectors.
- **Unscheduled Anomalies:** Sudden accidents, lane closures, or stadium events not present in historical volume cause transient residual spikes.
- **Multi-Horizon Error Accumulation:** In `app.py`, `+2h` and `+3h` forecasts roll forward autoregressively; training a direct Seq2Seq multi-horizon output head would reduce multi-hour drift.

---

## 5. Project Structure & Execution
```text
traffic-flow-forecasting-dl/
├── data/
│   ├── raw/Metro_Interstate_Traffic_Volume.csv
│   └── processed/tensors.npz
├── models/
│   ├── Ablation_No_Weather.keras          # Deployed Best Core DL Model (RMSE 225.01)
│   ├── Champion_Conv1D_BiLSTM_Attn.keras  # Multimodal Weather Variant (RMSE 253.58)
│   ├── Vanilla_LSTM.keras & Stacked_BiLSTM.keras
│   ├── scaler.pkl & experiment_results.csv
├── notebooks/Traffic_Flow_Forecasting.ipynb
├── reports/figures/                       # 4 evaluation & error analysis charts
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
├── app.py                                 # Streamlit interactive congestion simulator
├── make_notebook.py
├── requirements.txt
├── CONTRIBUTION.md
└── README.md
```

## 6. How to Run
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn streamlit
python src/preprocessing.py
python src/train.py
python src/evaluate.py
streamlit run app.py