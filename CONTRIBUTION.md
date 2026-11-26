
# Team Contribution Record — Project 4: Traffic Flow Forecasting

| Team Member | Designated Role | Key Responsibilities & Code Contributions | Owned Files & Verification |
| :--- | :--- | :--- | :--- |
| **Naman Maheshwari** | Team Lead & Data Pipeline Architect | Audited raw CSV traps (midnight-only holidays, 7,629 duplicate timestamps, 0 K / 9,831 mm sensor glitches, 7,387h outage); implemented leak-free chronological 70/15/15 split, cyclical encodings, and contiguous sliding windows. | `src/preprocessing.py`, `README.md` |
| **Anant Raikwar** | Deep Learning Lead | Built `Vanilla_LSTM`, `Stacked_BiLSTM`, and the custom Keras-serializable `TemporalAttention` + `Conv1D-BiLSTM` architecture with `EarlyStopping` and `ReduceLROnPlateau`. | `src/models.py`, `src/train.py`, `models/*.keras` |
| **Almas Akhtar Shah** | Evaluation & Ablation Lead | Designed baseline benchmarks (`Seasonal Naive`, `Persistence`, `HistGradientBoosting`), executed the weather-feature ablation study, mapped $V/C$ congestion tiers, and generated diagnostic plots. | `src/evaluate.py`, `models/experiment_results.csv`, `reports/figures/*.png` |
| **Abeer Trivedi** | Application & Presentation Lead | Built the interactive Streamlit multi-step (`+1h` to `+3h`) forecasting & what-if traffic surge simulator, compiled the team contribution log, and authored the 6-slide presentation deck. | `app.py`, `CONTRIBUTION.md`, Presentation Slides |
