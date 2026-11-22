import json
from pathlib import Path

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in src.splitlines()]}

cells = [
    md("# Project 4: Traffic Flow & Congestion Forecasting Using Deep Learning\n**Corridor:** Interstate 94 Westbound (ATR Station 301, Capacity = 7,280 veh/hr)\n**Architecture:** Conv1D + Stacked Bi-LSTM + Custom Temporal Attention"),
    md("## 1. Data Audit & Leak-Free Preprocessing (`src/preprocessing.py`)\n- Propagated 61 midnight-only `holiday` labels across full 24h dates.\n- Deduplicated 7,629 multi-weather timestamp rows (`48,204` -> `40,575` unique hours).\n- Nullified impossible sensor glitches (`temp = 0.0 K`, `rain_1h = 9831.3 mm`).\n- Reindexed to strict hourly grid, interpolated only gaps `<= 2h`, and excluded the 10-month 2014-2015 sensor outage from sliding windows.\n- Chronological 70/15/15 split with `MinMaxScaler` fit strictly on Train."),
    code("from src.preprocessing import run_pipeline\nrun_pipeline()"),
    md("## 2. Controlled Model Progression & Weather Ablation (`src/train.py`)\nTrains Seasonal Naive, Persistence, HistGradientBoosting, Vanilla LSTM, Stacked Bi-LSTM, Full Multimodal Conv1D-BiLSTM-Attention (25 features), and Core Spatio-Temporal Conv1D-BiLSTM-Attention (14 features) with `EarlyStopping` and `ReduceLROnPlateau`."),
    code("from src.train import train_all\ntrain_all()"),
    md("## 3. Dual-Task Evaluation & Visualizations (`src/evaluate.py`)\nEvaluates MAE, RMSE, MAPE, R2, and 4-Tier Volume-to-Capacity Congestion Classification (Accuracy & Macro-F1)."),
    code("from src.evaluate import evaluate_all\nevaluate_all()"),
    md("## 4. Display Saved Benchmark Table & Diagnostic Plots"),
    code("import pandas as pd\nfrom IPython.display import Image, display\n\ndf_res = pd.read_csv('models/experiment_results.csv')\ndisplay(df_res)\nfor img in ['01_training_curves.png', '02_model_benchmark.png', '03_actual_vs_predicted_week.png', '04_confusion_matrix_residuals.png']:\n    display(Image(filename=f'reports/figures/{img}'))"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("notebooks/Traffic_Flow_Forecasting.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
print(f"SUCCESS -> Generated {out_path}")