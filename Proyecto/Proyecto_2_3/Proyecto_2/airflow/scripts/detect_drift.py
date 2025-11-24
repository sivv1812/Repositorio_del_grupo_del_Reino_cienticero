# scripts/detect_drift.py
import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json

DATA_DIR = os.getenv('DATA_DIR', '/opt/airflow/data')
DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', 0.1))  # umbral PSI o p-valor KS

def psi(expected, actual, buckets=10):
    def scale_range(input, minv, maxv):
        return (input - np.min(input)) / (np.max(input) - np.min(input) + 1e-9) * (maxv - minv) + minv
    expected_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=buckets)[0] / len(actual)
    # replace datos cero para evitar log(0)
    expected_percents = np.where(expected_percents == 0, 1e-6, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-6, actual_percents)
    return np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))

def run():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'df_prepared.parquet'))
    df['week'] = pd.to_datetime(df['purchase_date']).dt.isocalendar().week
    last_week = df['week'].max()
    train_weeks = df[df['week'] < last_week]
    new_week = df[df['week'] == last_week]
    drift_flags = {}
    numeric_cols = train_weeks.select_dtypes(include=['int64','float64']).columns.tolist()
    for col in numeric_cols:
        if train_weeks[col].nunique() < 2 or new_week[col].nunique() < 2:
            continue
        psi_val = psi(train_weeks[col].values, new_week[col].values)
        ks_p = ks_2samp(train_weeks[col].values, new_week[col].values).pvalue
        drift = (psi_val > 0.2) or (ks_p < 0.01)
        drift_flags[col] = {'psi': float(psi_val), 'ks_pvalue': float(ks_p), 'drift': bool(drift)}
    # si muchos features drift genera drift global
    n_drift = sum(1 for v in drift_flags.values() if v['drift'])
    global_drift = n_drift > max(1, 0.05 * len(numeric_cols))
    out = {'per_feature': drift_flags, 'n_drift': n_drift, 'global_drift': bool(global_drift)}
    with open(os.path.join(DATA_DIR, 'drift_report.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("Drift report written:", out)
    return out

if __name__ == '__main__':
    print(run())
