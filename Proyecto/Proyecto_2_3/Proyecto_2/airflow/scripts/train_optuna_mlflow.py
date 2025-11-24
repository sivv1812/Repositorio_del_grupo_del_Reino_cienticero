# scripts/train_optuna_mlflow.py
import os
import joblib
import json
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier
import optuna
from optuna.pruners import MedianPruner
from utils import init_mlflow_experiment, get_mlflow_uri
from sklearn.model_selection import train_test_split

DATA_DIR = os.getenv('DATA_DIR', '/opt/airflow/data')
MLFLOW_URI = get_mlflow_uri()
RANDOM_STATE = 20351047

def load_data():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'df_prepared.parquet'))
    # transformar  targets en terciles aquí 
    TARGET = 'items'
    df = df.dropna(subset=[TARGET])
    y = df[TARGET].astype(float)
    q1, q2 = y.quantile([1/3, 2/3]).values
    bins = [-np.inf, q1, q2, np.inf]
    df['y_cat'] = pd.cut(y, bins=bins, labels=[0,1,2]).astype(int)
    X = df.drop(columns=[TARGET, 'y_cat', 'purchase_date', 'date'], errors='ignore')
    # Para simplificar  one-hot encoding aquí
    X = pd.get_dummies(X, drop_first=True)
    return train_test_split(X, df['y_cat'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['y_cat'])

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': RANDOM_STATE,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    return f1_score(y_val, preds, average='weighted')

def run():
    init_mlflow_experiment('xgboost_optuna_v2')
    mlflow.set_tracking_uri(MLFLOW_URI)
    X_train, X_test, y_train, y_test = load_data()
    # optuna ejecución
    def obj(trial):
        return objective(trial, X_train, y_train, X_test, y_test)
    study = optuna.create_study(direction='maximize', pruner=MedianPruner(n_warmup_steps=3))
    study.optimize(obj, n_trials=20)  # ajustar n_trials para producción

    best = study.best_params
    # Entrenar modelo final
    best_model = XGBClassifier(**best, random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss')
    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')

    # Log con MLflow
    with mlflow.start_run():
        mlflow.log_params(best)
        mlflow.log_metric('f1_weighted', float(f1))
    joblib.dump(best_model, os.path.join(DATA_DIR, 'model_best.joblib'))
    print("Training finished, F1:", f1)
    return {'f1': f1, 'params': best}

if __name__ == '__main__':
    run()
