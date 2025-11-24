# scripts/utils.py
import os
import mlflow

def get_mlflow_uri():
    # En docker-compose, el host es 'mlflow' y el puerto 5000
    return os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')

def init_mlflow_experiment(name='xgboost_optuna'):
    uri = get_mlflow_uri()
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(name)
    except Exception:
        try:
            mlflow.create_experiment(name)
            mlflow.set_experiment(name)
        except:
            pass