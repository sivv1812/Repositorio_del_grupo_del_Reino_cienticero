# optimize.py
#cargando librerias
import os
import json
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
import mlflow
import mlflow.sklearn

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import pickle


# aplicando configuracion global

RANDOM_STATE = 20351047
N_SPLITS = 5
SCORER = make_scorer(f1_score)

N_TRIALS = int(os.getenv("N_TRIALS", "1000"))
TARGET_COL = "Potability"


#cargando pipeline y cargando data

def load_data():
    df = pd.read_csv("water_potability.csv")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y

def build_pipeline(trial, y_target):
    impute_strategy = trial.suggest_categorical("imputer__strategy", ["median", "mean"])
    n_pos = int((y_target == 1).sum())
    n_neg = int((y_target == 0).sum())
    base_spw = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

    params = {
        "n_estimators": trial.suggest_int("xgb__n_estimators", 300, 1400),
        "max_depth": trial.suggest_int("xgb__max_depth", 3, 12),
        "learning_rate": trial.suggest_float("xgb__learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("xgb__subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb__colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("xgb__min_child_weight", 1.0, 12.0),
        "gamma": trial.suggest_float("xgb__gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("xgb__reg_alpha", 1e-8, 1e-1, log=True),
        "reg_lambda": trial.suggest_float("xgb__reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float(
            "xgb__scale_pos_weight",
            max(0.8, 0.5 * base_spw),
            2.5 * base_spw
        ),
    }

#aplicando XGBClassifier
    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        eval_metric="logloss",
        **{k.replace("xgb__", ""): v for k, v in params.items()}
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("xgb", xgb),
    ])

def objective_factory(X, y):
    def objective(trial):
        pipe = build_pipeline(trial, y_target=y)
        p = trial.params.copy()
        lr = p.get("xgb__learning_rate", None)
        depth = p.get("xgb__max_depth", None)
        run_name = f"XGBoost (lr={lr:.3g}, depth={depth})" if lr is not None else f"Trial {trial.number}"

        with mlflow.start_run(run_name=run_name, description="Optuna trial de XGBoost"):
            for k, v in p.items():
                mlflow.log_param(k, v)

            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(pipe, X, y, scoring=SCORER, cv=cv, n_jobs=-1)
            mean_f1 = float(np.mean(scores))
            std_f1 = float(np.std(scores))

            mlflow.log_metric("valid_f1", mean_f1)
            mlflow.log_metric("valid_f1_std", std_f1)
            mlflow.log_param("cv_folds", N_SPLITS)
            mlflow.log_param("n_samples", int(X.shape[0]))

            return mean_f1
    return objective

def rebuild_best_pipeline(best_params):
    imputer_strategy = best_params.get("imputer__strategy", "median")
    xgb_kwargs = {k.replace("xgb__", ""): v for k, v in best_params.items() if k.startswith("xgb__")}
    best_xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        objective="binary:logistic",
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        eval_metric="logloss",
        **xgb_kwargs
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("xgb", best_xgb),
    ])

#obteniendo mejor modelo
def get_best_model(experiment_id: str):
    runs = mlflow.search_runs([experiment_id])
    best = runs.sort_values("metrics.valid_f1", ascending=False).iloc[0]
    best_run_id = best["run_id"]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
    return best_model, best


#optimizando mejor modelo
def optimize_model():
    # 0) Carpetas locales
    Path("plots").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("configs").mkdir(exist_ok=True)

    # separando datos
    X, y = load_data()

    #  Usando  MLflow 
    experiment_name = os.getenv(
        "EXPERIMENT_NAME",
        f"lab8_optuna_xgb_RS{RANDOM_STATE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    print(f"[MLflow] Experiment: {exp.name} (id={exp.experiment_id})")

    # aplicando Optuna study con trials logueados en MLflow
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=0)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="xgb_optuna_f1")
    study.optimize(objective_factory(X, y), n_trials=N_TRIALS, show_progress_bar=False)

    print("Mejor f1 (CV):", study.best_value)
    print("Mejores hiperparámetros:", study.best_trial.params)

    # creando graficos de Optuna 
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

    fig1 = plot_optimization_history(study)
    fig1_path = "plots/optuna_optimization_history.png"
    fig1.figure.tight_layout()
    fig1.figure.savefig(fig1_path, dpi=160)
    plt.close(fig1.figure)

    fig2 = plot_param_importances(study)
    fig2_path = "plots/optuna_param_importances.png"
    fig2.figure.tight_layout()
    fig2.figure.savefig(fig2_path, dpi=160)
    plt.close(fig2.figure)

    with mlflow.start_run(run_name="Optuna study plots", description="Gráficos del estudio Optuna"):
        mlflow.log_artifact(fig1_path, artifact_path="plots")
        mlflow.log_artifact(fig2_path, artifact_path="plots")

    # entrenando el mejor modelo con el pipeline en todo el set
    best_params = study.best_trial.params.copy()
    best_pipeline = rebuild_best_pipeline(best_params)
    best_pipeline.fit(X, y)

    #  logueaando el modelo dentro del run del mejor trial
    runs_df = mlflow.search_runs([exp.experiment_id])
    best_trial_row = runs_df.sort_values("metrics.valid_f1", ascending=False).iloc[0]
    best_trial_run_id = best_trial_row["run_id"]
    with mlflow.start_run(run_id=best_trial_run_id):
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")
    print("[MLflow] Modelo agregado al run del mejor trial:", best_trial_run_id)

    # Aplicando Run final con log_model, pickle, configs, versions y FI
    final_run_name = f"Final model (best trial #{study.best_trial.number})"
    with mlflow.start_run(run_name=final_run_name, description="Mejor modelo tras Optuna (modelo ya entrenado)"):
        # Modelo en MLflow
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        # guardando modelo en Pickle
        model_pkl_path = "models/model.pkl"
        with open(model_pkl_path, "wb") as f:
            pickle.dump(best_pipeline, f)
        mlflow.log_artifact(model_pkl_path, artifact_path="models")

        # guardando mejores parametros en json
        best_params_path = "configs/best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        mlflow.log_artifact(best_params_path, artifact_path="configs")

        # aplicando Versiones
        versions = {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": __import__("xgboost").__version__,
            "optuna": optuna.__version__,
            "mlflow": mlflow.__version__,
        }
        versions_path = "configs/versions.json"
        with open(versions_path, "w") as f:
            json.dump(versions, f, indent=2)
        mlflow.log_artifact(versions_path, artifact_path="configs")

        # aplicando Feature importances
        xgb_step = best_pipeline.named_steps["xgb"]
        importances = getattr(xgb_step, "feature_importances_", None)
        if importances is not None:
            feat_names = list(X.columns)
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
            plt.figure(figsize=(8, 5))
            imp_df.head(20).plot(kind="bar", x="feature", y="importance", legend=False, title="Top-20 Feature Importances")
            plt.tight_layout()
            fi_path = "plots/feature_importances.png"
            plt.savefig(fi_path, dpi=160)
            plt.close()
            mlflow.log_artifact(fi_path, artifact_path="plots")

        # Métricas diagnóstica con set completo
        y_pred_all = best_pipeline.predict(X)
        mlflow.log_metric("final_fit_f1_on_full_data", float(f1_score(y, y_pred_all)))

    # Recuperando el mejor modelo con get_best_model desde MLflow y serializarlo
    best_model_from_mlflow, best_run_row = get_best_model(exp.experiment_id)
    model_pkl_path_gbm = "models/model_from_mlflow.pkl"
    with open(model_pkl_path_gbm, "wb") as f:
        pickle.dump(best_model_from_mlflow, f)
    print("[OK] Mejor modelo recuperado con get_best_model y guardado en:", model_pkl_path_gbm)
    print("[INFO] Mejor run_id:", best_run_row["run_id"], "| valid_f1:", best_run_row["metrics.valid_f1"])

def main():
    optimize_model()

if __name__ == "__main__":
    main()
