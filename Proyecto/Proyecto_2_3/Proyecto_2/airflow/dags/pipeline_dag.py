from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import os
import json

# Definir argumentos por defecto
default_args = {
    "owner": "equipo_datos",
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
}

DATA_DIR = os.getenv('DATA_DIR', '/opt/airflow/data')

# Función para decidir qué camino tomar
def _branch_on_drift():
    rpt_path = os.path.join(DATA_DIR, 'drift_report.json')
    
    # Si no existe el reporte, reentrenamos por seguridad
    if not os.path.exists(rpt_path):
        return 'retrain_task'
    
    with open(rpt_path, 'r') as f:
        data = json.load(f)
        
    # Si hay drift global (True), reentrenamos. Si no, saltamos.
    if data.get('global_drift', False):
        return 'retrain_task'
    else:
        return 'skip_retrain'

with DAG(
    dag_id='full_ml_pipeline_v2', 
    default_args=default_args,
    schedule_interval="@daily", 
    start_date=days_ago(1), 
    catchup=False,
    tags=['produccion', 'sodai']
) as dag:

    start = EmptyOperator(task_id='start')
    end = EmptyOperator(task_id='end')

    # 1. Extraer
    extract = BashOperator(
        task_id='extract',
        bash_command='python /opt/airflow/scripts/extract.py'
    )

    # 2. Preprocesar
    preprocess = BashOperator(
        task_id='preprocess',
        bash_command='python /opt/airflow/scripts/preprocess.py'
    )

    # 3. Detectar Drift
    detect = BashOperator(
        task_id='detect_drift',
        bash_command='python /opt/airflow/scripts/detect_drift.py'
    )

    # 4. Decisión (Branching)
    branch = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=_branch_on_drift
    )

    # Camino A: Reentrenar
    retrain = BashOperator(
        task_id='retrain_task',
        bash_command='python /opt/airflow/scripts/train_optuna_mlflow.py'
    )

    # Camino B: Saltar reentrenamiento
    skip = EmptyOperator(task_id='skip_retrain')

    # 5. Predecir (Este archivo lo crearemos ahora)
    predict = BashOperator(
        task_id='predict',
        bash_command='python /opt/airflow/scripts/predict.py',
        trigger_rule='none_failed_min_one_success' # Se ejecuta si viene de A o de B
    )

    # Flujo
    start >> extract >> preprocess >> detect >> branch
    branch >> retrain >> predict
    branch >> skip >> predict
    predict >> end