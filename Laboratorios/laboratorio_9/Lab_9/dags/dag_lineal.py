"""
DAG de Airflow 'hiring_lineal'
Orquesta el pipeline de ML:
1. Crea carpetas de ejecución
2. Descarga datos (ahora en Python, sin Bash)
3. Divide los datos (hold-out)
4. Entrena el modelo
5. Lanza la interfaz de Gradio
"""
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from pendulum import datetime
import sys
import os
import requests

# Agregamos 'dags' al path para que Python encuentre nuestros scripts
sys.path.append('./dags')

# Importamos funciones del módulo auxiliar
try:
    from hiring_functions import (
        create_folders,
        split_data,
        preprocess_and_train,
        gradio_interface
    )
except ImportError:
    print("Error: No se encontró 'hiring_functions.py'.")


@dag(
    dag_id="hiring_lineal",
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,  # Ejecución manual
    catchup=False
)
def hiring_pipeline():
    """
    DAG para el pipeline de entrenamiento y despliegue
    de la interfaz de predicción de contratación.
    """

    #  Marcador de inicio
    start = EmptyOperator(task_id="inicio_pipeline")

    #  Crear estructura de carpetas (usa la fecha de ejecución 'ds')
    @task(task_id="crear_carpetas")
    def task_create_folders(**kwargs):
        create_folders(**kwargs)

    #  Descargar datos (sin Bash, usando requests)
    @task(task_id="descargar_datos")
    def task_download_data(**kwargs):
        ds = kwargs["ds"]  # Fecha de ejecución
        raw_dir = f"./{ds}/raw"
        os.makedirs(raw_dir, exist_ok=True)

        url = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        output_path = os.path.join(raw_dir, "data_1.csv")

        print(f"Descargando datos desde: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Lanza error si la descarga falla

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"Datos guardados en: {output_path}")

    #  Dividir los datos (hold-out)
    @task(task_id="dividir_datos")
    def task_split_data(**kwargs):
        split_data(**kwargs)

    #  Preprocesar y entrenar modelo
    @task(task_id="entrenar_modelo")
    def task_train_model(**kwargs):
        preprocess_and_train(**kwargs)

    #  Montar interfaz de Gradio
    @task(task_id="lanzar_interfaz_gradio")
    def task_launch_gradio():
        print("Lanzando interfaz de Gradio...")
        print("La URL pública aparecerá en los logs de esta tarea.")
        gradio_interface()

    # Secuencia de tareas
    (
        start
        >> task_create_folders()
        >> task_download_data()
        >> task_split_data()
        >> task_train_model()
        >> task_launch_gradio()
    )


# Instancia del DAG
hiring_pipeline()
