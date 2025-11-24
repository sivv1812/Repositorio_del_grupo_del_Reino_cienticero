# Documentación del Pipeline Productivo (SodAI Drinks)

## 1. Descripción del DAG
El DAG **`full_ml_pipeline_v2`** ordena el ciclo de vida completo del modelo de predicción de compras. Se ejecuta diariamente (`@daily`) y su proposito es incorporar datos nuevos, evaluar su calidad de estos y actualizar el modelo predictivo en caso de requerirlo.

### Tareas del Pipeline:
1.  **`extract`**: Simula la extracción de datos crudos de los tres dataset usados en el proyecto `transacciones`, `clientes` y `productos`, desde el data lake hacia el entorno de procesamiento.
2.  **`preprocess`**: Limpia los datos, realiza el *feature engineering* creando variables temporales, dummies, entre otras y genera un dataset preparado (`df_prepared.parquet`).
3.  **`detect_drift`**: Compara la distribución de los datos de la semana más reciente contra la historia. Utiliza métricas estadísticas (PSI y KS-Test) para determinar si hubo Drift, o sea un cambio significativo, genera un reporte `drift_report.json`.
4.  **`branch_on_drift`**: Tarea de decisión lógica. Lee el reporte JSON:
    * Si `global_drift == True`: Desvía el flujo hacia `retrain_task`.
    * Si `global_drift == False`: Desvía el flujo hacia `skip_retrain`.
5.  **`retrain_task`**:  realiza un proceso de optimización de los hiperparámetros con **Optuna** en un nuevo modelo **XGBoost** con los datos más recientes y registra métricas/artefactos en MLflow.
6.  **`predict`**: Utiliza el mejor modelo disponible ya sea el nuevo entrenado o el anterior para generar predicciones de compra para la "próxima semana".

## 2. Diagrama de Flujo
El flujo lógico del codigo es el siguiente:
`Extract` -> `Preprocess` -> `Detect Drift` -> `Branching` -> (`Retrain` O `Skip`) -> `Predict` -> `End`

## 3. Lógica de Reentrenamiento y Drift
Para simular un entorno productivo robusto:
* **Detección de Drift:** Se implementó un script que calcula el *Population Stability Index* (PSI) y pruebas Kolmogorov-Smirnov en las variables numéricas. Si más del 5% de las variables presentan drift, se activa una alerta.
* **Reentrenamiento:** Solo se usa recursos computacionales entrenando cuando los datos lo requiere. Usamos MLflow para asegurar que siempre quede guardado el mejor modelo en `model_best.joblib` accesible a la API.