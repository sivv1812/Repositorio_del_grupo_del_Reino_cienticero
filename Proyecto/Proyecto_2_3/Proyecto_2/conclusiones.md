# Conclusiones del Proyecto MLOps - SodAI Drinks

# Aprendizajes y Desafíos
La implementación de un flujo nos permitió entender que el modelado en este caso xgboost como machine learning  es solo una pequeña parte de un sistema en producción, o de un dashboard.

1.  **El valor del Tracking:**
    * Centralizar las métricas y parámetros con MLflow fue fundamental. Antes, perder la configuración de un "buen modelo" era fácil. Ahora, cada ejecución del pipeline como la tarea `retrain_task` deja un registro histórico, permitiendo ver qué hiperparámetros genero el modelo actual.

2.  **Orquestación con Airflow:**
    * Airflow aporta robustez al manejar las dependencias. LA mayor complicación fue la comunicación entre tareas, en otras palabras, trasladar archivos de una tarea a otra. Solucionamos esto usando volúmenes compartidos de Docker.
    * La implementación del *Branching*  nos permite ahorrar recursos, permite al sistema  no reentrenar si los datos son estables.

3.  **Despliegue Web (FastAPI + Gradio):**
    * **Desafío Técnico:** Uno de los puntos más interesantes fue manejar la diferencia entre el entrenamiento donde se presenta con muchas variables históricas y la inferencia en la web donde el usuario ingresa los datos.
    * **Solución:** Implementar una estrategia de imputación en el Backend, donde se imputo las variables faltantes con ceros. Esto genera que el modelo tienda a predecir probabilidades alrededor del 12% para usuarios promedio, pero reacciona drásticamente bajando a 0% con variables logísticas como "visitas" o "entregas" son bajas, lo cual tiene sentido de negocio de las bebidas.

4.  **Dockerización:**
    * implementar un docker en el  proyecto garantizó el funcionamiento del codigo usado en cualquier computador. Tuvimos desafíos con permisos de carpetas sobre todo con MLflow, que solucionamos ajustando las rutas absolutas y los volúmenes en `docker-compose`.

## Mejoras Futuras
* Implementar un sistema de alertas por medio de email a los clientes sobre cuales son las predicciones.