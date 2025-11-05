# 1. Imagen base oficial de Apache Airflow
FROM apache/airflow:2.8.1

# 2. Cambiar a root temporalmente para instalar 'curl'
USER root
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. REGRESAR AL USUARIO 'airflow' (¡Este es el arreglo!)
USER airflow

# 4. Copiar e instalar librerías de Python (como 'airflow' user)
COPY requirements.txt ${AIRFLOW_HOME}/requirements.txt
RUN pip install --no-cache-dir -r ${AIRFLOW_HOME}/requirements.txt

# 5. Copiar tu proyecto (ya estamos como 'airflow')
COPY dags ${AIRFLOW_HOME}/dags
COPY plugins ${AIRFLOW_HOME}/plugins
RUN mkdir -p ${AIRFLOW_HOME}/models_registry

# 6. Exponer puertos
EXPOSE 8080 7860

# 7. Comando de inicio (se ejecuta como 'airflow')
CMD ["airflow", "standalone"]
