# Usamos la imagen oficial de Airflow
FROM apache/airflow:2.8.1

# Copiamos el archivo de requisitos
COPY requirements.txt /requirements.txt

# Instalamos las librerías necesarias para que corran tus scripts
RUN pip install --no-cache-dir -r /requirements.txt