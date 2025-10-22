
# generando Imagen base ligera
FROM python:3.10-slim

# estableciendo Directorio de trabajo dentro del contenedor
WORKDIR /app

#  Copiando e instalando dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copiando archivos de la aplicación al docker
COPY main.py .
COPY models/ models/

# Exponiendo el puerto del servidor en el puerto 8000
EXPOSE 8000

# Ejecutando FastAPI usando Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

