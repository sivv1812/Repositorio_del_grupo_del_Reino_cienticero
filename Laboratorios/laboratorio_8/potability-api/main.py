import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Estableciendo ruta del modelo creado
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
model = None  # Inicializa la variable global

# definiendo los input de la pagina web
class WaterFeatures(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# iniciando app
app = FastAPI(
    title="API de Predicción de Potabilidad del Agua",
    description="Predice si una muestra de agua es potable o no",
    version="1.0.0"
)

# Cargando el modelo en la app
def load_model():
    global model
    print(f" Directorio actual: {os.getcwd()}")
    print(f" Contenido de 'models': {os.listdir(MODEL_DIR)}")
    print(f" Intentando cargar modelo desde: {MODEL_PATH}")

    try:
        if not os.path.exists(MODEL_PATH):
            print(f" No se encontró '{MODEL_PATH}'. Buscando alternativos...")
            for f in os.listdir(MODEL_DIR):
                if f.endswith(".pkl"):
                    alt_path = os.path.join(MODEL_DIR, f)
                    print(f" Intentando cargar {alt_path}")
                    model = joblib.load(alt_path)
                    print(f" Modelo '{f}' cargado exitosamente.")
                    return
            print(" Ningún modelo válido fue cargado.")
            model = None
        else:
            model = joblib.load(MODEL_PATH)
            print(f" Modelo '{MODEL_PATH}' cargado correctamente.")
    except Exception as e:
        print(f" Error al cargar el modelo: {e}")
        model = None

#cargando modelo al inicio de la app
@app.on_event("startup")
def startup_event():
    load_model()

# mostrando en la pagina lo pedido en el enunciado
@app.get("/")
async def home():
    return {"modelo": "Clasificador de Potabilidad del Agua", 
            "problema": "Predice si una muestra de agua es potable (1) o no (0) basado en 9 características físico-químicas.", 
            "entrada": "Un objeto JSON con las 9 características (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity).",
              "salida": "Un objeto JSON con la predicción: {'potabilidad': 0} o {'potabilidad': 1}."}

#aplicando modelo segun datos proporcionados por el usuario
@app.post("/potabilidad/")
async def predecir_potabilidad(data: WaterFeatures):
    if model is None:
        return {"detail": "El modelo no está cargado en el servidor."}

    try:
        input_df = pd.DataFrame([data.dict()])
        pred = model.predict(input_df)
        return {"potabilidad": int(pred[0])}
    except Exception as e:
        return {"error": f"Error durante la predicción: {str(e)}"}

# ejecución local del sistema
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

