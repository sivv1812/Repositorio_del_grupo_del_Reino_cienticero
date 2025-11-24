import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="SodAI Prediction API")

# Ruta donde buscaremos el modelo (montada desde Airflow)
DATA_DIR = os.getenv('DATA_DIR', '/app/data')
MODEL_PATH = os.path.join(DATA_DIR, 'model_best.joblib')

model = None

# Definimos los datos que esperamos recibir (Inputs)
class PredictionInput(BaseModel):
    num_deliver_per_week: float
    num_visit_per_week: float
    customer_type: str
    size: str
    category: str
    brand: str

@app.on_event("startup")
def load_model():
    global model
    try:
        print(f"Buscando modelo en: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ Modelo cargado exitosamente.")
        else:
            print("⚠️ No se encontró el archivo del modelo. Asegúrate de que Airflow haya terminado.")
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(input_data: PredictionInput):
    if not model:
        raise HTTPException(status_code=503, detail="El modelo no está cargado aún.")
    
    try:
        # 1. Convertir el input a DataFrame
        data = input_data.dict()
        df = pd.DataFrame([data])
        
        # 2. Preprocesamiento (Convertir texto a dummies)
        df_processed = pd.get_dummies(df)
        
        # 3. Alinear columnas con el modelo (Truco para evitar errores de shape)
        # Obtenemos las columnas que el modelo espera
        model_cols = model.get_booster().feature_names
        
        # Creamos un DF vacío con esas columnas
        df_final = pd.DataFrame(columns=model_cols)
        
        # Llenamos con los datos que tenemos
        for col in df_processed.columns:
            if col in df_final.columns:
                df_final[col] = df_processed[col]
        
        # Rellenamos con 0 lo que falte (ej: si no eligió 'brand_B', esa columna es 0)
        df_final = df_final.fillna(0)
        
        # Aseguramos que sean números
        df_final = df_final[model_cols].astype(float)

        # 4. Predecir
        prediction = model.predict(df_final)[0]
        
        # Intentar obtener probabilidad si el modelo lo permite
        try:
            prob = model.predict_proba(df_final)[0][1]
        except:
            prob = 0.0
        
        return {
            "prediction": int(prediction),
            "probability": float(prob),
            "message": "CLIENTE COMPRARÁ 💰" if prediction == 1 else "No comprará ❌"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)