import os
import pandas as pd
import joblib
import numpy as np

DATA_DIR = os.getenv('DATA_DIR', '/opt/airflow/data')
MODEL_PATH = os.path.join(DATA_DIR, 'model_best.joblib')

def run():
    print("Iniciando predicción...")
    try:
        # Cargar datos
        df_path = os.path.join(DATA_DIR, 'df_prepared.parquet')
        if not os.path.exists(df_path):
            print("No se encontró dataset preparado.")
            return
            
        df = pd.read_parquet(df_path)
        
        # Tomamos una muestra para simular la predicción 
        input_data = df.sample(frac=0.1)
        
        # Preprocesar igual que en el entrenamiento 
        features = input_data.drop(columns=['items', 'y_cat', 'purchase_date', 'date'], errors='ignore')
        features = pd.get_dummies(features, drop_first=True)
        
        # Cargar modelo
        if not os.path.exists(MODEL_PATH):
            print("No hay modelo entrenado localmente.")
            return
            
        model = joblib.load(MODEL_PATH)
        
        # Alinear columnas (rellenar con 0 las que falten)
        model_cols = model.get_booster().feature_names if hasattr(model, 'get_booster') else features.columns
        for col in model_cols:
            if col not in features.columns:
                features[col] = 0
        features = features[list(model_cols)] 

        # Predecir
        preds = model.predict(features)
        
        # Guardar
        output = pd.DataFrame({
            'prediction': preds
        })
        
        output_path = os.path.join(DATA_DIR, 'predictions.csv')
        output.to_csv(output_path, index=False)
        print(f"Predicciones guardadas en {output_path}")
        
    except Exception as e:
        print(f"Error critico en prediccion: {e}")
        raise e

if __name__ == '__main__':
    run()