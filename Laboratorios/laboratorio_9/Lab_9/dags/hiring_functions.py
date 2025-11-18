import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib
import gradio as gr

BASE_DIR = os.getenv("AIRFLOW_HOME", "/opt/airflow")



# ============================================
# 1️⃣ CREAR CARPETAS
# ============================================
def create_folders(**kwargs):
    ds = kwargs.get("ds", "default_run")
    base_dir = f"./{ds}"

    for subfolder in ["raw", "processed", "models_registry"]:
        os.makedirs(f"{base_dir}/{subfolder}", exist_ok=True)

    print(f"✅ Carpetas creadas en {base_dir}")


# ============================================
# 2️⃣ DIVIDIR LOS DATOS (TRAIN / TEST)
# ============================================
def split_data(**kwargs):
    ds = kwargs.get("ds", "manual_run")
    raw_path = f"./{ds}/raw/data_1.csv"
    processed_dir = f"./{ds}/processed"
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {raw_path}")

    print(f"📂 Leyendo datos desde: {raw_path}")
    df = pd.read_csv(raw_path)

    # División en entrenamiento y prueba
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Guardamos los conjuntos
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"✅ Datos divididos correctamente:")
    print(f"   - Entrenamiento: {train_path} ({len(train)} filas)")
    print(f"   - Prueba: {test_path} ({len(test)} filas)")


# ============================================
# 3️⃣ ENTRENAR EL MODELO (versión robusta)
# ============================================
def preprocess_and_train(**kwargs):
    ds = kwargs.get("ds", "manual_run")
    base_dir = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    train_path = os.path.join(base_dir, ds, "processed", "train.csv")

    print(f"🤖 Entrenando modelo con datos: {train_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo de entrenamiento en {train_path}")

    # Cargar datos
    df = pd.read_csv(train_path)
    print(f"✅ Datos cargados correctamente. Filas: {len(df)}, Columnas: {list(df.columns)}")

    if "Salary" not in df.columns:
        raise ValueError("❌ No se encontró la columna objetivo 'Salary' en el dataset.")

    # Eliminar filas con valores faltantes
    df = df.dropna()

    # Seleccionamos solo columnas numéricas
    X = df.drop(columns=["Salary"]).select_dtypes(include=["number"])
    y = df["Salary"]

    if X.empty:
        raise ValueError("❌ No se encontraron variables numéricas para entrenar el modelo.")

    print(f"📊 Variables predictoras: {list(X.columns)}")

    # Crear pipeline de escalado + modelo lineal
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    # Entrenamiento
    model.fit(X, y)
    print("✅ Modelo entrenado correctamente.")

    # Guardar modelo
    model_dir = os.path.join(base_dir, "models_registry")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "latest_model.joblib")
    joblib.dump(model, model_path)

    print(f"💾 Modelo guardado en: {model_path}")
    print(f"📈 Coeficientes: {model.named_steps['regressor'].coef_}")
    print(f"Intercepto: {model.named_steps['regressor'].intercept_}")



# ============================================
# 4️⃣ INTERFAZ GRADIO
# ============================================
def gradio_interface():
    model_path = "./models_registry/latest_model.joblib"

    if not os.path.exists(model_path):
        print("⚠️ No se encontró el modelo entrenado. Ejecuta primero el pipeline.")
        return

    model = joblib.load(model_path)
    print("✅ Modelo cargado en memoria.")

    def predict_from_input(**inputs):
        df = pd.DataFrame([inputs])
        pred = model.predict(df)[0]
        return round(pred, 2)

    # Detecta las columnas del modelo (excluyendo Salary)
    example_input = model.feature_names_in_ if hasattr(model, "feature_names_in_") else []

    if not len(example_input):
        print("⚠️ No se pudieron detectar las columnas del modelo automáticamente.")
        return

    # Crear los campos de entrada en Gradio
    inputs = [gr.Number(label=col) for col in example_input]
    output = gr.Number(label="Predicted Salary")

    iface = gr.Interface(
        fn=lambda *args: predict_from_input(**dict(zip(example_input, args))),
        inputs=inputs,
        outputs=output,
        title="Predicción de Salario (Modelo Lineal)",
        description="Introduce los valores de las variables para obtener una predicción de salario."
    )

    print("🚀 Lanzando interfaz de Gradio...")
    iface.launch(share=True)

