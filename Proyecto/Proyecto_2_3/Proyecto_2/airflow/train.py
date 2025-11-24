import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar dataset de ejemplo
df = pd.read_csv("/opt/airflow/data/dataset.csv")   # pon tu dataset real aquí

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():

    # Modelo simple
    model = RandomForestClassifier(n_estimators=200, max_depth=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log params
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")
