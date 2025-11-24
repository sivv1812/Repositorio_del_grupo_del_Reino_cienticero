# scripts/extract.py
import os
import pandas as pd

DATA_DIR = os.getenv('DATA_DIR', '/opt/airflow/data')
#prepara los datos crudos y los guarda para etapas siguientes
def run():
    tx = pd.read_parquet(os.path.join(DATA_DIR, 'transacciones.parquet'))
    clients = pd.read_parquet(os.path.join(DATA_DIR, 'clientes.parquet'))
    products = pd.read_parquet(os.path.join(DATA_DIR, 'productos.parquet'))
    tx.to_parquet(os.path.join(DATA_DIR, 'tx_raw.parquet'), index=False)
    clients.to_parquet(os.path.join(DATA_DIR, 'clients_raw.parquet'), index=False)
    products.to_parquet(os.path.join(DATA_DIR, 'products_raw.parquet'), index=False)
    print("Extraction finished")

if __name__ == '__main__':
    run()
