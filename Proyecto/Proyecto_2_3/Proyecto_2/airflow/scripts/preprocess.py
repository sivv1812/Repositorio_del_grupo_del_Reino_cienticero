# scripts/preprocess.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.getenv('DATA_DIR', '/opt/airflow/data')

def run():
    tx = pd.read_parquet(os.path.join(DATA_DIR, 'tx_raw.parquet'))
    clients = pd.read_parquet(os.path.join(DATA_DIR, 'clients_raw.parquet'))
    products = pd.read_parquet(os.path.join(DATA_DIR, 'products_raw.parquet'))

    # limpiando datos
    tx = tx.drop_duplicates().dropna(subset=['purchase_date'])
    tx['purchase_date'] = pd.to_datetime(tx['purchase_date'])
    # crear features productos y clientes 
    products = products.drop_duplicates().dropna()
    clients = clients.drop_duplicates().dropna()

    # Guarda el dataset final preparado
    df_join = tx.merge(clients[['customer_id','Y','X','num_deliver_per_week','num_visit_per_week','customer_type']], on='customer_id', how='inner')
    df_join = df_join.merge(products[['product_id','size','category','brand']], on='product_id', how='inner')

    df_join.to_parquet(os.path.join(DATA_DIR, 'df_prepared.parquet'), index=False)
    print("Preprocessing finished")

if __name__ == '__main__':
    run()
