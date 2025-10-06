# generar_datos.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

print("Iniciando la generación del archivo CSV para clasificación...")

# --- Parámetros de Generación ---
NUM_FILAS = 100000
NOMBRE_ARCHIVO = 'datos_clasificacion.csv'

# --- Generación de Columnas ---

# 1. key_id: Un identificador único para cada fila
key_ids = range(1, NUM_FILAS + 1)

# 2. codmes: Fechas en formato YYYYMM, distribuidas en los últimos 2 años
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 6, 30)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]
codmes_list = [int(d.strftime('%Y%m')) for d in np.random.choice(date_range, size=NUM_FILAS)]

# 3. target: Variable objetivo binaria (0 o 1) con un ligero desbalance
#    Vamos a hacer que la clase 1 sea un 15% del total para que sea más realista
targets = np.random.choice([0, 1], size=NUM_FILAS, p=[0.85, 0.15])

# 4. Variables continuas (5 columnas)
cont_var_1 = np.random.normal(loc=50, scale=15, size=NUM_FILAS)
cont_var_2 = np.random.uniform(low=100, high=1000, size=NUM_FILAS)
cont_var_3 = np.random.gamma(shape=2, scale=2, size=NUM_FILAS)
cont_var_4 = cont_var_1 * 0.5 + np.random.normal(0, 5, NUM_FILAS) # Correlacionada con la var 1
cont_var_5 = np.random.rand(NUM_FILAS) * 100 # De 0 a 100

# 5. Variables categóricas (4 columnas)
categorias_region = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']
cat_var_1_region = np.random.choice(categorias_region, size=NUM_FILAS, p=[0.2, 0.3, 0.15, 0.15, 0.2])

categorias_producto = ['A', 'B', 'C', 'D']
cat_var_2_producto = np.random.choice(categorias_producto, size=NUM_FILAS)

categorias_canal = ['Online', 'Tienda', 'Telefono']
cat_var_3_canal = np.random.choice(categorias_canal, size=NUM_FILAS, p=[0.6, 0.3, 0.1])

cat_var_4_booleano = np.random.choice(['Si', 'No'], size=NUM_FILAS)

# --- Creación del DataFrame ---

df_dict = {
    'key_id': key_ids,
    'codmes': codmes_list,
    'cont_var_1': cont_var_1,
    'cont_var_2': cont_var_2,
    'cont_var_3': cont_var_3,
    'cont_var_4': cont_var_4,
    'cont_var_5': cont_var_5,
    'cat_var_1_region': cat_var_1_region,
    'cat_var_2_producto': cat_var_2_producto,
    'cat_var_3_canal': cat_var_3_canal,
    'cat_var_4_booleano': cat_var_4_booleano,
    'target': targets
}

df = pd.DataFrame(df_dict)

# --- Guardado del Archivo ---
try:
    df.to_csv(NOMBRE_ARCHIVO, index=False, encoding='utf-8')
    print(f"¡Éxito! Se ha creado el archivo '{NOMBRE_ARCHIVO}' con {NUM_FILAS} filas y {len(df.columns)} columnas.")
    print(f"Puedes encontrarlo en la misma carpeta donde ejecutaste este script.")
except Exception as e:
    print(f"Error al guardar el archivo: {e}")