# agente_preprocesamiento.py

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing_extensions import Annotated, Any
from langchain.tools import tool, Tool
import pickle
# --- Modelos de Pydantic para los argumentos de las herramientas ---
from pydantic import BaseModel, Field

# Cargar claves de API desde el archivo .env
load_dotenv()

class PreprocesarInput(BaseModel):
    ruta_csv: str = Field(description="Ruta al archivo CSV de entrada que se va a preprocesar.")
    ruta_salida_datos: str = Field(description="Ruta donde se guardará el archivo pickle con las features (X) procesadas.")
    ruta_salida_target: str = Field(description="Ruta donde se guardará el archivo pickle con la variable target (y).")

@tool(args_schema=PreprocesarInput)
def analizar_y_preprocesar_datos(ruta_csv: str, ruta_salida_datos: str, ruta_salida_target: str) -> str:
    """
    Analiza un archivo CSV, realiza un preprocesamiento completo y guarda los datos procesados.
    Pasos:
    1.  Lee el CSV.
    2.  Determina el tipo de tarea (clasificación/regresión) basado en la columna 'target'.
    3.  Maneja valores nulos: descarta columnas con más de 50% de nulos e imputa el resto.
    4.  Codifica variables categóricas: usa One-Hot Encoding para baja cardinalidad y Label Encoding para alta cardinalidad.
    5.  Guarda el DataFrame de features (X) y la serie del target (y) en archivos pickle separados.
    """
    print(f"--- Herramienta de Preprocesamiento: Analizando y limpiando '{ruta_csv}' ---")
    try:
        df = pd.read_csv(ruta_csv)

        if "target" not in df.columns:
            return "Error: El archivo CSV no contiene una columna 'target'."

        # Separar target y features iniciales
        y = df['target']
        columnas_llave = ['codmes', 'key_id']
        features_df = df.drop(columns=[col for col in ['target'] + columnas_llave if col in df.columns])

        # --- 1. Manejo de Nulos (Fillrate) ---
        print("Analizando fillrate y manejando valores nulos...")
        umbral_nulos = 0.5
        nulos_porc = features_df.isnull().sum() / len(features_df)
        columnas_a_descartar = nulos_porc[nulos_porc > umbral_nulos].index
        features_df.drop(columns=columnas_a_descartar, inplace=True)
        print(f"Columnas descartadas por exceso de nulos: {list(columnas_a_descartar)}")

        # --- 2. Imputación de Nulos restantes ---
        columnas_numericas = features_df.select_dtypes(include=np.number).columns
        columnas_categoricas = features_df.select_dtypes(exclude=np.number).columns

        imputer_num = SimpleImputer(strategy='mean')
        imputer_cat = SimpleImputer(strategy='most_frequent')

        if not columnas_numericas.empty:
            features_df[columnas_numericas] = imputer_num.fit_transform(features_df[columnas_numericas])
        if not columnas_categoricas.empty:
            features_df[columnas_categoricas] = imputer_cat.fit_transform(features_df[columnas_categoricas])
        
        print("Imputación de nulos completada.")

        # --- 3. Codificación de Categóricas ---
        print("Codificando variables categóricas...")
        columnas_categoricas = features_df.select_dtypes(include=['object', 'category']).columns
        
        if not columnas_categoricas.empty:
            # Decidir estrategia de codificación
            umbral_cardinalidad = 10
            one_hot_cols = [col for col in columnas_categoricas if features_df[col].nunique() < umbral_cardinalidad]
            label_enc_cols = [col for col in columnas_categoricas if col not in one_hot_cols]

            print(f"Columnas para One-Hot Encoding: {one_hot_cols}")
            print(f"Columnas para Label Encoding: {label_enc_cols}")

            # Aplicar Label Encoding
            for col in label_enc_cols:
                le = LabelEncoder()
                features_df[col] = le.fit_transform(features_df[col])

            # Aplicar One-Hot Encoding
            if one_hot_cols:
                features_df = pd.get_dummies(features_df, columns=one_hot_cols, drop_first=True)
        
        print("Codificación completada. Todas las features son numéricas.")

        # --- 4. Guardar resultados ---
        with open(ruta_salida_datos, 'wb') as f:
            pickle.dump(features_df, f)
        with open(ruta_salida_target, 'wb') as f:
            pickle.dump(y, f)

        # --- 5. Determinar tipo de tarea para el informe ---
        target_dtype = y.dtype
        unique_values = y.nunique()
        if pd.api.types.is_numeric_dtype(target_dtype) and unique_values > 20:
            tipo_tarea = "regresión"
        else:
            tipo_tarea = "clasificación"

        return (f"Preprocesamiento completado exitosamente. "
                f"Se detectó una tarea de '{tipo_tarea}'. "
                f"Los datos procesados se guardaron en '{ruta_salida_datos}' "
                f"y el target en '{ruta_salida_target}'. "
                f"Features finales: {list(features_df.columns)}")

    except FileNotFoundError:
        return f"Error: No se encontró el archivo en la ruta '{ruta_csv}'."
    except Exception as e:
        return f"Error inesperado durante el preprocesamiento: {e}"

# (El resto del código para crear el agente y el main() se eliminará de este archivo,
# ya que la orquestación se hará en `orquestador_ml.py`)