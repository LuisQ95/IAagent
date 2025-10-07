# agente_modelado.py

import os
import numpy as np
import pandas as pd
import pickle
from typing_extensions import Annotated, Union, Dict, Any
import ast
import json
from dotenv import load_dotenv

# --- Modelos, Métricas y Tuning ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss,
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)
import xgboost as xgb
import lightgbm as lgb

from langchain.tools import tool, Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv()

# --- Modelos de Pydantic para los argumentos de las herramientas ---
from pydantic import BaseModel, Field
class OptimizarEntrenarInput(BaseModel):
    ruta_datos_procesados: str = Field(description="Ruta al archivo pickle que contiene las features (X) procesadas.")
    ruta_target: str = Field(description="Ruta al archivo pickle que contiene la variable target (y).")
    nombre_modelo: str = Field(description="Nombre del modelo a entrenar. Opciones: 'RandomForest', 'XGBoost', 'LightGBM'.")
    ruta_salida_resultados: str = Field(description="Ruta del archivo JSON donde se guardarán las métricas y resultados del modelo.")

def investigar_hiperparametros(nombre_modelo: str, tipo_tarea: str) -> dict:
    """
    Utiliza un agente de IA con acceso a búsqueda web para encontrar los mejores
    hiperparámetros y rangos de valores para un modelo y tarea específicos.
    """
    print(f"--- Iniciando agente de investigación para hiperparámetros de {nombre_modelo} ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    tools = load_tools(["serpapi"], llm=llm) # Herramienta de búsqueda web

    # Construir un prompt detallado para el agente
    prompt = (
        f"Busca los hiperparámetros más importantes y rangos de valores comunes para un modelo '{nombre_modelo}' "
        f"en una tarea de '{tipo_tarea}'. Devuelve la respuesta únicamente como un string de un diccionario Python "
        f"apto para usar en GridSearchCV. Ejemplo de formato de salida: {{{{ 'n_estimators': [100, 200], 'max_depth': [10, 20] }}}}. "
        "No incluyas nada más en tu respuesta, solo el diccionario como texto. Asegúrate de que las claves y los valores de tipo string en el diccionario estén entre comillas simples."
    )

    # Usar el método moderno para crear el agente, eliminando el DeprecationWarning
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, chat_prompt)
    investigador = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Usar .invoke() en lugar de .run(), eliminando el DeprecationWarning
    respuesta = investigador.invoke({"input": f"hiperparámetros para {nombre_modelo}"})
    respuesta_texto = respuesta.get("output", "{}")
    print(f"Respuesta del investigador: {respuesta_texto}")

    # Limpiar el string de la respuesta si viene en un bloque de código markdown
    cleaned_text = respuesta_texto.strip()
    if "```python" in cleaned_text:
        # Extrae el contenido entre ```python y ```
        cleaned_text = cleaned_text.split("```python")[1].split("```")[0]
    elif cleaned_text.startswith("```"):
        # Para casos donde solo hay ``` sin 'python'
        cleaned_text = cleaned_text.strip("` \n")

    # Convertir el string de la respuesta a un diccionario real
    try:
        # Usar ast.literal_eval en el texto limpio
        return ast.literal_eval(cleaned_text.strip())
    except Exception as e:
        print(f"Error al parsear la respuesta del investigador: {e}. Usando parámetros por defecto.")
        # Fallback a parámetros por defecto si la respuesta no es válida
        return {"n_estimators": [50, 100], "max_depth": [10, 20]}

@tool(args_schema=OptimizarEntrenarInput)
def optimizar_y_entrenar_modelo(ruta_datos_procesados: str, ruta_target: str, nombre_modelo: str, ruta_salida_resultados: str) -> str:
    """
    Carga datos preprocesados, optimiza hiperparámetros de un modelo, lo entrena y evalúa.
    Pasos:
    1. Carga los datos de features (X) y el target (y) desde archivos pickle.
    2. Determina el tipo de tarea (clasificación/regresión).
    3. Define un espacio de búsqueda de hiperparámetros para el modelo especificado.
    4. Utiliza GridSearchCV para encontrar los mejores hiperparámetros.
    5. Entrena el modelo final con los mejores parámetros.
    6. Calcula métricas de evaluación y la importancia de variables.
    7. Guarda los resultados (métricas e importancias) en un archivo JSON.
    """
    print(f"--- Herramienta de Modelado: Optimizando y entrenando '{nombre_modelo}' ---")
    try:
        # 1. Cargar datos
        with open(ruta_datos_procesados, 'rb') as f:
            X = pickle.load(f)
        with open(ruta_target, 'rb') as f:
            y = pickle.load(f)

        # 2. Determinar tipo de tarea
        tipo_tarea = "regression" if y.dtype == 'float64' and y.nunique() > 20 else "classification"
        print(f"Tarea detectada: {tipo_tarea}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelos = {
            "classification": {
                "RandomForest": RandomForestClassifier(random_state=42),
                "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                "LightGBM": lgb.LGBMClassifier(random_state=42)
            },
            "regression": {
                "RandomForest": RandomForestRegressor(random_state=42),
                "XGBoost": xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
                "LightGBM": lgb.LGBMRegressor(random_state=42)
            }
        }

        if nombre_modelo not in modelos[tipo_tarea]:
            return f"Error: Modelo '{nombre_modelo}' no válido para {tipo_tarea}."

        # 3. Investigar dinámicamente los hiperparámetros
        print(f"Buscando dinámicamente los mejores hiperparámetros para {nombre_modelo}...")
        param_grid = investigar_hiperparametros(nombre_modelo, tipo_tarea)

        # 4. Optimización con GridSearchCV
        print(f"Iniciando GridSearchCV para {nombre_modelo}...")
        grid_search = GridSearchCV(modelos[tipo_tarea][nombre_modelo], param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

        # 5. Evaluación
        y_pred = best_model.predict(X_test)
        metricas = {}
        if tipo_tarea == "classification":
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            metricas['Accuracy'] = accuracy_score(y_test, y_pred)
            metricas['F1-Score'] = f1_score(y_test, y_pred, average='weighted')
            metricas['AUC'] = roc_auc_score(y_test, y_pred_proba)
            metricas['GINI'] = 2 * metricas['AUC'] - 1
        else: # regression
            metricas['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
            metricas['MAE'] = mean_absolute_error(y_test, y_pred)

        # 6. Importancia de variables
        importancias = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        # 7. Guardar resultados en JSON de forma acumulativa
        resultados = {
            "nombre_modelo": nombre_modelo,
            "tipo_tarea": tipo_tarea,
            "mejores_parametros": grid_search.best_params_,
            "metricas": {k: round(v, 4) for k, v in metricas.items()},
            "importancia_variables": importancias.to_dict('records')
        }

        # Cargar resultados existentes o crear un nuevo diccionario si el archivo no existe
        try:
            with open(ruta_salida_resultados, 'r') as f:
                datos_existentes = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            datos_existentes = {} # Si el archivo no existe o está vacío, empezamos de cero

        # Añadir o sobreescribir los resultados del modelo actual
        datos_existentes[nombre_modelo] = resultados

        with open(ruta_salida_resultados, 'w') as f:
            json.dump(datos_existentes, f, indent=4)

        # Formatear respuesta para el agente
        respuesta_str = f"Modelo '{nombre_modelo}' optimizado y entrenado.\n"
        respuesta_str += f"Métricas: {json.dumps(metricas, indent=2)}\n"
        respuesta_str += f"Resultados completos guardados en '{ruta_salida_resultados}'."
        
        return respuesta_str

    except Exception as e:
        return f"Error inesperado durante el modelado: {e}"