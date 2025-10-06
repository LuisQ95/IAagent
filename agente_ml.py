# agente_ml.py

import os
import pandas as pd
from dotenv import load_dotenv

# --- Modelos y Métricas de Scikit-learn y otras librerías ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, log_loss, # Clasificación
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error # Regresión
)
import xgboost as xgb
import lightgbm as lgb
import numpy as np

# --- Componentes de LangChain ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool

# --- Configuración Inicial ---

# Cargar claves de API desde el archivo .env
load_dotenv()

# --- Herramientas Personalizadas para el Agente de ML ---

@tool
def analizar_fuente_de_datos(ruta_csv: str) -> str:
    """
    Analiza un archivo CSV para determinar si el problema es de clasificación o regresión
    basado en la columna 'target'. También identifica las columnas de features.
    La función devuelve un string con el tipo de tarea ('classification' o 'regression')
    y la lista de columnas a usar como features.
    """
    print(f"--- Herramienta: Analizando datos de '{ruta_csv}' ---")
    try:
        df = pd.read_csv(ruta_csv)

        if "target" not in df.columns:
            return "Error: El archivo CSV no contiene una columna llamada 'target'."

        # Identificar columnas de llaves a excluir
        columnas_llave = ['codmes', 'key_id']
        features = [col for col in df.columns if col not in ['target'] + columnas_llave]

        # Lógica para discernir el tipo de tarea
        target_dtype = df['target'].dtype
        unique_values = df['target'].nunique()

        if pd.api.types.is_numeric_dtype(target_dtype):
            # Si es numérico, verificamos si parece categórico (pocos valores únicos enteros) o continuo
            if unique_values <= 20 and all(df['target'].dropna().apply(lambda x: x == int(x))):
                tipo_tarea = "classification"
                print(f"Detectado problema de CLASIFICACIÓN (target con {unique_values} valores únicos enteros).")
            else:
                tipo_tarea = "regression"
                print(f"Detectado problema de REGRESIÓN (target numérico continuo).")
        else:
            # Si no es numérico (ej. 'object'), es clasificación
            tipo_tarea = "classification"
            print(f"Detectado problema de CLASIFICACIÓN (target de tipo {target_dtype}).")

        return f"Análisis completado: El tipo de tarea es '{tipo_tarea}'. Las features a utilizar son: {features}"

    except FileNotFoundError:
        return f"Error: No se encontró el archivo en la ruta '{ruta_csv}'."
    except Exception as e:
        return f"Error inesperado durante el análisis de datos: {e}"

@tool
def entrenar_y_evaluar_modelo(ruta_csv: str, nombre_modelo: str) -> str:
    """
    Entrena y evalúa un modelo de Machine Learning. Primero, analiza los datos para
    determinar el tipo de tarea (clasificación/regresión), luego entrena el modelo
    especificado (opciones: 'XGBoost', 'LightGBM', 'RandomForest'), calcula las métricas
    apropiadas y devuelve la importancia de las variables.
    """
    print(f"--- Herramienta: Entrenando modelo '{nombre_modelo}' con datos de '{ruta_csv}' ---")
    
    # 1. Analizar los datos para obtener el contexto
    analisis_str = analizar_fuente_de_datos.run(ruta_csv)
    if "Error" in analisis_str:
        return analisis_str
        
    # Extraer tipo de tarea y features del string de análisis
    try:
        parts = analisis_str.split("'")
        tipo_tarea = parts[1]
        features_str = parts[3]
        features = [f.strip() for f in features_str.strip('[]').split(',')]
    except IndexError:
        return f"Error: No se pudo interpretar el resultado del análisis: {analisis_str}"

    # 2. Cargar y preparar datos
    df = pd.read_csv(ruta_csv)
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Seleccionar y entrenar el modelo
    modelos = {
        "classification": {
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": lgb.LGBMClassifier(random_state=42)
        },
        "regression": {
            "RandomForest": RandomForestRegressor(random_state=42),
            "XGBoost": xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
            "LightGBM": lgb.LGBMRegressor(random_state=42)
        }
    }

    if nombre_modelo not in modelos[tipo_tarea]:
        return f"Error: El modelo '{nombre_modelo}' no es válido para {tipo_tarea}. Opciones: {list(modelos[tipo_tarea].keys())}"

    modelo = modelos[tipo_tarea][nombre_modelo]
    print(f"Entrenando {nombre_modelo} para {tipo_tarea}...")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # 4. Evaluar el modelo con las métricas correctas
    metricas = {}
    if tipo_tarea == "classification":
        y_pred_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else None
        metricas['Accuracy'] = accuracy_score(y_test, y_pred)
        metricas['F1-Score'] = f1_score(y_test, y_pred, average='weighted')
        if y_pred_proba is not None:
            metricas['AUC'] = roc_auc_score(y_test, y_pred_proba)
            metricas['LogLoss'] = log_loss(y_test, modelo.predict_proba(X_test))
        # GINI = 2 * AUC - 1
        if 'AUC' in metricas:
            metricas['GINI'] = 2 * metricas['AUC'] - 1
    else: # regression
        metricas['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metricas['MAE'] = mean_absolute_error(y_test, y_pred)
        metricas['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)

    # 5. Obtener importancia de variables
    importancias = pd.DataFrame({
        'feature': features,
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    # 6. Formatear la respuesta
    respuesta_final = f"Modelo '{nombre_modelo}' entrenado exitosamente.\n"
    respuesta_final += f"Tipo de Tarea: {tipo_tarea}\n\n"
    respuesta_final += "--- Métricas de Validación ---\n"
    for k, v in metricas.items():
        respuesta_final += f"- {k}: {v:.4f}\n"
    respuesta_final += "\n--- Top 10 Variables más Importantes ---\n"
    respuesta_final += importancias.to_string(index=False)

    return respuesta_final


def crear_agente_ml():
    """
    Configura y crea una instancia del agente de Machine Learning.
    """
    print("Inicializando el modelo Gemini-Pro para el agente de ML...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1, # Más determinista para seguir instrucciones
        convert_system_message_to_human=True # Necesario para este tipo de agente
    )

    # Lista de herramientas personalizadas que el agente puede usar
    tools = [analizar_fuente_de_datos, entrenar_y_evaluar_modelo]

    print("Creando el agente de Machine Learning...")
    # Usamos un agente conversacional que puede recordar interacciones y usar herramientas
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors="Intenta de nuevo. Asegúrate de usar el formato correcto."
    )
    print("¡Agente de ML listo!")
    return agent_executor

def main():
    """
    Función principal para interactuar con el agente de ML desde la consola.
    """
    agente = crear_agente_ml()

    print("\n--- Agente de Machine Learning Activado ---")
    print("Ejemplo: 'entrena un modelo XGBoost con los datos de <ruta_a_tu_archivo>.csv'")
    print("Escribe 'salir' para terminar.")

    while True:
        pregunta_usuario = input("\nTú: ")
        if pregunta_usuario.lower() in ["salir", "exit", "quit"]:
            print("Agente: ¡Hasta la próxima sesión de modelado!")
            break
        
        respuesta = agente.invoke({"input": pregunta_usuario})
        print(f"Agente:\n{respuesta['output']}")

if __name__ == "__main__":
    main()