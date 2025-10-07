# agente_conexion.py

import pandas as pd
from typing import Union, Dict, Any
from langchain.tools import tool
from sqlalchemy import create_engine, text
import os

def obtener_datos_desde_csv(tool_input: Union[str, Dict[str, str]]) -> str:
    """
    Útil para cuando los datos de origen están en un archivo CSV local.
    Verifica si el archivo existe y devuelve la ruta para ser usada por otros agentes.
    La entrada puede ser un string con la ruta o un diccionario como {'ruta_csv': 'ruta/al/archivo.csv'}.
    """
    # Extraer la ruta del archivo, sin importar si viene como string o dict
    if isinstance(tool_input, dict):
        ruta_csv = tool_input.get('ruta_csv')
        if not ruta_csv:
            return "Error: El diccionario de entrada para Conector_CSV no contiene la clave 'ruta_csv'."
    elif isinstance(tool_input, str):
        # Limpiar el string por si el LLM envía "ruta_csv='...'"
        ruta_csv = tool_input.split('=')[-1].strip("'\" ")
    else:
        return f"Error: Formato de entrada no válido para Conector_CSV: {type(tool_input)}"

    print(f"--- Herramienta de Conexión: Verificando archivo CSV en '{ruta_csv}' ---")
    if not os.path.exists(ruta_csv):
        return f"Error: No se encontró el archivo CSV en la ruta especificada: {ruta_csv}"
    
    # El archivo existe, devolvemos la ruta para el siguiente paso del pipeline.
    return f"Archivo CSV verificado exitosamente. Usar la ruta: {ruta_csv}"

@tool
def obtener_datos_desde_postgres(
    usuario: str, 
    contrasena: str, 
    host: str, 
    puerto: str, 
    base_de_datos: str, 
    nombre_tabla: str,
    ruta_salida_csv: str = "temp_postgres_data.csv"
) -> str:
    """
    Útil para cuando los datos de origen están en una base de datos PostgreSQL.
    Se conecta a la base de datos, extrae la tabla completa y la guarda en un archivo CSV local.
    Devuelve la ruta del archivo CSV creado.
    """
    print(f"--- Herramienta de Conexión: Extrayendo tabla '{nombre_tabla}' desde PostgreSQL ---")
    try:
        # Construir la URL de conexión de SQLAlchemy
        db_url = f"postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{base_de_datos}"
        engine = create_engine(db_url)

        # Usar una conexión para ejecutar la consulta
        with engine.connect() as connection:
            # Consulta segura para evitar inyección SQL (aunque aquí es solo el nombre de la tabla)
            query = text(f'SELECT * FROM "{nombre_tabla}"')
            df = pd.read_sql_query(query, connection)

        # Guardar el DataFrame en un archivo CSV
        df.to_csv(ruta_salida_csv, index=False)

        return (f"Datos extraídos exitosamente de la tabla '{nombre_tabla}' de PostgreSQL. "
                f"Los datos se han guardado en el archivo local: {ruta_salida_csv}")

    except ImportError:
        return "Error: Para usar PostgreSQL, necesitas instalar 'sqlalchemy' y 'psycopg2-binary'. Ejecuta: pip install sqlalchemy psycopg2-binary"
    except Exception as e:
        return f"Error al conectar o extraer datos de PostgreSQL: {e}"

# Aquí podrías añadir más herramientas en el futuro, por ejemplo:
# @tool
# def obtener_datos_desde_oracle(...)
# @tool
# def obtener_datos_desde_gcs(...)