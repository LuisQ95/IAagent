# orquestador_ml.py

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from typing_extensions import Annotated

# Importar las herramientas de los agentes especializados
from agentes.agente_conexion import obtener_datos_desde_csv, obtener_datos_desde_postgres
from agentes.agente_preprocesamiento import analizar_y_preprocesar_datos, PreprocesarInput
from agentes.agente_modelado import optimizar_y_entrenar_modelo

# --- Prompt para el agente ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def crear_orquestador():
    """
    Crea el agente orquestador principal que utiliza las herramientas de los
    agentes especializados.
    """
    print("Inicializando el modelo Gemini para el Orquestador...")
    llm = ChatGoogleGenerativeAI( # gemini-1.5-pro-latest o gemini-pro
        model="gemini-2.5-flash", 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0, # Muy determinista para seguir el flujo de trabajo
        convert_system_message_to_human=True
    )

    # Definir las herramientas que el orquestador puede usar.
    # Cada herramienta corresponde a la capacidad principal de un agente especializado.
    tools = [
        Tool(
            name="Conector_CSV",
            func=obtener_datos_desde_csv,
            description="Útil para verificar la existencia de un archivo CSV local. La entrada a esta herramienta debe ser únicamente la ruta del archivo como un string."
        ),
        Tool(
            name="Conector_PostgreSQL",
            func=obtener_datos_desde_postgres,
            description="Útil para extraer datos de una tabla de PostgreSQL a un CSV. Argumentos requeridos: 'usuario', 'contrasena', 'host', 'puerto', 'base_de_datos', 'nombre_tabla'. Argumento opcional: 'ruta_salida_csv'."
        ),
        analizar_y_preprocesar_datos,
        optimizar_y_entrenar_modelo,
    ]

    # --- Creación del Agente con el método moderno ---

    # 1. Definir el prompt del agente
    # Este prompt le dice al agente cómo comportarse y dónde encontrar las variables de entrada.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente experto en flujos de trabajo de machine learning. Tu objetivo es ejecutar una secuencia de herramientas para cumplir con la solicitud del usuario. Piensa paso a paso."),
        ("human", "{input}"),
        # MessagesPlaceholder es donde el agente guarda su "memoria de trabajo" (los pasos intermedios).
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 2. Crear el agente que une el LLM, las herramientas y el prompt
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 3. Crear el ejecutor del agente, que es lo que realmente corre el bucle de pensamiento-acción.
    orquestador = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Intenta de nuevo. Asegúrate de usar el formato correcto y las herramientas disponibles."
    )

    print("¡Orquestador de ML listo!")
    return orquestador

def main():
    """
    Función principal para interactuar con el orquestador de ML.
    """
    orquestador = crear_orquestador()

    print("\n--- Orquestador de Flujos de Machine Learning ---")
    print("Describe el flujo de trabajo que deseas ejecutar.")
    print("Ejemplo: 'Usa el archivo datos_clasificacion.csv, preprocésalo y luego entrena un modelo XGBoost con los datos limpios.'")
    print("Escribe 'salir' para terminar.")

    while True:
        pregunta_usuario = input("\nTú: ")
        if pregunta_usuario.lower() in ["salir", "exit", "quit"]:
            break
        
        # Usar .invoke() en lugar de .run() para seguir la nueva API de LangChain
        respuesta = orquestador.invoke({"input": pregunta_usuario})
        print("\n--- Respuesta del Orquestador ---")
        print(respuesta.get("output"))
        print("---------------------------------")

if __name__ == "__main__":
    main()