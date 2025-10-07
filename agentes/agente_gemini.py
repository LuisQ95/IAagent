# agente_gemini.py

import os
from dotenv import load_dotenv

# --- Componentes Principales de LangChain ---
# Importamos el modelo de chat de Google
from langchain_google_genai import ChatGoogleGenerativeAI
# Importamos las funciones para crear y usar agentes (actualizado para evitar advertencias)
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools

def crear_agente_ia():
    """
    Configura y crea una instancia de un agente de IA.

    Esta función realiza los siguientes pasos:
    1. Carga las claves de API desde el archivo .env.
    2. Inicializa el modelo de lenguaje (LLM), en este caso, Gemini-Pro.
    3. Carga las herramientas que el agente podrá utilizar (búsqueda y calculadora).
    4. Inicializa el agente, uniendo el LLM y las herramientas.

    Returns:
        AgentExecutor: El agente listo para ser ejecutado.
    """
    # 1. Cargar variables de entorno
    # Esta función busca un archivo .env y carga sus variables para que
    # las librerías como LangChain puedan usarlas automáticamente.
    print("Cargando claves de API...")
    load_dotenv()

    # 2. Inicializar el Modelo de Lenguaje (LLM)
    # Usamos el modelo 'gemini-pro'. La 'temperature' controla la creatividad;
    # un valor bajo como 0.3 lo hace más preciso y menos propenso a inventar.
    print("Inicializando el modelo Gemini-Pro...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"), # Le pasamos la clave explícitamente
        temperature=0.3
    )

    # 3. Cargar Herramientas
    # Le damos al agente capacidades adicionales.
    # - "serpapi": Permite realizar búsquedas en Google.
    # - "llm-math": Un asistente matemático que usa el propio LLM para resolver cálculos.
    print("Cargando herramientas (Búsqueda en Google, Calculadora)...")
    # Usamos load_tools para cargar las herramientas deseadas.
    # "serpapi" usará automáticamente la SERPAPI_API_KEY del archivo .env.
    # "llm-math" no necesita clave y le da al agente la capacidad de calcular.
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

    # 4. Inicializar el Agente
    # Unimos el LLM y las herramientas.
    # - agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION: Un tipo de agente estándar que
    #   razona paso a paso ("React": Reason and Act) sobre qué herramienta usar.
    # - verbose=True: ¡Esencial! Muestra el "pensamiento" del agente en la terminal.
    # - handle_parsing_errors=True: Aporta robustez si el LLM no responde en el
    #   formato exacto que el agente espera.
    print("Creando el agente...")
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    print("¡Agente listo!")
    return agent_executor

def main():
    """
    Función principal que inicia el agente y gestiona la interacción con el usuario.
    """
    agente = crear_agente_ia()

    print("\n--- Asistente IA con Gemini Activado ---")
    print("Puedes empezar a chatear. Escribe 'salir' para terminar.")

    # Bucle de conversación
    while True:
        try:
            # Pedir entrada al usuario
            pregunta_usuario = input("\nTú: ")

            # Condición de salida
            if pregunta_usuario.lower() in ["salir", "exit", "quit"]:
                print("Agente: ¡Hasta luego! Ha sido un placer ayudarte.")
                break

            # Invocar al agente con la pregunta del usuario
            # El agente decidirá si responder directamente o usar una herramienta.
            # Usamos el formato de diccionario {"input": ...} que es el estándar actual.
            respuesta = agente.invoke({"input": pregunta_usuario})

            # Imprimir la respuesta final del agente
            print(f"Agente: {respuesta['output']}")

        except Exception as e:
            # Capturar cualquier error inesperado durante la ejecución del agente
            print(f"\n[ERROR]: Ha ocurrido un problema: {e}")
            print("Por favor, inténtalo de nuevo.")

# Este bloque asegura que la función main() solo se ejecute
# cuando el archivo es llamado directamente desde la terminal.
if __name__ == "__main__":
    main()
