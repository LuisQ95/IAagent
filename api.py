# API.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Importamos la función que crea nuestro agente desde el otro archivo.
# Esto es una buena práctica para no repetir código.
from agente_gemini import crear_agente_ia

# --- Modelos de Datos con Pydantic ---
# Definen la estructura de los datos que la API espera recibir y enviar.

class PreguntaRequest(BaseModel):
    """Modelo para la petición de entrada."""
    pregunta: str

class RespuestaResponse(BaseModel):
    """Modelo para la respuesta de salida."""
    respuesta: str

# --- Configuración de la App FastAPI ---

app = FastAPI(
    title="API del Agente de IA",
    description="Una API para interactuar con un agente conversacional basado en Gemini y LangChain.",
    version="1.0.0",
)

# --- Ciclo de Vida del Agente ---

# Usamos un diccionario para almacenar el agente una vez creado.
# Esto evita tener que inicializar el agente (que es un proceso lento) en cada petición.
state = {}

@app.on_event("startup")
def cargar_modelo():
    """
    Esta función se ejecuta una única vez cuando el servidor se inicia.
    Crea la instancia del agente y la guarda en el 'state'.
    """
    print("Iniciando servidor...")
    state["agente"] = crear_agente_ia()
    print("Servidor iniciado. El agente está listo.")


# --- Endpoints de la API ---

@app.get("/", include_in_schema=False)
async def root():
    """Endpoint raíz que da la bienvenida y redirige a la documentación."""
    return {
        "mensaje": "Bienvenido a la API del Agente de IA.",
        "documentacion_interactiva": "Por favor, ve a /docs para probar la API."
    }

@app.get("/preguntar", response_model=RespuestaResponse)
async def preguntar_al_agente(pregunta: str):
    """
    Recibe una pregunta, la procesa con el agente y devuelve la respuesta.
    """
    agente = state.get("agente")
    if not agente:
        raise HTTPException(status_code=503, detail="El agente no se ha inicializado. Por favor, espera un momento y reintenta.")
    
    try:
        print(f"Recibida pregunta para el agente: '{pregunta}'")
        # Invocamos al agente con la pregunta recibida en la petición.
        respuesta_agente = agente.invoke({"input": pregunta})
        
        # Extraemos la respuesta final del diccionario que devuelve el agente.
        respuesta_final = respuesta_agente.get("output", "No se pudo obtener una respuesta del agente.")
        
        print(f"Respuesta generada: '{respuesta_final}'")
        return {"respuesta": respuesta_final}
    except Exception as e:
        # Capturamos cualquier error que pueda ocurrir durante la invocación del agente.
        print(f"[ERROR en API /preguntar]: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error interno al procesar la pregunta.")


# --- Punto de entrada para ejecutar el servidor ---

if __name__ == "__main__":
    # Este bloque permite ejecutar la API directamente con `python API.py`
    # host="0.0.0.0" hace que la API sea accesible desde otros dispositivos en tu red.
    print("Iniciando servidor Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
