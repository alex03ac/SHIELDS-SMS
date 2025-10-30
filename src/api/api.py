from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import sys
import re


from scipy.sparse import hstack, csr_matrix
# --- Configuración de Rutas y Carga de Recursos ---
# Añadir el directorio 'src/data' al PATH para usar funciones de preprocesamiento
base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.join(base_dir, 'src', 'data'))

# Importar las funciones necesarias desde data_pipeline.py para la predicción.
from data_pipeline import limpiar_texto

# Definición de rutas para los archivos persistentes del modelo
MODEL_PATH = os.path.join(base_dir, 'models', 'svm_model.pkl')
VECTORIZER_PATH = os.path.join(base_dir, 'models', 'vectorizer.pkl')

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="API de Clasificación Shield-SMS (MVP)",
    description="API para detectar Smishing usando el modelo ML entrenado.",
    version="1.0.0"
)

clasificador = None
vectorizador = None


# --- Esquema de Solicitud (Input del SMS) ---

class SMSRequest(BaseModel):
    """Define el formato de datos esperado para la solicitud de clasificación."""
    sms_text: str


# --- Funciones de Carga y Predicción ---

def cargar_recursos():
    """Carga el modelo y el vectorizador guardados."""
    global clasificador, vectorizador

    try:
        with open(MODEL_PATH, 'rb') as f:
            clasificador = pickle.load(f)

        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizador = pickle.load(f)
        print("Recursos del modelo (Clasificador y Vectorizador) cargados exitosamente.")

    except FileNotFoundError as e:
        
        raise RuntimeError(f"Error 500: Archivos de modelo/vectorizador no encontrados. RUTA ESPERADA: {MODEL_PATH}")
    except Exception as e:
      
        raise RuntimeError(
            f"Error 500: Fallo al cargar los archivos .pkl. Asegúrate de usar las mismas versiones de librerías. Detalle: {e}")

def predecir_smishing(sms_text: str):
    """
    Función completa de predicción que imita el pipeline:
    Preprocesamiento -> Extracción de Características -> Clasificación.
    """
    global clasificador, vectorizador
    if clasificador is None or vectorizador is None:
        cargar_recursos()

    # 1. Preprocesamiento de texto 
    texto_limpio = limpiar_texto(sms_text)

    # 2. Vectorización (TF-IDF)
    X_tfidf = vectorizador.transform([texto_limpio])

    # 3. Extracción de Características Adicionales
    url_pattern = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(\b(tinyurl|bit\.ly|cutt\.ly|goo\.gl|t\.co)\b)'
    conteo_urls = len(re.findall(url_pattern, sms_text.lower()))
    
    # Detección de Conteo de Palabras Clave Sospechosas (Usamos el texto limpio)
    palabras_sospechosas = ['bloqueada', 'urgente', 'premio', 'error', 'suspender',
                             'actualiza', 'clave', 'contraseña', 'pago', 'acceso', 'transferencia']
    patron_palabras_sospechosas = r'|'.join(palabras_sospechosas)
    conteo_palabras = len(re.findall(patron_palabras_sospechosas, texto_limpio))

    # 4. Combinar Características
    X_extra = csr_matrix([conteo_urls, conteo_palabras])
    # Aseguramos el mismo orden: TF-IDF primero, luego las características manuales
    X_final = hstack([X_tfidf, X_extra])

    # 5. Clasificación
    prediction = clasificador.predict(X_final)[0]

    return prediction


# --- Endpoints de la API ---

@app.on_event("startup")
def iniciar_servidor():
    """Carga los recursos del modelo al iniciar la API."""
    cargar_recursos()


@app.get("/")
def raiz():
    """Endpoint de bienvenida simple para verificar que la API está viva."""
    return {"mensaje": "La API Shield-SMS está en funcionamiento. Usa el endpoint /classify para enviar un SMS."}


@app.post("/classify")
def endpoint_clasificar_sms(request: SMSRequest):
    """
    Endpoint principal que recibe un SMS y devuelve su clasificación.
    """
    try:
        prediction = predecir_smishing(request.sms_text)
        # Mapeo de la etiqueta a un mensaje amigable
        alerta = "Smishing detectado (¡ALERTA!)" if prediction == 'smishing' else "Mensaje legítimo (Bajo Riesgo)"

        return {
            "sms_input": request.sms_text,
            "classification": prediction,
            "mensaje_alerta": alerta
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la clasificación: {e}")
