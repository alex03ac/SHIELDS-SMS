from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import sys
import re
import requests  # NECESARIO PARA LLAMAR A LA API DE VIRUSTOTAL
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

# 🚨 CLAVE DE API DE VIRUSTOTAL 🚨
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "9c5260f99766e54a0659c94c72cbe4077014dadbd5f168c588335e68c97263ee")

# 🌟 NUEVA LISTA BLANCA DE DOMINIOS SEGUROS 🌟
DOMAIN_WHITELIST = {
    'youtube.com',
    'youtu.be',
    'google.com',
    'facebook.com',
    'twitter.com',
    'instagram.com',
    'linkedin.com',
    'wa.me'  # WhatsApp
}

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="API de Clasificación Shield-SMS",
    description="API para detectar Smishing usando ML y validación de VirusTotal.",
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


def check_url_virustotal(url: str) -> int:
    """Consulta la API de VirusTotal para verificar la reputación de una URL/Dominio."""
    if VIRUSTOTAL_API_KEY == "9c5260f99766e54a0659c94c72cbe4077014dadbd5f168c588335e68c97263ee" or not VIRUSTOTAL_API_KEY:
        print("ADVERTENCIA: Clave de VirusTotal no configurada. Saltando verificación externa.")
        return 0

    domain = re.sub(r'^https?://', '', url).split('/')[0]

    # 🌟 MODIFICACIÓN: No escanear dominios de la whitelist 🌟
    # Comprobar el dominio principal y subdominios (ej. www.youtube.com)
    if any(domain.endswith(safe_domain) for safe_domain in DOMAIN_WHITELIST):
        print(f"INFO: Dominio {domain} está en la whitelist, saltando VirusTotal.")
        return 0  # Tratar como seguro

    url_report_endpoint = f"https://www.virustotal.com/api/v3/domains/{domain}"
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}

    try:
        response = requests.get(url_report_endpoint, headers=headers, timeout=8)
        response.raise_for_status()
        data = response.json()

        if 'data' in data and 'attributes' in data['data']:
            attributes = data['data']['attributes']
            malicious_count = attributes.get('last_analysis_stats', {}).get('malicious', 0)

            if malicious_count >= 2:
                print(f"ALERTA VT: Dominio {domain} marcado como malicioso ({malicious_count} motores).")
                return 1

        return 0

    except requests.exceptions.RequestException as e:
        print(f"Error al consultar VirusTotal para {domain}: {e}")
        return 0


def predecir_smishing(sms_text: str):
    """
    Aplica la lógica de ML y la sobrescritura de VirusTotal.
    Retorna: (final_classification: str, fue_sobrescrito: bool)
    """
    global clasificador, vectorizador
    if clasificador is None or vectorizador is None:
        cargar_recursos()

    # 1. Preprocesamiento de texto (PLN)
    texto_limpio = limpiar_texto(sms_text)

    # 2. Extracción de Características ML
    url_pattern = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(\b(tinyurl|bit\.ly|cutt\.ly|goo\.gl|t\.co)\b)'
    found_urls = re.findall(url_pattern, sms_text.lower())

    # 🌟 MODIFICACIÓN: Lógica de Whitelist para el conteo de URLs 🌟
    conteo_urls = 0
    url_para_vt = None  # Guardar la primera URL no-whitelisted para VirusTotal

    if found_urls:
        for url_tuple in found_urls:
            url_str = "".join(url_tuple)
            if not url_str:
                continue

            domain = re.sub(r'^https?://', '', url_str).split('/')[0]

            # Comprobar si el dominio NO está en la whitelist
            if not any(domain.endswith(safe_domain) for safe_domain in DOMAIN_WHITELIST):
                conteo_urls += 1  # Contar solo URLs sospechosas
                if url_para_vt is None:
                    url_para_vt = url_str  # Guardar la primera URL sospechosa para VT

    # Conteo de palabras clave
    palabras_sospechosas = ['bloqueada', 'urgente', 'premio', 'error', 'suspender',
                            'actualiza', 'clave', 'contraseña', 'pago', 'acceso', 'transferencia']
    patron_palabras_sospechosas = r'|'.join(palabras_sospechosas)
    conteo_palabras = len(re.findall(patron_palabras_sospechosas, texto_limpio))

    # 3. Clasificación ML (Inicial)
    X_tfidf = vectorizador.transform([texto_limpio])
    # Usar el 'conteo_urls' filtrado (ignora YouTube, etc.)
    X_extra = csr_matrix([conteo_urls, conteo_palabras])
    X_final = hstack([X_tfidf, X_extra])
    prediction_ml = clasificador.predict(X_final)[0]

    # -------------------------------------------------------------
    # 4. LÓGICA DE SOBRESCRITURA CON VIRUSTOTAL
    # -------------------------------------------------------------

    final_classification = prediction_ml
    fue_sobrescrito = False

    # Solo si el ML lo marcó legítimo Y tenemos una URL sospechosa para verificar
    if prediction_ml == 'legítimo' and url_para_vt is not None:
        is_virustotal_malicious = check_url_virustotal(url_para_vt)

        if is_virustotal_malicious == 1:
            final_classification = 'smishing'
            fue_sobrescrito = True

    return final_classification, fue_sobrescrito


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
    Endpoint principal que recibe un SMS y devuelve su clasificación final.
    """
    try:
        prediction, fue_sobrescrito = predecir_smishing(request.sms_text)

        if prediction == 'smishing':
            alerta = "Smishing detectado (¡ALERTA!)"
            if fue_sobrescrito:
                alerta += " (Validado por VirusTotal)"
        else:
            alerta = "Mensaje legítimo (Bajo Riesgo)"

        return {
            "sms_input": request.sms_text,
            "classification": prediction,
            "mensaje_alerta": alerta
        }
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la clasificación: {e}")