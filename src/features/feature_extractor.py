"""
Este script contiene funciones para la extracción de características de mensajes SMS
para un sistema de detección de smishing.

Importaciones:
- pandas as pd: Para manipulación y análisis de datos estructurados.
- re: Para operaciones de expresiones regulares.
- TfidfVectorizer: Para convertir texto en vectores numéricos basados en la frecuencia de términos.
- hstack, csr_matrix: Para manejo eficiente de matrices dispersas.
- pickle: Para serialización de objetos Python.
- os, sys: Para operaciones del sistema y manipulación de rutas.
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import pickle
import os
import sys

# Añadir el directorio src al PATH para importar data_pipeline.py
# Esto permite que este script acceda a las funciones de carga y preprocesamiento
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from data_pipeline import cargar_datos, preprocesar_texto, obtener_datos_prueba_entrenamiento


# 1. Funciones de Extracción de Características de Ingeniería Social

def extraer_cantidad_url(sms_series: pd.Series) -> pd.Series:
    """
    Extrae el número de URLs encontradas en cada SMS.
    Se recomienda usar la columna de texto original (SMS_Text) para una detección precisa.
    """
    # Patrón robusto para URLs (http/https, dominios, y acortadores comunes)
    patron_url = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|(\b(tinyurl|bit\.ly|cutt\.ly|goo\.gl|t\.co)\b)'

    # Aplica la búsqueda y cuenta de URLs
    return sms_series.apply(lambda text: len(re.findall(patron_url, str(text).lower())))


def extraer_palabra_clave_sospechosa(sms_series: pd.Series) -> pd.Series:
    """
    Cuenta palabras clave que sugieren urgencia o peligro (típicas de smishing).
    Se usa la columna de texto preprocesado.
    """
    # Palabras clave comunes
    lista_palabras_sospechosas = ['bloqueada', 'urgente', 'premio', 'error', 'suspender',
                             'actualiza', 'clave', 'contraseña', 'pago', 'acceso', 'transferencia']

    # Crear un patrón de regex a partir de la lista
    patron_palabras_sospechosas = r'|'.join(lista_palabras_sospechosas)

    # Aplica la búsqueda y cuenta de keywords
    return sms_series.apply(lambda text: len(re.findall(patron_palabras_sospechosas, str(text).lower())))


# 2. Función Principal de Vectorización y Combinación

def crear_caracteristicas(X_train_text: pd.Series, X_test_text: pd.Series, df_original: pd.DataFrame):
    """
    Crea y combina características (TF-IDF y características de ingeniería social).
    """

    # Vectorización TF-IDF (Extracción de Características de Texto)

    # Inicializa el vectorizador TF-IDF
    tfidf = TfidfVectorizer(max_features=2000)  # Límite de 2000 características

    # 1. Ajustar el vectorizador solo en los datos de entrenamiento
    X_train_tfidf = tfidf.fit_transform(X_train_text)

    # 2. Transformar los datos de prueba
    X_test_tfidf = tfidf.transform(X_test_text)

    print("\nVectorización TF-IDF completo.")

    # Extracción de Características de Ingeniería Social

    # 1. Aseguramos la alineación de las características con los índices de Train y Test
    indices_entrenamiento = X_train_text.index
    indices_prueba = X_test_text.index

    # 2. Extraer y añadir las características al DataFrame temporal
    df_original['cantidad_url'] = extraer_cantidad_url(df_original['SMS_Text'])
    df_original['cantidad_palabra_clave'] = extraer_palabra_clave_sospechosa(df_original['preprocessed_text'])

    # 3. Seleccionar las características extraídas que corresponden a Train y Test
    # Usando csr_matrix para asegurar que los datos estén en formato disperso para que sea compatibles con TF-IDF
    X_train_extra = csr_matrix(df_original.loc[indices_entrenamiento, ['cantidad_url', 'cantidad_palabra_clave']].values)
    X_test_extra = csr_matrix(df_original.loc[indices_prueba, ['cantidad_url', 'cantidad_palabra_clave']].values)

    # 4. Combinar las matrices dispersas (TF-IDF + Características Extra)
    X_train_final = hstack([X_train_tfidf, X_train_extra])
    X_test_final = hstack([X_test_tfidf, X_test_extra])

    print("Extracción de características adicionales y combinación completada.")
    print(f"  Forma final de X_train (TF-IDF + Extras): {X_train_final.shape}")

    # Guardar Vectorizador (Necesario para predicción futura)

    directorio_modelo = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(directorio_modelo, exist_ok=True)

    ruta_vectorizador = os.path.join(directorio_modelo, 'vectorizer.pkl')
    with open(ruta_vectorizador, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"Vectorizador TF-IDF guardado en {ruta_vectorizador}")

    return X_train_final, X_test_final, tfidf


if __name__ == '__main__':
    # 1. Obtener datos del pipeline anterior
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_datos = os.path.join(directorio_actual, '..', '..', 'data', 'raw', 'sms_dataset_original.csv')

    sms_df = cargar_datos(ruta_datos)
    processed_df = preprocesar_texto(sms_df)

    X_train_text, X_test_text, y_train, y_test = obtener_datos_prueba_entrenamiento(processed_df, test_size=0.4)

    # 2. Crear las características
    X_train_feat, X_test_feat, tfidf_model = crear_caracteristicas(
        X_train_text, X_test_text, processed_df.copy()
    )

    print(f"\nNúmero total de características por mensaje: {X_train_feat.shape[1]}")