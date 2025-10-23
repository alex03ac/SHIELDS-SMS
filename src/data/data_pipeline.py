# --- BLOQUE DE IMPORTACIONES ---
# Estas líneas cargan las "cajas de herramientas" (librerías) que el script necesita.

import pandas as pd  # Importa Pandas (para manejar tablas de datos) y le da el apodo 'pd'.
import re  # Importa la librería de Expresiones Regulares (para buscar y reemplazar texto).
from nltk.corpus import \
    stopwords  # De NLTK, importa la lista de "stopwords" (palabras comunes como 'el', 'de', 'y').
from nltk.tokenize import \
    word_tokenize  # De NLTK, importa el "tokenizador" (para separar frases en palabras).
from sklearn.model_selection import \
    train_test_split  # De Scikit-learn, importa la función clave para dividir los datos.
import nltk  # Importa la librería NLTK completa (para descargar recursos).
import os  # Importa la librería 'os' (para interactuar con el sistema de archivos, como rutas).


# --- FUNCIÓN 1: REVISAR Y DESCARGAR RECURSOS DE NLTK ---
# Esta función se asegura de que NLTK tenga los paquetes de datos necesarios
# (como la lista de stopwords y el tokenizador 'punkt') antes de usarlos.

def revisar_y_descargar_recursos_nltk():
    """
    Comprueba si los recursos necesarios de NLTK ('stopwords', 'punkt') están
    instalados. Si no lo están, los descarga automáticamente.
    """
    # Lista de los paquetes de datos que este script necesita.
    resources = ['stopwords', 'punkt', 'punkt_tab']

    # Inicia un bucle para revisar cada recurso de la lista.
    for resource in resources:
        try:
            # Intenta "encontrar" el recurso en las carpetas de NLTK.
            # La ruta de búsqueda es diferente para 'punkt' y 'stopwords'.
            if resource == 'punkt_tab':
                nltk.data.find(f'tokenizers/{resource}')
            elif resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}')
            else:
                # 'stopwords' está en la carpeta 'corpora'.
                nltk.data.find(f'corpora/{resource}')

        except LookupError:
            # Este bloque se ejecuta si el 'try' falla (es decir, el recurso no se encontró).
            print(f"Descargando recurso '{resource}' de NLTK...")
            # Descarga el recurso faltante de internet.
            nltk.download(resource)


# --- EJECUCIÓN 1 ---
# Llama (ejecuta) la función que acabamos de definir para que se descargue
# todo lo necesario ANTES de que el script principal lo intente usar.
revisar_y_descargar_recursos_nltk()


# --- FUNCIÓN 2: CARGAR DATOS ---
# Esta función carga el archivo CSV que contiene los SMS
# y lo convierte en una tabla de Pandas (DataFrame).

def cargar_datos(file_path: str = 'data/raw/sms_dataset_original.csv') -> pd.DataFrame:
    """
    Carga el dataset de SMS desde una ruta de archivo CSV.

    Args:
        file_path (str): La ruta al archivo .csv.

    Returns:
        pd.DataFrame: Un DataFrame de Pandas con los datos, o None si falla.
    """
    try:
        # Intenta leer el archivo CSV usando Pandas.
        # 'encoding='latin-1'' se usa por si el archivo tiene tildes o caracteres especiales.
        # 'sep=',' indica que las columnas están separadas por comas.
        # 'quotechar='"' indica que el texto puede estar entre comillas dobles.
        df = pd.read_csv(file_path, encoding='latin-1', sep=',', quotechar='"')
        print(f"Datos cargados exitosamente desde: {file_path}")
        # Devuelve la tabla (DataFrame) cargada.
        return df
    except FileNotFoundError:
        # Se ejecuta si el 'try' falla porque el archivo no existe en esa ruta.
        print(f"ERROR: Archivo no encontrado en {file_path}. Asegúrate de que el dataset de prueba exista.")
        # Devuelve 'None' (nada) para que el resto del script sepa que falló.
        return None


# --- FUNCIÓN 3: LIMPIAR TEXTO ---
# Esta función toma UN solo texto (un SMS) y le aplica una limpieza básica.

def limpiar_texto(text: str) -> str:
    """
    Aplica limpieza básica a un string de texto:
    1. Convierte a minúsculas.
    2. Elimina puntuación y caracteres especiales.
    3. Elimina números.

    Args:
        text (str): El string de texto original.

    Returns:
        str: El texto limpio.
    """
    # Comprobación de seguridad: si la entrada no es un texto (ej. es un valor nulo o NaN),
    # devuelve un texto vacío para evitar errores.
    if not isinstance(text, str):
        return ""

    # 1. Convierte todo el texto a minúsculas (ej. "Hola" -> "hola").
    text = text.lower()

    # 2. Elimina puntuación y caracteres especiales.
    # Usa Regex (re.sub) para reemplazar cualquier cosa que NO sea (^)
    # una letra/número (\w) o un espacio (\s) con nada ('').
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Elimina todos los números.
    # Usa Regex para reemplazar uno o más dígitos (\d+) con nada ('').
    text = re.sub(r'\d+', '', text)

    # Devuelve el texto ya limpio.
    return text


# --- FUNCIÓN 4: PREPROCESAR TEXTO (PRINCIPAL) ---
# Esta función toma el DataFrame completo, aplica la limpieza de la Función 3,
# y además "tokeniza" (separa en palabras) y elimina las "stopwords".

def preprocesar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todo el preprocesamiento de texto al DataFrame.
    Crea columnas nuevas para el texto limpio y el texto preprocesado.

    Args:
        df (pd.DataFrame): El DataFrame cargado.

    Returns:
        pd.DataFrame: El DataFrame con las nuevas columnas procesadas.
    """
    # Comprobación de seguridad: si el DataFrame está vacío (por si la carga falló).
    if df is None or df.empty:
        return None

    # Rellena cualquier celda vacía (NaN) en la columna 'SMS_Text' con un texto vacío.
    # Esto es crucial para que la función 'limpiar_texto' no falle.
    df['SMS_Text'] = df['SMS_Text'].fillna('')

    # Crea una nueva columna 'cleaned_text'.
    # '.apply(limpiar_texto)' ejecuta la Función 3 para CADA fila de la columna 'SMS_Text'.
    df['cleaned_text'] = df['SMS_Text'].apply(limpiar_texto)

    # Obtiene la lista de stopwords en español de NLTK y la convierte en un 'set'.
    # Un 'set' es mucho más rápido para buscar que una lista.
    stop_words = set(stopwords.words('spanish'))

    # Define una función "interna" que se usará solo aquí.
    def tokenizar_y_filtrar(text):
        """
        1. Separa el texto en palabras (tokens).
        2. Elimina las stopwords y palabras de 1 letra.
        3. Vuelve a unir el texto.
        """
        # 1. Separa el texto en una lista de palabras (ej. "hola como estas" -> ['hola', 'como', 'estas'])
        tokens = word_tokenize(text, language='spanish')

        # 2. Filtra la lista de tokens.
        # Es una "list comprehension" que mantiene la palabra 'word' SOLO SI...
        filtered_tokens = [word for word in tokens
                           if word not in stop_words  # ...la palabra NO está en las stopwords
                           and len(word) > 1]  # ...Y la palabra mide más de 1 letra.

        # 3. Vuelve a unir las palabras filtradas en un solo string, separadas por un espacio.
        return " ".join(filtered_tokens)

    # Crea la columna final 'preprocessed_text'.
    # Aplica la función 'tokenizar_y_filtrar' a la columna 'cleaned_text'.
    df['preprocessed_text'] = df['cleaned_text'].apply(tokenizar_y_filtrar)

    # Imprime un mensaje de estado y una muestra (.head()) del DataFrame
    # para que el usuario vea el antes ('SMS_Text') y el después ('preprocessed_text').
    print("\nPreprocesamiento completado. Muestra de los datos:")
    print(df[['SMS_Text', 'preprocessed_text', 'Label']].head())

    # Devuelve el DataFrame totalmente procesado.
    return df


# --- FUNCIÓN 5: OBTENER DATOS DE PRUEBA Y ENTRENAMIENTO ---
# Esta función toma el DataFrame procesado y lo divide en los 4 conjuntos
# de datos que se necesitan para entrenar y evaluar un modelo de Machine Learning.
#

def obtener_datos_prueba_entrenamiento(df: pd.DataFrame, test_size: float = 0.4, random_state: int = 42) -> tuple:
    """
    Divide el DataFrame en conjuntos de entrenamiento y prueba (X, y).

    Args:
        df (pd.DataFrame): El DataFrame procesado.
        test_size (float): El porcentaje de datos a usar para el conjunto de prueba (ej. 0.4 = 40%).
        random_state (int): Una "semilla" para que la división aleatoria sea siempre la misma.

    Returns:
        tuple: Una tupla con 4 elementos: (X_train, X_test, y_train, y_test).
    """
    # Comprobación de seguridad.
    if df is None or df.empty:
        return None, None, None, None

    # 'X' (las características) = La columna de texto que el modelo usará para predecir.
    X = df['preprocessed_text']
    # 'y' (la etiqueta) = La columna que el modelo debe aprender a predecir (ej. "smishing" o "seguro").
    y = df['Label']

    # Llama a la función de sklearn para dividir los datos.
    # 'test_size=0.4' -> 40% para prueba, 60% para entrenamiento.
    # 'random_state=42' -> Asegura que la división sea reproducible.
    # 'stratify=y' -> ¡MUY IMPORTANTE! Asegura que la proporción de 'smishing' y 'seguro'
    # sea la misma en el grupo de entrenamiento y en el de prueba.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Imprime un resumen de cómo quedaron divididos los datos.
    print(f"\nDivisión de datos completada:")
    print(f"  Tamaño de entrenamiento: {len(X_train)} mensajes")
    print(f"  Tamaño de prueba: {len(X_test)} mensajes")
    print(f"  Distribución de clases en Entrenamiento:\n{y_train.value_counts()}")
    print(f"  Distribución de clases en Prueba:\n{y_test.value_counts()}")

    # Devuelve los 4 conjuntos de datos.
    return X_train, X_test, y_train, y_test


# --- BLOQUE DE EJECUCIÓN PRINCIPAL ---
# Este bloque es lo que realmente se ejecuta cuando corres el script
# desde la terminal (ej. `python este_archivo.py`).
# Orquesta la ejecución de todas las funciones anteriores en orden.

if __name__ == '__main__':
    # Esta línea especial significa: "ejecuta el siguiente código solo si
    # este archivo es el script principal (y no si está siendo importado)".

    # --- Construcción de la ruta al archivo ---
    # Obtiene la ruta de la carpeta donde se encuentra ESTE archivo .py.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta al archivo CSV.
    # 'os.path.join' une las partes de la ruta de forma inteligente (con / o \).
    # '..' significa "subir un nivel de carpeta".
    # Asume que la estructura es: SHIELDSMS-lipa_edition/src/features/este_archivo.py
    # Sube 2 niveles (a 'SHIELDSMS-lipa_edition') y luego baja a 'data/raw/...'
    data_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'sms_dataset_original.csv')

    # --- Flujo de ejecución ---

    # 1. Llama a la función para cargar los datos.
    sms_df = cargar_datos(data_path)

    # 2. Comprueba si la carga fue exitosa (si no devolvió 'None').
    if sms_df is not None:
        # 3. Si tuvo éxito, pasa el DataFrame a la función de preprocesamiento.
        processed_df = preprocesar_texto(sms_df)

        # 4. Pasa el DataFrame procesado a la función de división.
        # Las variables X_train, X_test, etc. quedan listas para ser usadas
        # por otro script que entrene el modelo.
        X_train, X_test, y_train, y_test = obtener_datos_prueba_entrenamiento(processed_df, test_size=0.4)
