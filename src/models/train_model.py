import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# Importacion para balanceo de la muestra de entrenamiento
from imblearn.over_sampling import RandomOverSampler 

# Añadir directorios al PATH para importar módulos personalizados
base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.join(base_dir, 'src', 'data'))
sys.path.append(os.path.join(base_dir, 'src', 'features'))

from data_pipeline import cargar_datos, preprocesar_texto, obtener_datos_prueba_entrenamiento
from feature_extractor import crear_caracteristicas


def grafiacar_matriz_confusion(cm, labels, model_name="SVM"):
    """Genera un mapa de calor para la matriz de confusión y lo guarda en reports/figures/."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True,        # Mostrar los valores numéricos
        fmt='d',           # Formatear los números como enteros
        cmap='Blues',      # Esquema de color
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(f'Matriz de Confusión: {model_name}')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    
    # Rutas para guardar la imagen (RNF-01)
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'reports', 'figures')
    os.makedirs(reports_dir, exist_ok=True)
    file_path = os.path.join(reports_dir, f'{model_name}_matriz_confusion.png')
    
    plt.savefig(file_path)
    print(f"\nGráfico guardado en: {file_path}")
    plt.close() # Cierra la figura para liberar memoria


def entrenar_y_evaluar_svm(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    """Entrena y evalúa el modelo SVM e imprime/grafica la matriz de confusión."""
    model_name = 'SVM'
    print(f"\n--- Entrenando y evaluando modelo {model_name} ---")

    # 1. Crear el modelo SVM
    modelo = SVC(kernel='linear', random_state=42)
    
    # 2. Entrenamiento
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # 3. Predicción en el conjunto de prueba
    y_predicho = modelo.predict(X_prueba)

    # 4. Evaluación
    precision = accuracy_score(y_prueba, y_predicho)
    reporte = classification_report(y_prueba, y_predicho, zero_division=0)
    
    # Cálculo de la Matriz de Confusión
    labels = sorted(y_prueba.unique()) 
    matriz_c = confusion_matrix(y_prueba, y_predicho, labels=labels) 

    # Mapa de calor/matriz de confusion
    grafiacar_matriz_confusion(matriz_c, labels, model_name)

    print("\nReporte de Clasificación:")
    print(reporte)

    return modelo, precision


def entrenar_svm(processed_df: pd.DataFrame):
    """Ejecuta el pipeline de entrenamiento del modelo SVM, balanceo y guardado."""

    # 1. Obtener los conjuntos de entrenamiento y prueba (test_size=0.2 se usa por el desbalance)
    X_train_text, X_test_text, y_train_orig, y_test_orig = obtener_datos_prueba_entrenamiento(processed_df, test_size=0.2)

    # 2. Crear las características a partir del texto (Llamada a feature_extractor.py)
    X_train_raw, X_test, _ = crear_caracteristicas(X_train_text, X_test_text, processed_df)

    print("\n--- Aplicando RandomOverSampler al Conjunto de Entrenamiento ---")
    
    ros = RandomOverSampler(random_state=42)
    # El remuestreo se hace sobre las matrices dispersas
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_raw, y_train_orig)
    
    # Usamos .shape[0] para matrices dispersas
    print(f"  Tamaño de Entrenamiento Original: {X_train_raw.shape[0]} -> Remuestreado: {X_train_resampled.shape[0]}")
    print(f"  Nueva Distribución de Clases:\n{y_train_resampled.value_counts()}")
    # ------------------------------------------------------------------

    # 3. Entrenar y evaluar el modelo SVM
    # Entrenamos con datos BALANCEADOS y evaluamos con datos ORIGINALES (X_test, y_test_orig)
    svm_model, svm_acc = entrenar_y_evaluar_svm(X_train_resampled, X_test, y_train_resampled, y_test_orig)
    
    print(f"\n--- Entrenamiento completado ---")
    print(f"Precisión final del modelo SVM: {svm_acc:.4f}")

    # 4. Guardar el modelo
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'svm_model.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(svm_model, f, protocol=4)

    print(f"El modelo final 'SVM' ha sido guardado en {model_path}")


if __name__ == '__main__':
    # 1. Cargar y preprocesar los datos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'sms_dataset_original.csv')

    sms_df = cargar_datos(data_path)
    if sms_df is not None:
        processed_df = preprocesar_texto(sms_df)

        # Ejecutar el entrenamiento del modelo SVM
        entrenar_svm(processed_df)