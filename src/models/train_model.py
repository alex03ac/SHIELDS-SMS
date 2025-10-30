"""
Este script implementa el entrenamiento, evaluación y guardado del modelo
Support Vector Machine (SVM) para el sistema de detección de smishing SHIELD-SMS.

Importaciones:
- pandas as pd: Para manejo y análisis de datos en estructuras tipo DataFrame.
- SVC (de sklearn.svm): Implementa el modelo de clasificación Support Vector Machine.
- classification_report, accuracy_score (de sklearn.metrics): Para generar métricas de evaluación y medir la precisión.
- pickle: Para guardar y cargar modelos entrenados en archivos binarios.
- os, sys: Para manipular rutas de archivos y añadir directorios al PATH para importar módulos personalizados.
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import sys

# Añadir directorios al PATH para importar módulos personalizados
base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.join(base_dir, 'src', 'data'))
sys.path.append(os.path.join(base_dir, 'src', 'features'))

from data_pipeline import cargar_datos, preprocesar_texto, obtener_datos_prueba_entrenamiento
from feature_extractor import crear_caracteristicas


def entrenar_y_evaluar_svm(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    """Entrena y evalúa un modelo SVM."""
    print("\n--- Entrenando y evaluando modelo SVM ---")

    # 1. Crear el modelo SVM
    modelo = SVC(kernel='linear', random_state=42)

    # 2. Entrenamiento
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # 3. Predicción en el conjunto de prueba
    y_predicho = modelo.predict(X_prueba)

    # 4. Evaluación
    precision = accuracy_score(y_prueba, y_predicho)
    reporte = classification_report(y_prueba, y_predicho, zero_division=0)

    print("\nResultados de Evaluación para SVM:")
    print(f"Precisión (Accuracy): {precision:.4f}")
    print(reporte)

    return modelo, precision


def entrenar_svm(processed_df: pd.DataFrame):
    """Ejecuta el pipeline completo de entrenamiento y guardado del modelo SVM."""

    # 1. Obtener los conjuntos de entrenamiento y prueba
    X_train_text, X_test_text, y_train, y_test = obtener_datos_prueba_entrenamiento(processed_df, test_size=0.4)

    # 2. Crear las características a partir del texto
    X_train, X_test, _ = crear_caracteristicas(X_train_text, X_test_text, processed_df)

    # 3. Entrenar y evaluar el modelo SVM
    svm_model, svm_acc = entrenar_y_evaluar_svm(X_train, X_test, y_train, y_test)

    print(f"\n--- Entrenamiento completado ---")
    print(f"Precisión final del modelo SVM: {svm_acc:.4f}")

    # 4. Guardar el modelo entrenado
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'svm_model.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(svm_model, f, protocol=4)

    print(f"El modelo SVM ha sido guardado en {model_path}")


if __name__ == '__main__':
    # 1. Cargar y preprocesar los datos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'sms_dataset_original.csv')

    sms_df = cargar_datos(data_path)
    if sms_df is not None:
        processed_df = preprocesar_texto(sms_df)

        # Ejecutar el entrenamiento del modelo SVM
        entrenar_svm(processed_df)
