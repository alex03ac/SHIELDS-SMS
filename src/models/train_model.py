"""
Este script implementa el entrenamiento, evaluación y selección de modelos de aprendizaje automático
para el sistema de detección de smishing SHIELDS-SMS.

Importaciones:
- pandas as pd: Para manejo y análisis de datos en estructuras tipo DataFrame.
- SVC (de sklearn.svm): Implementa el modelo de clasificación Support Vector Machine.
- RandomForestClassifier (de sklearn.ensemble): Implementa el modelo de clasificación basado en bosques aleatorios.
- classification_report, accuracy_score (de sklearn.metrics): Para generar métricas de evaluación y medir la precisión.
- pickle: Para guardar y cargar modelos entrenados en archivos binarios.
- os, sys: Para manipular rutas de archivos y añadir directorios al PATH para importar módulos personalizados.
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import sys

# Añadir directorios al PATH para importar módulos
base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.join(base_dir, 'src', 'data'))
sys.path.append(os.path.join(base_dir, 'src', 'features'))

from data_pipeline import cargar_datos, preprocesar_texto, obtener_datos_prueba_entrenamiento
from feature_extractor import crear_caracteristicas


def entrenar_y_evaluar_modelo(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba, nombre_modelo: str):
    """Entrena y evalúa un modelo de clasificación dado."""
    print(f"\n--- Entrenando y evaluando {nombre_modelo} ---")

    if nombre_modelo == 'SVM':
        # Modelo Support Vector Machine
        modelo = SVC(kernel='linear', random_state=42)
    elif nombre_modelo == 'RandomForest':
        # Modelo Random Forest
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Modelo {nombre_modelo} no soportado.")

    # 1. Entrenamiento
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # 2. Predicción en el conjunto de prueba
    y_predicho = modelo.predict(X_prueba)

    # 3. Evaluación
    precision = accuracy_score(y_prueba, y_predicho)
    reporte = classification_report(y_prueba, y_predicho, zero_division=0)

    print(f"\nResultados de Evaluación para {nombre_modelo}:")
    print(f"Precisión (Accuracy): {precision:.4f}")
    print(reporte)

    return modelo, precision


def run_model_training(processed_df: pd.DataFrame):
    """Ejecuta el pipeline completo de entrenamiento y selección de modelo."""

    # 1. Obtener los conjuntos de entrenamiento y prueba
    X_train_text, X_test_text, y_train, y_test = obtener_datos_prueba_entrenamiento(processed_df, test_size=0.4)

    # 2. Crear las características a partir del texto (Llama a feature_extractor.py)
    X_train, X_test, _ = crear_caracteristicas(X_train_text, X_test_text, processed_df)

    # 3. Entrenar y evaluar modelos
    models = {}

    # Comparar SVM y Random Forest
    svm_model, svm_acc = entrenar_y_evaluar_modelo(X_train, X_test, y_train, y_test, 'SVM')
    models['SVM'] = {'model': svm_model, 'accuracy': svm_acc}

    rf_model, rf_acc = entrenar_y_evaluar_modelo(X_train, X_test, y_train, y_test, 'RandomForest')
    models['RandomForest'] = {'model': rf_model, 'accuracy': rf_acc}

    # 4. Selección del mejor modelo
    best_model_name = max(models, key=lambda name: models[name]['accuracy'])
    best_model_data = models[best_model_name]

    print(f"\n--- Selección del Mejor Modelo ---")
    print(f"El mejor modelo es: {best_model_name} con una precisión de: {best_model_data['accuracy']:.4f}")

    # 5. Guardar el mejor modelo (Persistencia)
    model_dir = os.path.join(base_dir, 'models')
    best_model_path = os.path.join(model_dir, 'final_best_model.pkl')

    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_data['model'], f, protocol=4)

    print(f"El modelo final '{best_model_name}' ha sido guardado en {best_model_path}")


if __name__ == '__main__':
    # 1. Cargar y Preprocesar los datos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'sms_dataset_original.csv')

    sms_df = cargar_datos(data_path)
    if sms_df is not None:
        processed_df = preprocesar_texto(sms_df)

        # Ejecutar el entrenamiento y selección
        run_model_training(processed_df)