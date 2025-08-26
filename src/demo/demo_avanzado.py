#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo de funcionalidades avanzadas de visualización y evaluación de modelos.

Este script demuestra las capacidades avanzadas de visualización y evaluación
implementadas en el proyecto Tech_Sphere para el análisis de modelos de clasificación
de textos médicos.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Añadir el directorio raíz al path para importar módulos propios
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Importar funciones de visualización y evaluación
from src.visualization.visualize import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_per_label_metrics,
    plot_metrics_radar
)

# Importar utilidades comunes para demos
from src.demo.demo_utils import (
    load_model,
    save_model,
    load_data,
    prepare_train_test_data,
    vectorize_and_transform,
    train_simple_model,
    evaluate_model_basic,
    display_metrics,
    create_figures_dir
)

# Importar funciones de evaluación avanzada
from src.models.evaluate_model import (
    evaluate_model,
    visualize_results
)


def cargar_datos_y_modelo():
    """Carga los datos de prueba y el modelo entrenado."""
    print("Cargando modelo y datos...")
    
    # Intentar cargar un modelo existente
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not available_models:
        print("No se encontraron modelos entrenados en la carpeta 'models/'")
        print("Entrenando un modelo simple para la demostración...")
        
        # Cargar datos
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        data_path = os.path.join(data_dir, 'raw', 'challenge_data-18-ago.csv')
        
        # Cargar y preprocesar datos
        df = load_data(data_path)
        if df is None:
            return None, None, None, None, None, None
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = prepare_train_test_data(df)
        
        # Vectorizar y transformar
        vectorizer, mlb, X_train_vec, X_test_vec, y_train_bin, y_test_bin = vectorize_and_transform(
            X_train, X_test, y_train, y_test, max_features=5000
        )
        
        # Entrenar un modelo simple
        model = train_simple_model(X_train_vec, y_train_bin, model_type='logistic')
        
        # Guardar el modelo
        model_path = os.path.join(models_dir, 'demo_model.pkl')
        save_model(model, vectorizer, mlb, model_path)
        print(f"Modelo guardado en {model_path}")
        
        return model, vectorizer, mlb, X_test_vec, y_test_bin, df.loc[X_test.index]
    else:
        # Cargar modelo existente
        model_path = os.path.join(models_dir, available_models[0])
        model, vectorizer, mlb = load_model(model_path)
        
        # Cargar datos de prueba
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        test_data_path = os.path.join(data_dir, 'processed', 'test_data.csv')
        
        if not os.path.exists(test_data_path):
            test_data_path = os.path.join(data_dir, 'raw', 'challenge_data-18-ago.csv')
        
        # Cargar y preprocesar datos
        df_test = load_data(test_data_path)
        if df_test is None:
            return None, None, None, None, None, None
        
        # Extraer características y preparar etiquetas
        X_test = vectorizer.transform(df_test['text_processed'])
        y_test = mlb.transform(df_test['group'].str.split('|'))
        
        return model, vectorizer, mlb, X_test, y_test, df_test


def main():
    """Función principal para ejecutar la demo."""
    print("=== DEMO DE FUNCIONALIDADES AVANZADAS ===\n")
    
    # Cargar datos y modelo
    model, vectorizer, mlb, X_test, y_test, df_test = cargar_datos_y_modelo()
    if model is None:
        print("No se pudo cargar el modelo o los datos. Finalizando demo.")
        return
    
    # Evaluar modelo
    print("\n=== EVALUACIÓN DEL MODELO ===\n")
    metrics, y_pred, y_pred_proba = evaluate_model(model, vectorizer, mlb, X_test, y_test)
    
    # Mostrar métricas
    display_metrics(metrics)
    
    # Visualizar resultados
    visualize_results(metrics, y_test, y_pred, y_pred_proba, mlb)
    
    print("\n=== DEMO COMPLETADA ===\n")
    print("Revise las visualizaciones generadas en 'reports/figures/'")


if __name__ == "__main__":
    main()