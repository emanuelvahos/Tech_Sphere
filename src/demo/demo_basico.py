#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo básica de visualización y evaluación de modelos.

Este script demuestra las capacidades básicas de visualización y evaluación
implementadas en el proyecto Tech_Sphere para el análisis de modelos de clasificación
de textos médicos, sin depender de bibliotecas externas complejas.
"""

import os
import sys
import matplotlib.pyplot as plt

# Añadir el directorio raíz al path para importar módulos propios
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Importar funciones de visualización básicas
from src.visualization.visualize import plot_confusion_matrix, plot_metrics_radar

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
    create_figures_dir,
    plot_simple_confusion_matrix,
    plot_simple_metrics_radar
)


def cargar_o_crear_modelo():
    """Carga un modelo existente o crea uno nuevo si no existe."""
    print("Cargando o creando modelo...")
    
    # Ruta al modelo
    model_path = os.path.join('models', 'demo_model_simple.pkl')
    
    # Verificar si el modelo existe
    if os.path.exists(model_path):
        print(f"Cargando modelo existente desde {model_path}")
        model, vectorizer, mlb = load_model(model_path)
        if model is not None:
            return model, vectorizer, mlb, True
    
    print("No se encontró un modelo existente. Creando uno nuevo...")
    return None, None, None, False


def entrenar_modelo(df):
    """Entrena un modelo simple de clasificación."""
    print("Entrenando modelo...")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = prepare_train_test_data(df)
    
    # Vectorizar y transformar
    vectorizer, mlb, X_train_vec, X_test_vec, y_train_bin, y_test_bin = vectorize_and_transform(
        X_train, X_test, y_train, y_test, max_features=5000
    )
    
    # Entrenar modelo
    model = train_simple_model(X_train_vec, y_train_bin, model_type='logistic')
    
    # Guardar modelo
    model_path = os.path.join('models', 'demo_model_simple.pkl')
    save_model(model, vectorizer, mlb, model_path)
    
    return model, vectorizer, mlb, X_test_vec, y_test_bin


def evaluar_modelo(model, X_test, y_test, mlb):
    """Evalúa el modelo y muestra métricas básicas."""
    print("\nEvaluando modelo...")
    
    # Evaluar modelo
    y_pred, metrics = evaluate_model_basic(model, X_test, y_test)
    
    # Mostrar métricas
    display_metrics(metrics)
    
    # Visualizar resultados
    print("\nGenerando visualizaciones...")
    
    # Crear directorio para figuras
    figures_dir = create_figures_dir()
    
    # Generar matriz de confusión simple
    confusion_matrix_path = os.path.join(figures_dir, "confusion_matrix_global.png")
    plot_simple_confusion_matrix(y_test, y_pred, save_path=confusion_matrix_path)
    
    # Generar gráfico de radar para métricas
    metrics_radar_path = os.path.join(figures_dir, "metrics_radar.png")
    plot_simple_metrics_radar(metrics, save_path=metrics_radar_path)
    
    print(f"Visualizaciones guardadas en {figures_dir}")
    
    return y_pred


def main():
    """Función principal."""
    print("=== DEMO BÁSICA DE EVALUACIÓN DE MODELOS ===\n")
    
    # Cargar o crear modelo
    model, vectorizer, mlb, modelo_cargado = cargar_o_crear_modelo()
    
    # Cargar datos
    data_path = os.path.join('data', 'raw', 'challenge_data-18-ago.csv')
    df = load_data(data_path)
    if df is None:
        print("No se pudieron cargar los datos. Finalizando demo.")
        return
    
    # Si no se cargó un modelo, entrenar uno nuevo
    if not modelo_cargado:
        model, vectorizer, mlb, X_test, y_test = entrenar_modelo(df)
    else:
        # Preparar datos de prueba
        _, X_test, _, y_test = prepare_train_test_data(df)
        _, _, _, X_test_vec, _, y_test_bin = vectorize_and_transform(
            None, X_test, None, y_test, max_features=5000
        )
        X_test, y_test = X_test_vec, y_test_bin
    
    # Evaluar modelo
    evaluar_modelo(model, X_test, y_test, mlb)
    
    print("\n=== DEMO COMPLETADA ===\n")
    print("Revise las visualizaciones generadas en 'reports/figures/'")


if __name__ == "__main__":
    main()