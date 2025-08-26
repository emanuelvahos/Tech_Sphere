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
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Añadir el directorio raíz al path para importar módulos propios
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Importar funciones de visualización y evaluación
from src.visualization.visualize import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_per_label_metrics,
    plot_metrics_radar
)

from src.models.evaluate_model import (
    load_model,
    load_test_data,
    preprocess_dataset,
    evaluate_model,
    display_metrics,
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
        
        if not os.path.exists(data_path):
            print(f"No se encontró el archivo de datos en {data_path}")
            return None, None, None, None, None, None
        
        # Cargar y preprocesar datos
        df = pd.read_csv(data_path)
        df['text_processed'] = df['title'] + " " + df['abstract']
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            df['text_processed'], 
            df['group'].str.split('|'),
            test_size=0.2, 
            random_state=42
        )
        
        # Vectorizar texto
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Preparar etiquetas
        mlb = MultiLabelBinarizer()
        y_train_bin = mlb.fit_transform(y_train)
        y_test_bin = mlb.transform(y_test)
        
        # Entrenar un modelo simple
        model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        model.fit(X_train_vec, y_train_bin)
        
        # Guardar el modelo
        model_path = os.path.join(models_dir, 'demo_model.pkl')
        joblib.dump((model, vectorizer, mlb), model_path)
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
        
        df_test = load_test_data(test_data_path)
        if df_test is None:
            return None, None, None, None, None, None
        
        # Preprocesar datos
        df_processed = preprocess_dataset(df_test)
        
        # Extraer características
        X_test = vectorizer.transform(df_processed['text_processed'])
        
        # Preparar etiquetas
        y_test = mlb.transform(df_processed['group'].str.split('|'))
        
        return model, vectorizer, mlb, X_test, y_test, df_processed


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