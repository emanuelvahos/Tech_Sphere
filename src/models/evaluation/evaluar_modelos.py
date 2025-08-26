# -*- coding: utf-8 -*-
"""
Script para evaluar los modelos de clasificación de literatura médica.

Este script permite evaluar los modelos entrenados con un conjunto de datos de prueba.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Directorio de modelos
MODELS_DIR = os.path.join('models')
RAW_DATA_PATH = os.path.join('data', 'raw', 'challenge_data-18-ago.csv')

def load_model(model_path):
    """Carga un modelo entrenado desde un archivo pickle."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"Modelo cargado correctamente desde {model_path}")
        return model_data['model'], model_data['vectorizer'], model_data['mlb']
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None, None

def load_test_data(data_path, test_size=0.2, random_state=42):
    """Carga y prepara un conjunto de datos de prueba."""
    # Cargar datos
    df = pd.read_csv(data_path, sep=';')
    
    # Preparar etiquetas
    df['group_list'] = df['group'].apply(lambda x: x.split('|') if pd.notna(x) else [])
    
    # Dividir en entrenamiento y prueba (solo usaremos prueba)
    df_test = df.sample(frac=test_size, random_state=random_state)
    
    return df_test

def evaluate_model(model, vectorizer, mlb, test_data):
    """Evalúa un modelo con un conjunto de datos de prueba."""
    # Combinar título y resumen
    test_data['text_combined'] = test_data['title'] + " " + test_data['abstract']
    
    # Extraer características
    X_test = vectorizer.transform(test_data['text_combined'])
    
    # Binarizar etiquetas
    y_test = mlb.transform(test_data['group_list'])
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    report = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True)
    
    # Métricas adicionales
    metrics = {
        'micro_f1': f1_score(y_test, y_pred, average='micro'),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
        'samples_f1': f1_score(y_test, y_pred, average='samples'),
        'micro_precision': precision_score(y_test, y_pred, average='micro'),
        'micro_recall': recall_score(y_test, y_pred, average='micro')
    }
    
    return report, metrics

def list_available_models():
    """Lista los modelos disponibles en el directorio de modelos."""
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    return models

def main():
    # Listar modelos disponibles
    available_models = list_available_models()
    
    if not available_models:
        print("No se encontraron modelos en el directorio 'models/'.")
        return
    
    print("Modelos disponibles:")
    for i, model_name in enumerate(available_models):
        print(f"{i+1}. {model_name}")
    
    # Cargar datos de prueba
    print("\nCargando datos de prueba...")
    test_data = load_test_data(RAW_DATA_PATH)
    print(f"Datos de prueba cargados: {test_data.shape[0]} registros")
    
    # Evaluar cada modelo
    results = {}
    
    for model_name in available_models:
        print(f"\nEvaluando modelo: {model_name}")
        model_path = os.path.join(MODELS_DIR, model_name)
        
        # Cargar modelo
        model, vectorizer, mlb = load_model(model_path)
        
        if model is None:
            print(f"No se pudo cargar el modelo {model_name}.")
            continue
        
        # Evaluar modelo
        report, metrics = evaluate_model(model, vectorizer, mlb, test_data)
        
        # Guardar resultados
        results[model_name] = {
            'report': report,
            'metrics': metrics
        }
        
        # Mostrar resultados
        print(f"\nResultados para {model_name}:")
        print(f"F1 Score (micro): {metrics['micro_f1']:.4f}")
        print(f"F1 Score (macro): {metrics['macro_f1']:.4f}")
        print(f"F1 Score (weighted): {metrics['weighted_f1']:.4f}")
        print(f"Precision (micro): {metrics['micro_precision']:.4f}")
        print(f"Recall (micro): {metrics['micro_recall']:.4f}")
        
        print("\nInforme de clasificación por categoría:")
        for category in mlb.classes_:
            cat_metrics = report[category]
            print(f"  - {category}:")
            print(f"    Precision: {cat_metrics['precision']:.4f}")
            print(f"    Recall: {cat_metrics['recall']:.4f}")
            print(f"    F1-score: {cat_metrics['f1-score']:.4f}")
    
    # Comparar modelos
    print("\n=== Comparación de Modelos ===\n")
    print("Modelo                   | F1 (micro) | F1 (macro) | Precision | Recall")
    print("-" * 75)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:25} | {metrics['micro_f1']:.4f}    | {metrics['macro_f1']:.4f}    | {metrics['micro_precision']:.4f}   | {metrics['micro_recall']:.4f}")
    
    # Determinar el mejor modelo según F1 micro
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['micro_f1'])
    print(f"\nMejor modelo según F1 micro: {best_model[0]}")

if __name__ == "__main__":
    main()