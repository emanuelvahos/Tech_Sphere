# -*- coding: utf-8 -*-
"""
Script unificado para probar los modelos de clasificación de literatura médica.

Este script combina las funcionalidades de demo_modelo.py y probar_modelos.py,
permitiendo cargar un modelo específico o seleccionar entre los disponibles.
"""

import os
import pickle
import pandas as pd
import numpy as np
import argparse

# Configuración de rutas
MODELS_DIR = os.path.join('models')
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')

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

def predict_category(title, abstract, model, vectorizer, mlb):
    """Predice la categoría de un texto médico basado en su título y resumen."""
    # Combinar título y resumen (sin preprocesamiento complejo para esta demo)
    combined_text = title + " " + abstract
    
    # Extraer características usando el vectorizador
    X = vectorizer.transform([combined_text])
    
    # Realizar la predicción
    y_pred = model.predict(X)
    
    # Convertir la predicción a categorías
    categories = mlb.classes_[y_pred[0].astype(bool)]
    
    return list(categories)

def get_prediction_probabilities(title, abstract, model, vectorizer, mlb):
    """Obtiene las probabilidades de predicción para cada categoría."""
    # Combinar título y resumen
    combined_text = title + " " + abstract
    
    # Extraer características usando el vectorizador
    X = vectorizer.transform([combined_text])
    
    # Obtener probabilidades de predicción si el modelo lo soporta
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
        # Crear un diccionario de categoría -> probabilidad
        probs = {cat: prob for cat, prob in zip(mlb.classes_, y_proba[0])}
        return probs
    else:
        # Si el modelo no soporta probabilidades, devolver decisión binaria
        y_pred = model.predict(X)
        # Crear un diccionario de categoría -> 1.0 si está presente, 0.0 si no
        probs = {cat: 1.0 if pred else 0.0 for cat, pred in zip(mlb.classes_, y_pred[0])}
        return probs

def list_available_models():
    """Lista los modelos disponibles en el directorio de modelos."""
    if not os.path.exists(MODELS_DIR):
        print(f"El directorio {MODELS_DIR} no existe.")
        return []
    
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    return models

def select_model_interactive():
    """Permite al usuario seleccionar un modelo de forma interactiva."""
    # Listar modelos disponibles
    available_models = list_available_models()
    
    if not available_models:
        print("No se encontraron modelos en el directorio 'models/'.")
        return None
    
    print("Modelos disponibles:")
    for i, model_name in enumerate(available_models):
        print(f"{i+1}. {model_name}")
    
    # Seleccionar un modelo
    while True:
        try:
            selection = int(input("\nSelecciona un modelo por número (o 0 para salir): "))
            if selection == 0:
                return None
            if 1 <= selection <= len(available_models):
                break
            print(f"Por favor, ingresa un número entre 1 y {len(available_models)}.")
        except ValueError:
            print("Por favor, ingresa un número válido.")
    
    model_path = os.path.join(MODELS_DIR, available_models[selection-1])
    return model_path

def test_model_with_examples(model, vectorizer, mlb):
    """Prueba el modelo con ejemplos predefinidos."""
    # Ejemplos de textos médicos para clasificar
    examples = [
        {
            "title": "Efectos cardiovasculares de los inhibidores de la ECA en pacientes hipertensos",
            "abstract": "Este estudio evalúa los efectos de los inhibidores de la enzima convertidora de angiotensina (ECA) en la presión arterial y la función cardíaca en pacientes con hipertensión. Los resultados muestran una reducción significativa en la presión arterial sistólica y diastólica, así como mejoras en la función ventricular izquierda."
        },
        {
            "title": "Biomarcadores para la detección temprana del cáncer de páncreas",
            "abstract": "Investigación sobre nuevos biomarcadores séricos para la detección temprana del adenocarcinoma pancreático. El estudio identifica un panel de cinco proteínas que, en combinación, muestran una sensibilidad del 85% y una especificidad del 90% para la detección de cáncer de páncreas en estadios tempranos."
        },
        {
            "title": "Evaluación de la función renal en pacientes con cirrosis hepática",
            "abstract": "Este estudio evalúa diferentes métodos para medir la función renal en pacientes con cirrosis hepática avanzada. Los resultados indican que la cistatina C sérica es un marcador más preciso de la tasa de filtración glomerular que la creatinina en estos pacientes, especialmente en presencia de ascitis."
        },
        {
            "title": "Nuevos enfoques en el tratamiento de la enfermedad de Alzheimer",
            "abstract": "Revisión de las terapias emergentes para la enfermedad de Alzheimer, incluyendo inhibidores de la proteína tau, moduladores de la microglía y terapias basadas en anticuerpos monoclonales. Se discuten los resultados preliminares de ensayos clínicos recientes y las perspectivas futuras para el tratamiento de esta enfermedad neurodegenerativa."
        }
    ]
    
    # Clasificar cada ejemplo
    for i, example in enumerate(examples):
        print(f"\n--- Ejemplo {i+1} ---")
        print(f"Título: {example['title']}")
        print(f"Resumen: {example['abstract'][:100]}...")
        
        # Predecir categorías
        categories = predict_category(example['title'], example['abstract'], model, vectorizer, mlb)
        print(f"Categorías predichas: {categories}")
        
        # Obtener probabilidades
        probs = get_prediction_probabilities(example['title'], example['abstract'], model, vectorizer, mlb)
        print("Probabilidades:")
        for cat, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            if prob > 0:
                print(f"  - {cat}: {prob:.4f}")

def interactive_prediction(model, vectorizer, mlb):
    """Permite al usuario ingresar textos para clasificar interactivamente."""
    print("\n--- Ingresa tu propio texto ---")
    print("(Presiona Enter sin texto para salir)")
    
    while True:
        title = input("\nTítulo: ")
        if not title:
            break
            
        abstract = input("Resumen: ")
        
        # Predecir categorías
        categories = predict_category(title, abstract, model, vectorizer, mlb)
        print(f"Categorías predichas: {categories}")
        
        # Obtener probabilidades
        probs = get_prediction_probabilities(title, abstract, model, vectorizer, mlb)
        print("Probabilidades:")
        for cat, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            if prob > 0:
                print(f"  - {cat}: {prob:.4f}")

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Probar modelos de clasificación de literatura médica')
    parser.add_argument('--model', type=str, help='Ruta al modelo específico a cargar')
    parser.add_argument('--interactive-select', action='store_true', 
                        help='Seleccionar modelo interactivamente')
    parser.add_argument('--skip-examples', action='store_true', 
                        help='Omitir ejemplos predefinidos')
    parser.add_argument('--skip-interactive', action='store_true', 
                        help='Omitir modo interactivo')
    args = parser.parse_args()
    
    # Determinar qué modelo cargar
    model_path = None
    if args.interactive_select:
        model_path = select_model_interactive()
        if model_path is None:
            return
    elif args.model:
        model_path = args.model
    else:
        # Usar modelo por defecto
        model_path = DEFAULT_MODEL_PATH
    
    # Cargar el modelo
    print(f"Cargando modelo desde {model_path}...")
    model, vectorizer, mlb = load_model(model_path)
    
    if model is None:
        print("No se pudo cargar el modelo. Asegúrate de que el archivo existe.")
        return
    
    print("\nModelo cargado correctamente. Puedes clasificar textos médicos.")
    print("Categorías disponibles:", mlb.classes_)
    
    # Ejecutar ejemplos predefinidos si no se omiten
    if not args.skip_examples:
        test_model_with_examples(model, vectorizer, mlb)
    
    # Modo interactivo si no se omite
    if not args.skip_interactive:
        interactive_prediction(model, vectorizer, mlb)

if __name__ == "__main__":
    main()