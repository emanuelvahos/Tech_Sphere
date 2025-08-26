# -*- coding: utf-8 -*-
"""
Script simplificado para demostrar el uso del modelo entrenado.
"""

import os
import pickle
import pandas as pd
import numpy as np

# Ruta al modelo entrenado
MODEL_PATH = os.path.join('models', 'svm_model.pkl')

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

def main():
    # Cargar el modelo
    print("Cargando modelo...")
    model, vectorizer, mlb = load_model(MODEL_PATH)
    
    if model is None:
        print("No se pudo cargar el modelo. Asegúrate de que el archivo existe.")
        return
    
    print("\nModelo cargado correctamente. Puedes clasificar textos médicos.")
    print("Categorías disponibles:", mlb.classes_)
    
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

    # Permitir al usuario ingresar su propio texto
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

if __name__ == "__main__":
    main()