# -*- coding: utf-8 -*-
"""
Script para probar los modelos entrenados con datos de entrada del usuario.

Este script permite cargar un modelo entrenado y utilizarlo para clasificar
textos médicos proporcionados por el usuario.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Añadir el directorio raíz al path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocess import clean_text, remove_stopwords, lemmatize_tokens, tokenize_text
from src.features.feature_extraction import extract_tfidf_features


def load_model(model_path):
    """Carga un modelo entrenado desde un archivo pickle.
    
    Args:
        model_path (str): Ruta al archivo del modelo guardado.
        
    Returns:
        tuple: (modelo, vectorizador, mlb) donde:
            - modelo es el modelo de clasificación entrenado
            - vectorizador es el vectorizador TF-IDF entrenado
            - mlb es el MultiLabelBinarizer entrenado
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"Modelo cargado correctamente desde {model_path}")
        return model_data['model'], model_data['vectorizer'], model_data['mlb']
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None, None


def preprocess_text(text):
    """Preprocesa un texto aplicando limpieza, eliminación de stopwords y lematización.
    
    Args:
        text (str): Texto a preprocesar.
        
    Returns:
        str: Texto preprocesado.
    """
    # Usar la función tokenize_text que ya incluye limpieza, eliminación de stopwords y lematización
    tokens = tokenize_text(text, remove_stop=True, lemmatize=True)
    return ' '.join(tokens)


def predict_category(title, abstract, model, vectorizer, mlb):
    """Predice la categoría de un texto médico basado en su título y resumen.
    
    Args:
        title (str): Título del artículo médico.
        abstract (str): Resumen del artículo médico.
        model: Modelo entrenado para la clasificación.
        vectorizer: Vectorizador TF-IDF entrenado.
        mlb: MultiLabelBinarizer entrenado.
        
    Returns:
        list: Lista de categorías predichas.
    """
    # Preprocesar el texto
    processed_title = preprocess_text(title)
    processed_abstract = preprocess_text(abstract)
    
    # Combinar título y resumen
    combined_text = processed_title + " " + processed_abstract
    
    # Extraer características usando el vectorizador
    X = vectorizer.transform([combined_text])
    
    # Realizar la predicción
    y_pred = model.predict(X)
    
    # Convertir la predicción a categorías
    categories = mlb.classes_[y_pred[0].astype(bool)]
    
    return list(categories)


def get_prediction_probabilities(title, abstract, model, vectorizer, mlb):
    """Obtiene las probabilidades de predicción para cada categoría.
    
    Args:
        title (str): Título del artículo médico.
        abstract (str): Resumen del artículo médico.
        model: Modelo entrenado para la clasificación.
        vectorizer: Vectorizador TF-IDF entrenado.
        mlb: MultiLabelBinarizer entrenado.
        
    Returns:
        dict: Diccionario con las probabilidades para cada categoría.
    """
    # Preprocesar el texto
    processed_title = preprocess_text(title)
    processed_abstract = preprocess_text(abstract)
    
    # Combinar título y resumen
    combined_text = processed_title + " " + processed_abstract
    
    # Extraer características usando el vectorizador
    X = vectorizer.transform([combined_text])
    
    # Obtener probabilidades de predicción
    try:
        y_prob = model.predict_proba(X)
        
        # Crear diccionario de probabilidades por categoría
        probabilities = {}
        for i, category in enumerate(mlb.classes_):
            probabilities[category] = y_prob[i][0][1]  # Probabilidad de la clase positiva
            
        return probabilities
    except:
        # Si el modelo no soporta predict_proba, devolver None
        return None


def main():
    """Función principal para interactuar con el usuario y probar el modelo."""
    print("\n=== SISTEMA DE CLASIFICACIÓN DE LITERATURA MÉDICA ===\n")
    
    # Listar modelos disponibles
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not available_models:
        print("No se encontraron modelos entrenados en la carpeta 'models/'")
        print("Por favor, entrene un modelo primero usando train_model.py")
        return
    
    print("Modelos disponibles:")
    for i, model_file in enumerate(available_models):
        print(f"  {i+1}. {model_file}")
    
    # Seleccionar modelo
    try:
        model_idx = int(input("\nSeleccione un modelo por número: ")) - 1
        if model_idx < 0 or model_idx >= len(available_models):
            print("Selección inválida")
            return
    except ValueError:
        print("Por favor, ingrese un número válido")
        return
    
    model_path = os.path.join(models_dir, available_models[model_idx])
    model, vectorizer, mlb = load_model(model_path)
    
    if model is None:
        return
    
    print("\nEl modelo puede clasificar textos en las siguientes categorías:")
    for category in mlb.classes_:
        print(f"  - {category}")
    
    while True:
        print("\n" + "-"*50)
        print("Ingrese un texto médico para clasificar (o 'salir' para terminar)")
        
        title = input("\nTítulo del artículo: ")
        if title.lower() == 'salir':
            break
            
        abstract = input("Resumen del artículo: ")
        if abstract.lower() == 'salir':
            break
        
        # Realizar predicción
        predicted_categories = predict_category(title, abstract, model, vectorizer, mlb)
        
        print("\nCategorías predichas:")
        if predicted_categories:
            for category in predicted_categories:
                print(f"  - {category}")
        else:
            print("  No se pudo clasificar el texto en ninguna categoría")
        
        # Mostrar probabilidades si están disponibles
        probabilities = get_prediction_probabilities(title, abstract, model, vectorizer, mlb)
        if probabilities:
            print("\nProbabilidades por categoría:")
            for category, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {category}: {prob:.4f}")


if __name__ == "__main__":
    main()