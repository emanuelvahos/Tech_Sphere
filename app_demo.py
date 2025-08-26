# -*- coding: utf-8 -*-
"""
Aplicación web para demostrar el uso de los modelos de clasificación de literatura médica.

Esta aplicación permite a los usuarios ingresar títulos y resúmenes de artículos médicos
y obtener predicciones de categorías utilizando los modelos entrenados.
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify

# Directorio de modelos
MODELS_DIR = os.path.join('models')

# Crear aplicación Flask
app = Flask(__name__)

# Variable global para almacenar el modelo cargado
loaded_model = None
loaded_vectorizer = None
loaded_mlb = None
loaded_model_name = None

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

def list_available_models():
    """Lista los modelos disponibles en el directorio de modelos."""
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    return models

def predict_category(title, abstract, model, vectorizer, mlb):
    """Predice la categoría de un texto médico basado en su título y resumen."""
    # Combinar título y resumen
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
        probs = {cat: float(prob) for cat, prob in zip(mlb.classes_, y_proba[0])}
        return probs
    else:
        # Si el modelo no soporta probabilidades, devolver decisión binaria
        y_pred = model.predict(X)
        # Crear un diccionario de categoría -> 1.0 si está presente, 0.0 si no
        probs = {cat: 1.0 if pred else 0.0 for cat, pred in zip(mlb.classes_, y_pred[0])}
        return probs

@app.route('/')
def index():
    """Página principal de la aplicación."""
    models = list_available_models()
    return render_template('app_demo.html', models=models, current_model=loaded_model_name)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """Carga un modelo seleccionado."""
    global loaded_model, loaded_vectorizer, loaded_mlb, loaded_model_name
    
    model_name = request.form.get('model')
    if not model_name:
        return jsonify({'success': False, 'error': 'No se seleccionó ningún modelo'})
    
    model_path = os.path.join(MODELS_DIR, model_name)
    model, vectorizer, mlb = load_model(model_path)
    
    if model is None:
        return jsonify({'success': False, 'error': f'No se pudo cargar el modelo {model_name}'})
    
    loaded_model = model
    loaded_vectorizer = vectorizer
    loaded_mlb = mlb
    loaded_model_name = model_name
    
    return jsonify({'success': True, 'model': model_name})

@app.route('/predict', methods=['POST'])
def predict():
    """Realiza una predicción basada en el título y resumen proporcionados."""
    if loaded_model is None:
        return jsonify({'success': False, 'error': 'No hay ningún modelo cargado'})
    
    title = request.form.get('title', '')
    abstract = request.form.get('abstract', '')
    
    if not title and not abstract:
        return jsonify({'success': False, 'error': 'Se requiere al menos un título o un resumen'})
    
    categories = predict_category(title, abstract, loaded_model, loaded_vectorizer, loaded_mlb)
    probabilities = get_prediction_probabilities(title, abstract, loaded_model, loaded_vectorizer, loaded_mlb)
    
    return jsonify({
        'success': True,
        'categories': categories,
        'probabilities': probabilities
    })

# Cargar el primer modelo disponible al iniciar la aplicación
def load_initial_model():
    """Carga el primer modelo disponible al iniciar la aplicación."""
    global loaded_model, loaded_vectorizer, loaded_mlb, loaded_model_name
    
    models = list_available_models()
    if models:
        model_path = os.path.join(MODELS_DIR, models[0])
        model, vectorizer, mlb = load_model(model_path)
        
        if model is not None:
            loaded_model = model
            loaded_vectorizer = vectorizer
            loaded_mlb = mlb
            loaded_model_name = models[0]
            print(f"Modelo inicial cargado: {models[0]}")

if __name__ == '__main__':
    # Cargar el modelo inicial
    load_initial_model()
    
    # Iniciar la aplicación
    app.run(debug=True)