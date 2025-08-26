# -*- coding: utf-8 -*-
"""
Aplicación web simple para clasificación de literatura médica.

Este script crea una aplicación web con Flask que permite a los usuarios
clasificar textos médicos utilizando los modelos entrenados.
"""

import os
import sys
import pickle
from flask import Flask, render_template, request, jsonify

# Añadir el directorio raíz al path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.preprocess import clean_text, remove_stopwords, lemmatize_text

# Inicializar la aplicación Flask
app = Flask(__name__)

# Variables globales para el modelo
MODEL = None
VECTORIZER = None
MLB = None


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
    cleaned_text = clean_text(text)
    text_without_stopwords = remove_stopwords(cleaned_text)
    lemmatized_text = lemmatize_text(text_without_stopwords)
    return lemmatized_text


def predict_category(title, abstract):
    """Predice la categoría de un texto médico basado en su título y resumen.
    
    Args:
        title (str): Título del artículo médico.
        abstract (str): Resumen del artículo médico.
        
    Returns:
        list: Lista de categorías predichas.
    """
    global MODEL, VECTORIZER, MLB
    
    if MODEL is None or VECTORIZER is None or MLB is None:
        return []
    
    # Preprocesar el texto
    processed_title = preprocess_text(title)
    processed_abstract = preprocess_text(abstract)
    
    # Combinar título y resumen
    combined_text = processed_title + " " + processed_abstract
    
    # Extraer características usando el vectorizador
    X = VECTORIZER.transform([combined_text])
    
    # Realizar la predicción
    y_pred = MODEL.predict(X)
    
    # Convertir la predicción a categorías
    categories = MLB.classes_[y_pred[0].astype(bool)]
    
    return list(categories)


def get_prediction_probabilities(title, abstract):
    """Obtiene las probabilidades de predicción para cada categoría.
    
    Args:
        title (str): Título del artículo médico.
        abstract (str): Resumen del artículo médico.
        
    Returns:
        dict: Diccionario con las probabilidades para cada categoría.
    """
    global MODEL, VECTORIZER, MLB
    
    if MODEL is None or VECTORIZER is None or MLB is None:
        return {}
    
    # Preprocesar el texto
    processed_title = preprocess_text(title)
    processed_abstract = preprocess_text(abstract)
    
    # Combinar título y resumen
    combined_text = processed_title + " " + processed_abstract
    
    # Extraer características usando el vectorizador
    X = VECTORIZER.transform([combined_text])
    
    # Obtener probabilidades de predicción
    try:
        y_prob = MODEL.predict_proba(X)
        
        # Crear diccionario de probabilidades por categoría
        probabilities = {}
        for i, category in enumerate(MLB.classes_):
            probabilities[category] = float(y_prob[i][0][1])  # Probabilidad de la clase positiva
            
        return probabilities
    except:
        # Si el modelo no soporta predict_proba, devolver None
        return {}


@app.route('/')
def home():
    """Ruta principal de la aplicación."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Ruta para realizar predicciones."""
    # Obtener datos del formulario
    title = request.form.get('title', '')
    abstract = request.form.get('abstract', '')
    
    # Validar entrada
    if not title and not abstract:
        return jsonify({
            'error': 'Se requiere al menos un título o un resumen'
        }), 400
    
    # Realizar predicción
    categories = predict_category(title, abstract)
    probabilities = get_prediction_probabilities(title, abstract)
    
    # Preparar respuesta
    response = {
        'categories': categories,
        'probabilities': probabilities
    }
    
    return jsonify(response)


def create_templates_folder():
    """Crea la carpeta de templates y el archivo index.html si no existen."""
    templates_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Crear archivo index.html si no existe
    index_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_path):
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clasificación de Literatura Médica</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f8f9fa;
        }
        .card {
            margin-bottom: 1.5rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .result-category {
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            border-radius: 0.25rem;
            background-color: #e9ecef;
        }
        .progress {
            height: 0.5rem;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Clasificación de Literatura Médica</h1>
        
        <div class="row">
            <div class="col-md-8 offset-md-2">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="card-title mb-0">Ingrese el texto médico a clasificar</h5>
                    </div>
                    <div class="card-body">
                        <form id="prediction-form">
                            <div class="mb-3">
                                <label for="title" class="form-label">Título del artículo</label>
                                <input type="text" class="form-control" id="title" name="title" placeholder="Ingrese el título del artículo médico">
                            </div>
                            <div class="mb-3">
                                <label for="abstract" class="form-label">Resumen (Abstract)</label>
                                <textarea class="form-control" id="abstract" name="abstract" rows="6" placeholder="Ingrese el resumen del artículo médico"></textarea>
                            </div>
                            <div class="d-grid">
                                <button type="submit" class="btn btn-primary">Clasificar</button>
                            </div>
                        </form>
                    </div>
                </div>
                
                <div id="results" class="card d-none">
                    <div class="card-header bg-success text-white">
                        <h5 class="card-title mb-0">Resultados de la clasificación</h5>
                    </div>
                    <div class="card-body">
                        <h6>Categorías predichas:</h6>
                        <div id="categories-container" class="mb-4">
                            <!-- Las categorías se insertarán aquí -->
                        </div>
                        
                        <h6>Probabilidades por categoría:</h6>
                        <div id="probabilities-container">
                            <!-- Las probabilidades se insertarán aquí -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const resultsCard = document.getElementById('results');
            const categoriesContainer = document.getElementById('categories-container');
            const probabilitiesContainer = document.getElementById('probabilities-container');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Error en la solicitud');
                }
                
                const data = await response.json();
                
                // Mostrar categorías predichas
                categoriesContainer.innerHTML = '';
                if (data.categories && data.categories.length > 0) {
                    data.categories.forEach(category => {
                        const categoryDiv = document.createElement('div');
                        categoryDiv.className = 'result-category';
                        categoryDiv.textContent = category;
                        categoriesContainer.appendChild(categoryDiv);
                    });
                } else {
                    categoriesContainer.innerHTML = '<p>No se pudo clasificar el texto en ninguna categoría.</p>';
                }
                
                // Mostrar probabilidades
                probabilitiesContainer.innerHTML = '';
                if (data.probabilities && Object.keys(data.probabilities).length > 0) {
                    // Ordenar categorías por probabilidad descendente
                    const sortedCategories = Object.entries(data.probabilities)
                        .sort((a, b) => b[1] - a[1]);
                    
                    sortedCategories.forEach(([category, probability]) => {
                        const probabilityDiv = document.createElement('div');
                        probabilityDiv.className = 'mb-2';
                        
                        const label = document.createElement('div');
                        label.className = 'd-flex justify-content-between';
                        label.innerHTML = `<span>${category}</span><span>${(probability * 100).toFixed(2)}%</span>`;
                        
                        const progress = document.createElement('div');
                        progress.className = 'progress';
                        progress.innerHTML = `<div class="progress-bar" role="progressbar" style="width: ${probability * 100}%" aria-valuenow="${probability * 100}" aria-valuemin="0" aria-valuemax="100"></div>`;
                        
                        probabilityDiv.appendChild(label);
                        probabilityDiv.appendChild(progress);
                        probabilitiesContainer.appendChild(probabilityDiv);
                    });
                } else {
                    probabilitiesContainer.innerHTML = '<p>No hay información de probabilidades disponible.</p>';
                }
                
                // Mostrar resultados
                resultsCard.classList.remove('d-none');
                
            } catch (error) {
                console.error('Error:', error);
                alert('Ocurrió un error al procesar la solicitud.');
            }
        });
    </script>
</body>
</html>
''')


def main():
    """Función principal para iniciar la aplicación."""
    global MODEL, VECTORIZER, MLB
    
    # Crear carpeta de templates
    create_templates_folder()
    
    # Cargar el modelo
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not available_models:
        print("No se encontraron modelos entrenados en la carpeta 'models/'")
        print("Por favor, entrene un modelo primero usando train_model.py")
        return
    
    # Usar el primer modelo disponible (o permitir selección)
    model_path = os.path.join(models_dir, available_models[0])
    MODEL, VECTORIZER, MLB = load_model(model_path)
    
    if MODEL is None:
        print("Error al cargar el modelo. La aplicación no funcionará correctamente.")
    
    # Iniciar la aplicación
    print("Iniciando aplicación web en http://127.0.0.1:5000/")
    app.run(debug=True)


if __name__ == "__main__":
    main()