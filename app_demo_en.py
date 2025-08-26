# -*- coding: utf-8 -*-
"""
Web application to demonstrate the use of medical literature classification models.

This application allows users to enter medical article titles and abstracts
and get category predictions using the trained models.
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify

# Models directory
MODELS_DIR = os.path.join('models')

# Create Flask application
app = Flask(__name__)

# Global variables to store the loaded model
loaded_model = None
loaded_vectorizer = None
loaded_mlb = None
loaded_model_name = None

def load_model(model_path):
    """Loads a trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        print(f"Model successfully loaded from {model_path}")
        return model_data['model'], model_data['vectorizer'], model_data['mlb']
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None, None, None

def list_available_models():
    """Lists the available models in the models directory."""
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    return models

def predict_category(title, abstract, model, vectorizer, mlb):
    """Predicts the category of a medical text based on its title and abstract."""
    # Combine title and abstract
    combined_text = title + " " + abstract
    
    # Extract features using the vectorizer
    X = vectorizer.transform([combined_text])
    
    # Make the prediction
    y_pred = model.predict(X)
    
    # Convert the prediction to categories
    categories = mlb.classes_[y_pred[0].astype(bool)]
    
    return list(categories)

def get_prediction_probabilities(title, abstract, model, vectorizer, mlb):
    """Gets the prediction probabilities for each category."""
    # Combine title and abstract
    combined_text = title + " " + abstract
    
    # Extract features using the vectorizer
    X = vectorizer.transform([combined_text])
    
    # Get prediction probabilities if the model supports it
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
        # Create a dictionary of category -> probability
        probs = {cat: float(prob) for cat, prob in zip(mlb.classes_, y_proba[0])}
        return probs
    else:
        # If the model doesn't support probabilities, return binary decision
        y_pred = model.predict(X)
        # Create a dictionary of category -> 1.0 if present, 0.0 if not
        probs = {cat: 1.0 if pred else 0.0 for cat, pred in zip(mlb.classes_, y_pred[0])}
        return probs

@app.route('/')
def index():
    """Main page of the application."""
    models = list_available_models()
    return render_template('app_demo_en.html', models=models, current_model=loaded_model_name)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """Loads a selected model."""
    global loaded_model, loaded_vectorizer, loaded_mlb, loaded_model_name
    
    model_name = request.form.get('model')
    if not model_name:
        return jsonify({'success': False, 'error': 'No model was selected'})
    
    model_path = os.path.join(MODELS_DIR, model_name)
    model, vectorizer, mlb = load_model(model_path)
    
    if model is None:
        return jsonify({'success': False, 'error': f'Could not load model {model_name}'})
    
    loaded_model = model
    loaded_vectorizer = vectorizer
    loaded_mlb = mlb
    loaded_model_name = model_name
    
    return jsonify({'success': True, 'model': model_name})

@app.route('/predict', methods=['POST'])
def predict():
    """Makes a prediction based on the provided title and abstract."""
    if loaded_model is None:
        return jsonify({'success': False, 'error': 'No model is loaded'})
    
    title = request.form.get('title', '')
    abstract = request.form.get('abstract', '')
    
    if not title and not abstract:
        return jsonify({'success': False, 'error': 'At least a title or an abstract is required'})
    
    categories = predict_category(title, abstract, loaded_model, loaded_vectorizer, loaded_mlb)
    probabilities = get_prediction_probabilities(title, abstract, loaded_model, loaded_vectorizer, loaded_mlb)
    
    return jsonify({
        'success': True,
        'categories': categories,
        'probabilities': probabilities
    })

# Load the first available model when starting the application
def load_initial_model():
    """Loads the first available model when starting the application."""
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
            print(f"Initial model loaded: {models[0]}")

if __name__ == '__main__':
    # Load the initial model
    load_initial_model()
    
    # Start the application
    app.run(debug=True)