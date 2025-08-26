# -*- coding: utf-8 -*-
"""
Script unificado para crear y entrenar modelos de clasificación de literatura médica.

Este script combina las funcionalidades de train_models.py, train_model_demo.py y crear_modelos.py
en un único script más eficiente y completo. Permite entrenar varios tipos de modelos
(SVM, Random Forest, Logistic Regression) con diferentes configuraciones y guardarlos
para su uso posterior.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Añadir el directorio raíz al path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data.preprocess import tokenize_text

# Configurar directorios
RAW_DATA_PATH = os.path.join('data', 'raw', 'challenge_data-18-ago.csv')
PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')

# Crear directorios si no existen
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def preprocess_text(text):
    """Preprocesa un texto aplicando limpieza, eliminación de stopwords y lematización."""
    if pd.isna(text):
        return ""
    # Usar la función tokenize_text que ya incluye limpieza, eliminación de stopwords y lematización
    tokens = tokenize_text(text, remove_stop=True, lemmatize=True)
    return ' '.join(tokens)

def load_and_preprocess_data(data_path=RAW_DATA_PATH, verbose=True):
    """Carga y preprocesa los datos desde el archivo CSV."""
    if verbose:
        print("Cargando datos...")
    df = pd.read_csv(data_path, sep=';')
    if verbose:
        print(f"Datos cargados: {df.shape[0]} registros")

    if verbose:
        print("Preprocesando textos...")
    
    # Preprocesar títulos y resúmenes
    df['title_processed'] = df['title'].apply(preprocess_text)
    df['abstract_processed'] = df['abstract'].apply(preprocess_text)
    
    # Combinar título y resumen para el entrenamiento
    df['text_combined'] = df['title_processed'] + ' ' + df['abstract_processed']
    
    # Convertir categorías a formato de lista
    df['categories'] = df['categories'].apply(lambda x: x.split(',') if pd.notna(x) else [])
    
    if verbose:
        print("Preprocesamiento completado")
    
    return df

def extract_features_and_labels(df, max_features=10000, verbose=True):
    """Extrae características y etiquetas de los datos preprocesados."""
    if verbose:
        print("Extrayendo características...")
    
    # Vectorizar texto usando TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['text_combined'])
    
    # Codificar etiquetas multi-clase
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['categories'])
    
    if verbose:
        print(f"Características extraídas: {X.shape[1]} características")
        print(f"Etiquetas codificadas: {y.shape[1]} categorías")
    
    return X, y, vectorizer, mlb

def split_dataset(X, y, test_size=0.2, random_state=42, verbose=True):
    """Divide el conjunto de datos en entrenamiento y prueba."""
    if verbose:
        print("Dividiendo conjunto de datos...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if verbose:
        print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
        print(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='svm', verbose=True):
    """Entrena un modelo según el tipo especificado."""
    if verbose:
        print(f"Entrenando modelo {model_type}...")
    
    if model_type.lower() == 'svm':
        model = OneVsRestClassifier(LinearSVC(random_state=42))
    elif model_type.lower() == 'rf':
        model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    elif model_type.lower() == 'lr':
        model = OneVsRestClassifier(LogisticRegression(random_state=42))
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    model.fit(X_train, y_train)
    
    if verbose:
        print(f"Modelo {model_type} entrenado correctamente")
    
    return model

def evaluate_model(model, X_test, y_test, mlb, verbose=True):
    """Evalúa el rendimiento del modelo en el conjunto de prueba."""
    if verbose:
        print("Evaluando modelo...")
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nInforme de clasificación detallado:")
        print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_model(model, vectorizer, mlb, model_name, metrics=None, verbose=True):
    """Guarda el modelo entrenado junto con el vectorizador y el codificador de etiquetas."""
    model_path = os.path.join(MODELS_DIR, model_name)
    
    # Crear diccionario con todos los componentes necesarios
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'mlb': mlb
    }
    
    # Añadir métricas si están disponibles
    if metrics:
        model_data['metrics'] = metrics
    
    # Guardar en formato pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    if verbose:
        print(f"Modelo guardado en {model_path}")

def train_all_models(data_path=RAW_DATA_PATH, max_features=10000, test_size=0.2, random_state=42, verbose=True):
    """Entrena todos los modelos disponibles y los guarda."""
    # Cargar y preprocesar datos
    df = load_and_preprocess_data(data_path, verbose)
    
    # Extraer características y etiquetas
    X, y, vectorizer, mlb = extract_features_and_labels(df, max_features, verbose)
    
    # Dividir conjunto de datos
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size, random_state, verbose)
    
    # Definir modelos a entrenar
    models_to_train = [
        {'type': 'svm', 'name': 'svm_model.pkl'},
        {'type': 'rf', 'name': 'random_forest_model.pkl'},
        {'type': 'lr', 'name': 'logistic_regression_model.pkl'}
    ]
    
    # Entrenar y evaluar cada modelo
    results = {}
    for model_info in models_to_train:
        model_type = model_info['type']
        model_name = model_info['name']
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Procesando modelo: {model_type}")
            print(f"{'='*50}")
        
        # Entrenar modelo
        model = train_model(X_train, y_train, model_type, verbose)
        
        # Evaluar modelo
        metrics = evaluate_model(model, X_test, y_test, mlb, verbose)
        
        # Guardar modelo
        save_model(model, vectorizer, mlb, model_name, metrics, verbose)
        
        # Almacenar resultados
        results[model_type] = metrics
    
    return results

def train_single_model(model_type='svm', data_path=RAW_DATA_PATH, max_features=10000, 
                      test_size=0.2, random_state=42, model_name=None, verbose=True):
    """Entrena un único modelo del tipo especificado y lo guarda."""
    # Determinar nombre del archivo si no se proporciona
    if model_name is None:
        model_name = f"{model_type}_model.pkl"
    
    # Cargar y preprocesar datos
    df = load_and_preprocess_data(data_path, verbose)
    
    # Extraer características y etiquetas
    X, y, vectorizer, mlb = extract_features_and_labels(df, max_features, verbose)
    
    # Dividir conjunto de datos
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size, random_state, verbose)
    
    # Entrenar modelo
    model = train_model(X_train, y_train, model_type, verbose)
    
    # Evaluar modelo
    metrics = evaluate_model(model, X_test, y_test, mlb, verbose)
    
    # Guardar modelo
    save_model(model, vectorizer, mlb, model_name, metrics, verbose)
    
    return model, vectorizer, mlb, metrics

def train_demo_model(data_path=RAW_DATA_PATH, verbose=True):
    """Entrena un modelo básico de demostración (SVM) con configuración simplificada."""
    return train_single_model(
        model_type='svm',
        data_path=data_path,
        max_features=5000,  # Menos características para un modelo más ligero
        model_name='demo_model_simple.pkl',
        verbose=verbose
    )

def main():
    """Función principal que procesa argumentos de línea de comandos y ejecuta el entrenamiento."""
    parser = argparse.ArgumentParser(description='Entrenamiento de modelos de clasificación de literatura médica')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['all', 'svm', 'rf', 'lr', 'demo'],
                        help='Tipo de modelo a entrenar (all, svm, rf, lr, demo)')
    parser.add_argument('--data', type=str, default=RAW_DATA_PATH,
                        help='Ruta al archivo de datos CSV')
    parser.add_argument('--features', type=int, default=10000,
                        help='Número máximo de características para TF-IDF')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proporción del conjunto de prueba')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para reproducibilidad')
    parser.add_argument('--quiet', action='store_true',
                        help='Modo silencioso (sin mensajes de progreso)')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.model == 'all':
        train_all_models(
            data_path=args.data,
            max_features=args.features,
            test_size=args.test_size,
            random_state=args.seed,
            verbose=verbose
        )
    elif args.model == 'demo':
        train_demo_model(
            data_path=args.data,
            verbose=verbose
        )
    else:
        train_single_model(
            model_type=args.model,
            data_path=args.data,
            max_features=args.features,
            test_size=args.test_size,
            random_state=args.seed,
            verbose=verbose
        )

if __name__ == '__main__':
    main()