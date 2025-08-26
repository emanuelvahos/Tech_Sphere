#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilidades comunes para scripts de demostración.

Este módulo contiene funciones compartidas entre los scripts de demostración
básicos y avanzados para evitar duplicación de código.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss


def load_model(model_path):
    """Carga un modelo entrenado desde un archivo pickle o joblib.
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        tuple: (model, vectorizer, mlb) o (None, None, None) si hay error
    """
    try:
        if model_path.endswith('.pkl'):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            if isinstance(model_data, tuple):
                # Formato (model, vectorizer, mlb)
                return model_data
            elif isinstance(model_data, dict):
                # Formato {'model': model, 'vectorizer': vectorizer, 'mlb': mlb}
                return model_data['model'], model_data['vectorizer'], model_data['mlb']
        else:
            # Intentar cargar con joblib
            return joblib.load(model_path)
            
        print(f"Modelo cargado correctamente desde {model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None, None


def save_model(model, vectorizer, mlb, model_path, use_joblib=False):
    """Guarda un modelo entrenado en un archivo pickle o joblib.
    
    Args:
        model: Modelo entrenado
        vectorizer: Vectorizador TF-IDF
        mlb: MultiLabelBinarizer
        model_path: Ruta donde guardar el modelo
        use_joblib: Si es True, usa joblib en lugar de pickle
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'mlb': mlb
    }
    
    try:
        if use_joblib:
            joblib.dump(model_data, model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
        
        print(f"Modelo guardado en {model_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")


def load_data(data_path):
    """Carga los datos desde un archivo CSV.
    
    Args:
        data_path: Ruta al archivo CSV
        
    Returns:
        DataFrame o None si hay error
    """
    if not os.path.exists(data_path):
        print(f"No se encontró el archivo de datos en {data_path}")
        return None
    
    print(f"Cargando datos desde {data_path}")
    try:
        # Intentar con diferentes separadores
        try:
            df = pd.read_csv(data_path, sep=';')
        except:
            df = pd.read_csv(data_path)
        
        # Preprocesamiento básico
        if 'title' in df.columns and 'abstract' in df.columns:
            df['text_processed'] = df['title'] + " " + df['abstract']
        
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None


def prepare_train_test_data(df, test_size=0.2, random_state=42):
    """Prepara los datos de entrenamiento y prueba.
    
    Args:
        df: DataFrame con los datos
        test_size: Proporción del conjunto de prueba
        random_state: Semilla aleatoria
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Verificar que existan las columnas necesarias
    if 'text_processed' not in df.columns or 'group' not in df.columns:
        print("El DataFrame no tiene las columnas necesarias (text_processed, group)")
        return None, None, None, None
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_processed'], 
        df['group'].str.split('|'),
        test_size=test_size, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def vectorize_and_transform(X_train, X_test, y_train, y_test, max_features=5000):
    """Vectoriza los textos y transforma las etiquetas.
    
    Args:
        X_train: Textos de entrenamiento
        X_test: Textos de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        max_features: Número máximo de características para TF-IDF
        
    Returns:
        tuple: (vectorizer, mlb, X_train_vec, X_test_vec, y_train_bin, y_test_bin)
    """
    # Vectorizar texto
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Preparar etiquetas
    mlb = MultiLabelBinarizer()
    y_train_bin = mlb.fit_transform(y_train)
    y_test_bin = mlb.transform(y_test)
    
    return vectorizer, mlb, X_train_vec, X_test_vec, y_train_bin, y_test_bin


def train_simple_model(X_train_vec, y_train_bin, model_type='logistic'):
    """Entrena un modelo simple de clasificación.
    
    Args:
        X_train_vec: Características de entrenamiento vectorizadas
        y_train_bin: Etiquetas de entrenamiento binarizadas
        model_type: Tipo de modelo ('logistic', 'svm', 'rf')
        
    Returns:
        Modelo entrenado
    """
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    
    if model_type == 'logistic':
        base_model = LogisticRegression(max_iter=1000)
    elif model_type == 'svm':
        base_model = LinearSVC()
    elif model_type == 'rf':
        base_model = RandomForestClassifier(n_estimators=100)
    else:
        base_model = LogisticRegression(max_iter=1000)
    
    # Entrenar modelo
    model = OneVsRestClassifier(base_model)
    model.fit(X_train_vec, y_train_bin)
    
    return model


def evaluate_model_basic(model, X_test, y_test):
    """Evalúa el modelo y calcula métricas básicas.
    
    Args:
        model: Modelo entrenado
        X_test: Características de prueba
        y_test: Etiquetas de prueba
        
    Returns:
        tuple: (y_pred, metrics_dict)
    """
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    hamming = hamming_loss(y_test, y_pred)
    
    # Crear diccionario de métricas
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'hamming_loss': hamming
    }
    
    return y_pred, metrics


def display_metrics(metrics):
    """Muestra las métricas de evaluación.
    
    Args:
        metrics: Diccionario con las métricas
    """
    print("\nMétricas de evaluación:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")


def create_figures_dir():
    """Crea el directorio para guardar figuras si no existe.
    
    Returns:
        str: Ruta al directorio de figuras
    """
    figures_dir = os.path.join('reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def plot_simple_confusion_matrix(y_test, y_pred, save_path=None):
    """Genera una matriz de confusión simple.
    
    Args:
        y_test: Etiquetas reales
        y_pred: Etiquetas predichas
        save_path: Ruta donde guardar la figura (opcional)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Crear una matriz de confusión simple para datos binarios aplanados
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test.flatten(), y_pred.flatten())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión Global")
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_simple_metrics_radar(metrics, save_path=None):
    """Genera un gráfico de radar para las métricas.
    
    Args:
        metrics: Diccionario con las métricas
        save_path: Ruta donde guardar la figura (opcional)
    """
    # Preparar datos para el gráfico
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', '1 - Hamming Loss']
    values = [
        metrics['accuracy'], 
        metrics['precision'], 
        metrics['recall'], 
        metrics['f1_score'], 
        1 - metrics['hamming_loss']
    ]
    
    # Crear gráfico de radar simple
    plt.figure(figsize=(10, 8))
    
    # Número de variables
    N = len(metrics_names)
    
    # Ángulos para cada eje
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el polígono
    
    # Valores para el gráfico
    values += values[:1]  # Cerrar el polígono
    
    # Crear el gráfico polar
    ax = plt.subplot(111, polar=True)
    
    # Dibujar el polígono y rellenarlo
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Añadir etiquetas
    plt.xticks(angles[:-1], metrics_names)
    
    # Añadir título
    plt.title("Métricas del Modelo", size=15)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()