#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo básica de visualización y evaluación de modelos.

Este script demuestra las capacidades básicas de visualización y evaluación
implementadas en el proyecto Tech_Sphere para el análisis de modelos de clasificación
de textos médicos, sin depender de bibliotecas externas complejas.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

# Añadir el directorio raíz al path para importar módulos propios
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Importar funciones de visualización básicas
from src.visualization.visualize import plot_confusion_matrix, plot_metrics_radar


def cargar_o_crear_modelo():
    """Carga un modelo existente o crea uno nuevo si no existe."""
    print("Cargando o creando modelo...")
    
    # Ruta al modelo
    model_path = os.path.join('models', 'demo_model_simple.pkl')
    
    # Verificar si el modelo existe
    if os.path.exists(model_path):
        print(f"Cargando modelo existente desde {model_path}")
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            model = model_data['model']
            vectorizer = model_data['vectorizer']
            mlb = model_data['mlb']
            return model, vectorizer, mlb, True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return None, None, None, False
    
    print("No se encontró un modelo existente. Creando uno nuevo...")
    return None, None, None, False


def cargar_datos():
    """Carga los datos de entrenamiento."""
    # Ruta a los datos
    data_path = os.path.join('data', 'raw', 'challenge_data-18-ago.csv')
    
    if not os.path.exists(data_path):
        print(f"No se encontró el archivo de datos en {data_path}")
        return None
    
    print(f"Cargando datos desde {data_path}")
    df = pd.read_csv(data_path, sep=';')
    
    # Preprocesamiento básico
    df['text_processed'] = df['title'] + " " + df['abstract']
    
    return df


def entrenar_modelo(df):
    """Entrena un modelo simple de clasificación."""
    print("Entrenando modelo...")
    
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
    
    # Entrenar modelo
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train_vec, y_train_bin)
    
    # Guardar modelo
    model_path = os.path.join('models', 'demo_model_simple.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'mlb': mlb
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Modelo guardado en {model_path}")
    
    return model, vectorizer, mlb, X_test_vec, y_test_bin


def evaluar_modelo(model, X_test, y_test, mlb):
    """Evalúa el modelo y muestra métricas básicas."""
    print("\nEvaluando modelo...")
    
    # Predecir
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    hamming = hamming_loss(y_test, y_pred)
    
    # Mostrar métricas
    print("\nMétricas de evaluación:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    
    # Visualizar matriz de confusión
    print("\nGenerando visualizaciones...")
    
    # Crear directorio para figuras si no existe
    figures_dir = os.path.join('reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generar matriz de confusión simple (sin usar la función de visualize.py)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Crear una matriz de confusión simple para datos binarios aplanados
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test.flatten(), y_pred.flatten())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusión Global")
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.savefig(os.path.join(figures_dir, "confusion_matrix_global.png"))
    plt.close()
    
    # Generar gráfico de radar para métricas (implementación simplificada)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', '1 - Hamming Loss']
    values = [accuracy, precision, recall, f1, 1 - hamming]
    
    # Crear gráfico de radar simple
    plt.figure(figsize=(10, 8))
    
    # Número de variables
    N = len(metrics)
    
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
    plt.xticks(angles[:-1], metrics)
    
    # Añadir título
    plt.title("Métricas del Modelo", size=15)
    
    plt.savefig(os.path.join(figures_dir, "metrics_radar.png"))
    plt.close()
    
    print(f"Visualizaciones guardadas en {figures_dir}")
    
    return y_pred


def main():
    """Función principal."""
    print("=== DEMO BÁSICA DE EVALUACIÓN DE MODELOS ===\n")
    
    # Cargar o crear modelo
    model, vectorizer, mlb, modelo_cargado = cargar_o_crear_modelo()
    
    # Cargar datos
    df = cargar_datos()
    if df is None:
        print("No se pudieron cargar los datos. Finalizando demo.")
        return
    
    # Si no se cargó un modelo, entrenar uno nuevo
    if not modelo_cargado:
        model, vectorizer, mlb, X_test, y_test = entrenar_modelo(df)
    else:
        # Preparar datos de prueba
        _, X_test, _, y_test = train_test_split(
            df['text_processed'], 
            df['group'].str.split('|'),
            test_size=0.2, 
            random_state=42
        )
        X_test_vec = vectorizer.transform(X_test)
        y_test_bin = mlb.transform(y_test)
        X_test, y_test = X_test_vec, y_test_bin
    
    # Evaluar modelo
    evaluar_modelo(model, X_test, y_test, mlb)
    
    print("\n=== DEMO COMPLETADA ===\n")
    print("Revise las visualizaciones generadas en 'reports/figures/'")


if __name__ == "__main__":
    main()