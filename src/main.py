# -*- coding: utf-8 -*-
"""Script simplificado para análisis exploratorio de datos de literatura médica."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
import re

# Asegurar que los recursos de NLTK estén disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configuraciones
plt.style.use('ggplot')
sns.set(style='whitegrid')
pd.set_option('display.max_columns', None)


def create_directories():
    """Crea los directorios necesarios para el proyecto si no existen."""
    directories = [
        'data/processed',
        'reports/figures',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path, sep=';')


def analyze_data(df):
    """Realiza un análisis exploratorio básico de los datos."""
    print("\nInformación general del dataset:")
    print(f"Número de registros: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    print("\nEstadísticas descriptivas:")
    print(df.describe(include='all'))
    
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    
    print("\nDuplicados en el dataset:")
    print(f"Número de duplicados: {df.duplicated().sum()}")
    
    # Análisis de la columna 'group'
    print("\nDistribución de grupos:")
    groups = df['group'].str.split('|', expand=False).explode().value_counts()
    print(groups)
    
    # Análisis de longitud de textos
    df['title_length'] = df['title'].str.len()
    df['abstract_length'] = df['abstract'].str.len()
    
    print("\nEstadísticas de longitud de títulos:")
    print(df['title_length'].describe())
    
    print("\nEstadísticas de longitud de abstracts:")
    print(df['abstract_length'].describe())
    
    return df


def visualize_data(df):
    """Genera visualizaciones básicas de los datos."""
    # Crear directorio para figuras
    os.makedirs('reports/figures', exist_ok=True)
    
    # Distribución de grupos
    plt.figure(figsize=(10, 6))
    groups = df['group'].str.split('|', expand=False).explode().value_counts()
    sns.barplot(x=groups.index, y=groups.values)
    plt.title('Distribución de Categorías Médicas')
    plt.xlabel('Categoría')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/figures/group_distribution.png')
    plt.close()
    
    # Distribución de longitud de títulos
    plt.figure(figsize=(10, 6))
    sns.histplot(df['title_length'], kde=True)
    plt.title('Distribución de Longitud de Títulos')
    plt.xlabel('Longitud (caracteres)')
    plt.ylabel('Frecuencia')
    plt.savefig('reports/figures/title_length_distribution.png')
    plt.close()
    
    # Distribución de longitud de abstracts
    plt.figure(figsize=(10, 6))
    sns.histplot(df['abstract_length'], kde=True)
    plt.title('Distribución de Longitud de Abstracts')
    plt.xlabel('Longitud (caracteres)')
    plt.ylabel('Frecuencia')
    plt.savefig('reports/figures/abstract_length_distribution.png')
    plt.close()
    
    # Longitud promedio por categoría
    plt.figure(figsize=(12, 6))
    category_length = df.copy()
    category_length['category'] = df['group']
    avg_lengths = category_length.groupby('category')[['title_length', 'abstract_length']].mean().reset_index()
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='category', y='title_length', data=avg_lengths)
    plt.title('Longitud Promedio de Títulos por Categoría')
    plt.xlabel('Categoría')
    plt.ylabel('Longitud Promedio (caracteres)')
    plt.xticks(rotation=90)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='category', y='abstract_length', data=avg_lengths)
    plt.title('Longitud Promedio de Abstracts por Categoría')
    plt.xlabel('Categoría')
    plt.ylabel('Longitud Promedio (caracteres)')
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig('reports/figures/length_by_category.png')
    plt.close()


def main():
    """Función principal que ejecuta el análisis exploratorio."""
    print("Iniciando análisis exploratorio de datos...")
    
    # Crear directorios necesarios
    create_directories()
    
    # Cargar datos
    data_path = 'data/raw/challenge_data-18-ago.csv'
    if not os.path.exists(data_path):
        data_path = 'challenge_data-18-ago.csv'
    
    df = load_data(data_path)
    
    # Analizar datos
    df = analyze_data(df)
    
    # Visualizar datos
    visualize_data(df)
    
    print("\nAnálisis exploratorio completado. Las visualizaciones se han guardado en 'reports/figures/'")


if __name__ == "__main__":
    main()