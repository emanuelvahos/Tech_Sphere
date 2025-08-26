# -*- coding: utf-8 -*-
"""
Script para demostrar el proceso de entrenamiento de un modelo de clasificación de literatura médica.

Este script procesa los datos, extrae características y entrena un modelo SVM básico.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Añadir el directorio raíz al path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data.preprocess import clean_text, remove_stopwords, lemmatize_tokens, tokenize_text

# Configurar directorios
RAW_DATA_PATH = os.path.join('data', 'raw', 'challenge_data-18-ago.csv')
PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')

# Crear directorios si no existen
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Cargar datos
print("Cargando datos...")
df = pd.read_csv(RAW_DATA_PATH, sep=';')
print(f"Datos cargados: {df.shape[0]} registros")

# Preprocesar textos
print("Preprocesando textos...")

def preprocess_text(text):
    """Preprocesa un texto aplicando limpieza, eliminación de stopwords y lematización."""
    if pd.isna(text):
        return ""
    # Usar la función tokenize_text que ya incluye limpieza, eliminación de stopwords y lematización
    tokens = tokenize_text(text, remove_stop=True, lemmatize=True)
    return ' '.join(tokens)

# Aplicar preprocesamiento
df['title_processed'] = df['title'].apply(preprocess_text)
df['abstract_processed'] = df['abstract'].apply(preprocess_text)
df['text_combined'] = df['title_processed'] + " " + df['abstract_processed']

# Preparar etiquetas
print("Preparando etiquetas...")
df['group_list'] = df['group'].apply(lambda x: x.split('|') if pd.notna(x) else [])

# Binarizar etiquetas
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['group_list'])
print(f"Categorías: {mlb.classes_}")

# Extraer características TF-IDF
print("Extrayendo características TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text_combined'])
print(f"Matriz de características: {X.shape}")

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Entrenar modelo SVM
print("Entrenando modelo SVM...")
model = OneVsRestClassifier(LinearSVC(random_state=42))
model.fit(X_train, y_train)

# Evaluar modelo
print("Evaluando modelo...")
y_pred = model.predict(X_test)
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Guardar modelo, vectorizador y binarizador
print("Guardando modelo...")
model_data = {
    'model': model,
    'vectorizer': vectorizer,
    'mlb': mlb
}

model_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Modelo guardado en {model_path}")
print("\nAhora puedes usar este modelo con los scripts:")
print("- src/models/test_model.py: Para probar el modelo con textos individuales")
print("- src/models/evaluate_model.py: Para evaluar el modelo con un conjunto de datos")
print("- src/app.py: Para usar el modelo a través de una interfaz web")