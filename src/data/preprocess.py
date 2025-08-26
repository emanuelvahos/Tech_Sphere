# -*- coding: utf-8 -*-
"""
Módulo para el preprocesamiento de datos de literatura médica.

Este módulo contiene funciones para limpiar, tokenizar y transformar
textos médicos para su uso en modelos de clasificación.
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional

# Importaciones para procesamiento de texto
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Asegurar que los recursos de NLTK estén disponibles
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.
    
    Args:
        file_path: Ruta al archivo CSV.
        
    Returns:
        DataFrame con los datos cargados.
    """
    return pd.read_csv(file_path, sep=';')


def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando caracteres especiales y normalizando.
    
    Args:
        text: Texto a limpiar.
        
    Returns:
        Texto limpio.
    """
    if pd.isna(text):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Normalizar caracteres unicode
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Eliminar caracteres especiales pero mantener puntuación importante
    text = re.sub(r'[^\w\s.,;:?!-]', '', text)
    
    # Reemplazar múltiples espacios con uno solo
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def remove_stopwords(tokens: List[str], lang: str = 'english', custom_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Elimina stopwords de una lista de tokens.
    
    Args:
        tokens: Lista de tokens.
        lang: Idioma para las stopwords.
        custom_stopwords: Lista adicional de stopwords personalizadas.
        
    Returns:
        Lista de tokens sin stopwords.
    """
    stop_words = set(stopwords.words(lang))
    
    # Añadir stopwords personalizadas si se proporcionan
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    return [word for word in tokens if word not in stop_words]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Aplica stemming a una lista de tokens.
    
    Args:
        tokens: Lista de tokens.
        
    Returns:
        Lista de tokens con stemming aplicado.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Aplica lematización a una lista de tokens.
    
    Args:
        tokens: Lista de tokens.
        
    Returns:
        Lista de tokens con lematización aplicada.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]


def lemmatize_text(text: str) -> str:
    """
    Aplica lematización a un texto completo.
    
    Args:
        text: Texto a lematizar.
        
    Returns:
        Texto con lematización aplicada.
    """
    # Limpiar el texto
    clean = clean_text(text)
    
    # Tokenizar
    tokens = word_tokenize(clean)
    
    # Lematizar tokens
    lemmatized_tokens = lemmatize_tokens(tokens)
    
    # Unir tokens en un texto
    return ' '.join(lemmatized_tokens)


def tokenize_text(text: str, remove_stop: bool = True, stem: bool = False, 
                 lemmatize: bool = True, lang: str = 'english',
                 custom_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Tokeniza un texto con opciones para eliminar stopwords, aplicar stemming o lematización.
    
    Args:
        text: Texto a tokenizar.
        remove_stop: Si es True, elimina stopwords.
        stem: Si es True, aplica stemming.
        lemmatize: Si es True, aplica lematización.
        lang: Idioma para las stopwords.
        custom_stopwords: Lista adicional de stopwords personalizadas.
        
    Returns:
        Lista de tokens procesados.
    """
    # Limpiar el texto
    clean = clean_text(text)
    
    # Tokenizar
    tokens = word_tokenize(clean)
    
    # Eliminar stopwords si se solicita
    if remove_stop:
        tokens = remove_stopwords(tokens, lang, custom_stopwords)
    
    # Aplicar stemming si se solicita
    if stem:
        tokens = stem_tokens(tokens)
    
    # Aplicar lematización si se solicita
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    
    return tokens


def preprocess_dataframe(df: pd.DataFrame, text_columns: List[str], 
                        remove_stop: bool = True, stem: bool = False,
                        lemmatize: bool = True, lang: str = 'english',
                        custom_stopwords: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Preprocesa las columnas de texto de un DataFrame.
    
    Args:
        df: DataFrame a procesar.
        text_columns: Lista de nombres de columnas de texto a procesar.
        remove_stop: Si es True, elimina stopwords.
        stem: Si es True, aplica stemming.
        lemmatize: Si es True, aplica lematización.
        lang: Idioma para las stopwords.
        custom_stopwords: Lista adicional de stopwords personalizadas.
        
    Returns:
        DataFrame con columnas adicionales para los textos procesados.
    """
    df_processed = df.copy()
    
    # Procesar cada columna de texto
    for col in text_columns:
        # Columna para texto limpio
        df_processed[f"{col}_clean"] = df_processed[col].apply(clean_text)
        
        # Columna para tokens
        df_processed[f"{col}_tokens"] = df_processed[col].apply(
            lambda x: tokenize_text(x, remove_stop, stem, lemmatize, lang, custom_stopwords)
        )
        
        # Columna para texto procesado unido
        df_processed[f"{col}_processed"] = df_processed[f"{col}_tokens"].apply(lambda x: ' '.join(x))
    
    return df_processed


def prepare_multilabel_targets(df: pd.DataFrame, target_column: str, 
                              label_separator: str = '|') -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepara las etiquetas para clasificación multi-etiqueta.
    
    Args:
        df: DataFrame con los datos.
        target_column: Nombre de la columna que contiene las etiquetas.
        label_separator: Separador utilizado en las etiquetas múltiples.
        
    Returns:
        Tupla con (DataFrame con columnas binarias para cada etiqueta, lista de nombres de etiquetas).
    """
    # Extraer todas las etiquetas únicas
    all_labels = set()
    for labels in df[target_column].dropna():
        all_labels.update(labels.split(label_separator))
    
    # Ordenar las etiquetas para consistencia
    all_labels = sorted(list(all_labels))
    
    # Crear columnas binarias para cada etiqueta
    for label in all_labels:
        df[label] = df[target_column].apply(
            lambda x: 1 if pd.notna(x) and label in x.split(label_separator) else 0
        )
    
    return df, all_labels


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Guarda los datos procesados en un archivo CSV.
    
    Args:
        df: DataFrame con los datos procesados.
        output_path: Ruta donde guardar el archivo CSV.
    """
    df.to_csv(output_path, index=False)


def main():
    """
    Función principal para ejecutar el preprocesamiento desde la línea de comandos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocesar datos de literatura médica')
    parser.add_argument('--input', type=str, required=True, help='Ruta al archivo CSV de entrada')
    parser.add_argument('--output', type=str, required=True, help='Ruta para guardar el archivo CSV procesado')
    parser.add_argument('--no-stopwords', action='store_true', help='No eliminar stopwords')
    parser.add_argument('--stem', action='store_true', help='Aplicar stemming')
    parser.add_argument('--no-lemmatize', action='store_true', help='No aplicar lematización')
    
    args = parser.parse_args()
    
    # Cargar datos
    print(f"Cargando datos desde {args.input}...")
    df = load_data(args.input)
    
    # Preprocesar
    print("Preprocesando datos...")
    df_processed = preprocess_dataframe(
        df, 
        text_columns=['title', 'abstract'],
        remove_stop=not args.no_stopwords,
        stem=args.stem,
        lemmatize=not args.no_lemmatize
    )
    
    # Preparar etiquetas multi-etiqueta
    print("Preparando etiquetas multi-etiqueta...")
    df_processed, labels = prepare_multilabel_targets(df_processed, 'group')
    print(f"Etiquetas encontradas: {labels}")
    
    # Guardar resultados
    print(f"Guardando datos procesados en {args.output}...")
    save_processed_data(df_processed, args.output)
    print("¡Preprocesamiento completado!")


if __name__ == "__main__":
    main()