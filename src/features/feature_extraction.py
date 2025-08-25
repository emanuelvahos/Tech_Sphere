# -*- coding: utf-8 -*-
"""
Módulo para la extracción de características de textos médicos.

Este módulo contiene funciones para extraer características relevantes
de textos médicos para su uso en modelos de clasificación.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Any

# Importaciones para procesamiento de texto y extracción de características
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler

# Para word embeddings
import gensim
from gensim.models import Word2Vec, FastText


def create_bow_features(texts: List[str], max_features: int = 5000, 
                       ngram_range: Tuple[int, int] = (1, 2),
                       binary: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Crea características de Bag of Words.
    
    Args:
        texts: Lista de textos procesados.
        max_features: Número máximo de características.
        ngram_range: Rango de n-gramas a considerar.
        binary: Si es True, usa representación binaria en lugar de conteos.
        
    Returns:
        Tupla con (matriz de características, lista de nombres de características).
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        binary=binary
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return X, feature_names


def create_tfidf_features(texts: List[str], max_features: int = 5000,
                         ngram_range: Tuple[int, int] = (1, 2),
                         use_idf: bool = True,
                         sublinear_tf: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Crea características TF-IDF.
    
    Args:
        texts: Lista de textos procesados.
        max_features: Número máximo de características.
        ngram_range: Rango de n-gramas a considerar.
        use_idf: Si es True, usa ponderación IDF.
        sublinear_tf: Si es True, usa escala logarítmica para TF.
        
    Returns:
        Tupla con (matriz de características, lista de nombres de características).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        use_idf=use_idf,
        sublinear_tf=sublinear_tf
    )
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return X, feature_names


def apply_dimensionality_reduction(X: np.ndarray, n_components: int = 100,
                                 method: str = 'svd') -> np.ndarray:
    """
    Aplica reducción de dimensionalidad a las características.
    
    Args:
        X: Matriz de características.
        n_components: Número de componentes a mantener.
        method: Método de reducción ('svd' o 'lda').
        
    Returns:
        Matriz de características reducida.
    """
    if method.lower() == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    elif method.lower() == 'lda':
        reducer = LatentDirichletAllocation(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Método de reducción no soportado: {method}. Use 'svd' o 'lda'.")
    
    X_reduced = reducer.fit_transform(X)
    return X_reduced


def train_word_embeddings(tokenized_texts: List[List[str]], method: str = 'word2vec',
                         vector_size: int = 100, window: int = 5,
                         min_count: int = 1, epochs: int = 10) -> Any:
    """
    Entrena un modelo de word embeddings.
    
    Args:
        tokenized_texts: Lista de textos tokenizados.
        method: Método de embeddings ('word2vec' o 'fasttext').
        vector_size: Dimensión de los vectores de embeddings.
        window: Tamaño de la ventana de contexto.
        min_count: Frecuencia mínima de palabras a considerar.
        epochs: Número de épocas de entrenamiento.
        
    Returns:
        Modelo de word embeddings entrenado.
    """
    if method.lower() == 'word2vec':
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs
        )
    elif method.lower() == 'fasttext':
        model = FastText(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs
        )
    else:
        raise ValueError(f"Método de embeddings no soportado: {method}. Use 'word2vec' o 'fasttext'.")
    
    return model


def create_document_embeddings(tokenized_texts: List[List[str]], 
                             embedding_model: Any) -> np.ndarray:
    """
    Crea embeddings a nivel de documento promediando los embeddings de palabras.
    
    Args:
        tokenized_texts: Lista de textos tokenizados.
        embedding_model: Modelo de word embeddings entrenado.
        
    Returns:
        Matriz de embeddings de documentos.
    """
    vector_size = embedding_model.wv.vector_size
    doc_embeddings = np.zeros((len(tokenized_texts), vector_size))
    
    for i, tokens in enumerate(tokenized_texts):
        valid_tokens = [token for token in tokens if token in embedding_model.wv]
        if valid_tokens:
            doc_embeddings[i] = np.mean([embedding_model.wv[token] for token in valid_tokens], axis=0)
    
    return doc_embeddings


def extract_text_statistics(texts: List[str]) -> pd.DataFrame:
    """
    Extrae estadísticas básicas de los textos.
    
    Args:
        texts: Lista de textos.
        
    Returns:
        DataFrame con estadísticas de los textos.
    """
    stats = pd.DataFrame()
    
    # Longitud del texto
    stats['text_length'] = [len(text) for text in texts]
    
    # Número de palabras
    stats['word_count'] = [len(text.split()) for text in texts]
    
    # Longitud promedio de palabras
    stats['avg_word_length'] = [np.mean([len(word) for word in text.split()]) if text else 0 for text in texts]
    
    # Número de oraciones (aproximado)
    stats['sentence_count'] = [text.count('.') + text.count('!') + text.count('?') for text in texts]
    
    return stats


def extract_medical_features(texts: List[str], medical_terms: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Extrae características específicas del dominio médico.
    
    Args:
        texts: Lista de textos.
        medical_terms: Diccionario con términos médicos por categoría.
        
    Returns:
        DataFrame con características médicas.
    """
    features = pd.DataFrame()
    
    # Para cada categoría de términos médicos
    for category, terms in medical_terms.items():
        # Contar ocurrencias de términos de esta categoría en cada texto
        features[f'{category}_term_count'] = [
            sum(1 for term in terms if term.lower() in text.lower()) for text in texts
        ]
        
        # Calcular densidad de términos (ocurrencias / longitud del texto)
        features[f'{category}_term_density'] = [
            count / (len(text) + 1) for count, text in zip(features[f'{category}_term_count'], texts)
        ]
    
    return features


def combine_features(feature_sets: List[np.ndarray], scale: bool = True) -> np.ndarray:
    """
    Combina múltiples conjuntos de características.
    
    Args:
        feature_sets: Lista de matrices de características.
        scale: Si es True, estandariza las características.
        
    Returns:
        Matriz combinada de características.
    """
    # Asegurar que todas las matrices sean densas
    dense_features = []
    for features in feature_sets:
        if hasattr(features, 'toarray'):
            dense_features.append(features.toarray())
        else:
            dense_features.append(features)
    
    # Combinar horizontalmente
    X_combined = np.hstack(dense_features)
    
    # Estandarizar si se solicita
    if scale:
        scaler = StandardScaler()
        X_combined = scaler.fit_transform(X_combined)
    
    return X_combined


def main():
    """
    Función principal para ejecutar la extracción de características desde la línea de comandos.
    """
    import argparse
    import os
    import pickle
    
    parser = argparse.ArgumentParser(description='Extraer características de textos médicos')
    parser.add_argument('--input', type=str, required=True, help='Ruta al archivo CSV procesado')
    parser.add_argument('--output-dir', type=str, required=True, help='Directorio para guardar las características')
    parser.add_argument('--method', type=str, default='tfidf', choices=['bow', 'tfidf', 'embeddings', 'combined'],
                       help='Método de extracción de características')
    parser.add_argument('--max-features', type=int, default=5000, help='Número máximo de características para BoW/TF-IDF')
    parser.add_argument('--embedding-size', type=int, default=100, help='Dimensión de los embeddings')
    
    args = parser.parse_args()
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar datos procesados
    print(f"Cargando datos desde {args.input}...")
    df = pd.read_csv(args.input)
    
    # Extraer características según el método seleccionado
    print(f"Extrayendo características usando método: {args.method}...")
    
    if args.method == 'bow':
        # Bag of Words para título y abstract
        X_title, title_features = create_bow_features(
            df['title_processed'].fillna(''),
            max_features=args.max_features
        )
        
        X_abstract, abstract_features = create_bow_features(
            df['abstract_processed'].fillna(''),
            max_features=args.max_features
        )
        
        # Combinar características
        X = combine_features([X_title, X_abstract])
        
        # Guardar vectorizadores y características
        with open(os.path.join(args.output_dir, 'bow_features.pkl'), 'wb') as f:
            pickle.dump({
                'title_features': title_features,
                'abstract_features': abstract_features
            }, f)
        
    elif args.method == 'tfidf':
        # TF-IDF para título y abstract
        X_title, title_features = create_tfidf_features(
            df['title_processed'].fillna(''),
            max_features=args.max_features
        )
        
        X_abstract, abstract_features = create_tfidf_features(
            df['abstract_processed'].fillna(''),
            max_features=args.max_features
        )
        
        # Combinar características
        X = combine_features([X_title, X_abstract])
        
        # Guardar vectorizadores y características
        with open(os.path.join(args.output_dir, 'tfidf_features.pkl'), 'wb') as f:
            pickle.dump({
                'title_features': title_features,
                'abstract_features': abstract_features
            }, f)
        
    elif args.method == 'embeddings':
        # Convertir tokens a listas
        title_tokens = df['title_tokens'].apply(eval if isinstance(df['title_tokens'].iloc[0], str) else lambda x: x)
        abstract_tokens = df['abstract_tokens'].apply(eval if isinstance(df['abstract_tokens'].iloc[0], str) else lambda x: x)
        
        # Entrenar modelos de embeddings
        title_model = train_word_embeddings(
            title_tokens.tolist(),
            vector_size=args.embedding_size
        )
        
        abstract_model = train_word_embeddings(
            abstract_tokens.tolist(),
            vector_size=args.embedding_size
        )
        
        # Crear embeddings de documentos
        X_title = create_document_embeddings(title_tokens.tolist(), title_model)
        X_abstract = create_document_embeddings(abstract_tokens.tolist(), abstract_model)
        
        # Combinar características
        X = combine_features([X_title, X_abstract])
        
        # Guardar modelos
        title_model.save(os.path.join(args.output_dir, 'title_embeddings.model'))
        abstract_model.save(os.path.join(args.output_dir, 'abstract_embeddings.model'))
        
    elif args.method == 'combined':
        # TF-IDF
        X_title_tfidf, _ = create_tfidf_features(
            df['title_processed'].fillna(''),
            max_features=args.max_features // 2
        )
        
        X_abstract_tfidf, _ = create_tfidf_features(
            df['abstract_processed'].fillna(''),
            max_features=args.max_features
        )
        
        # Estadísticas de texto
        title_stats = extract_text_statistics(df['title_clean'].fillna(''))
        abstract_stats = extract_text_statistics(df['abstract_clean'].fillna(''))
        
        # Combinar todas las características
        X = combine_features([
            X_title_tfidf, 
            X_abstract_tfidf,
            title_stats.values,
            abstract_stats.values
        ])
    
    # Guardar matriz de características
    print(f"Guardando matriz de características ({X.shape[0]} muestras, {X.shape[1]} características)...")
    np.save(os.path.join(args.output_dir, 'X_features.npy'), X)
    
    # Guardar etiquetas
    y_columns = ['cardiovascular', 'hepatorenal', 'neurological', 'oncological']
    y = df[y_columns].values
    np.save(os.path.join(args.output_dir, 'y_labels.npy'), y)
    
    print("¡Extracción de características completada!")


if __name__ == "__main__":
    main()