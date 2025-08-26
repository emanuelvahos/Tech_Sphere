# -*- coding: utf-8 -*-
"""
Script para evaluar un modelo específico con datos de prueba.

Este script permite cargar un modelo entrenado y evaluarlo con datos de prueba
para obtener métricas detalladas de rendimiento.
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss, jaccard_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Añadir el directorio raíz al path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocess import clean_text, remove_stopwords, lemmatize_text
from src.features.feature_extraction import extract_tfidf_features
from src.visualization.visualize import plot_confusion_matrix, plot_metrics_radar


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


def load_test_data(test_data_path):
    """Carga los datos de prueba desde un archivo CSV.
    
    Args:
        test_data_path (str): Ruta al archivo CSV con datos de prueba.
        
    Returns:
        pandas.DataFrame: DataFrame con los datos de prueba.
    """
    try:
        df = pd.read_csv(test_data_path)
        print(f"Datos de prueba cargados correctamente desde {test_data_path}")
        return df
    except Exception as e:
        print(f"Error al cargar los datos de prueba: {e}")
        return None


def preprocess_dataset(df):
    """Preprocesa el conjunto de datos aplicando limpieza, eliminación de stopwords y lematización.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos a preprocesar.
        
    Returns:
        pandas.DataFrame: DataFrame con los datos preprocesados.
    """
    # Crear copias para no modificar los originales
    df_processed = df.copy()
    
    # Preprocesar títulos
    print("Preprocesando títulos...")
    df_processed['title_processed'] = df_processed['title'].apply(clean_text)
    df_processed['title_processed'] = df_processed['title_processed'].apply(remove_stopwords)
    df_processed['title_processed'] = df_processed['title_processed'].apply(lemmatize_text)
    
    # Preprocesar resúmenes
    print("Preprocesando resúmenes...")
    df_processed['abstract_processed'] = df_processed['abstract'].apply(clean_text)
    df_processed['abstract_processed'] = df_processed['abstract_processed'].apply(remove_stopwords)
    df_processed['abstract_processed'] = df_processed['abstract_processed'].apply(lemmatize_text)
    
    # Combinar título y resumen procesados
    df_processed['text_processed'] = df_processed['title_processed'] + " " + df_processed['abstract_processed']
    
    return df_processed


def analyze_threshold_impact(y_test, y_pred_proba, mlb_classes, thresholds=None):
    """Analiza el impacto de diferentes umbrales de decisión en las métricas de rendimiento.
    
    Args:
        y_test: Matriz de etiquetas reales.
        y_pred_proba: Matriz de probabilidades predichas.
        mlb_classes: Nombres de las etiquetas.
        thresholds: Lista de umbrales a evaluar. Si es None, se utilizan valores de 0.1 a 0.9.
        
    Returns:
        Diccionario con métricas por umbral y por etiqueta.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = {}
    
    # Para cada etiqueta
    for i, label in enumerate(mlb_classes):
        label_results = {
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
            'threshold': thresholds
        }
        
        # Para cada umbral
        for threshold in thresholds:
            # Aplicar umbral
            y_pred_binary = (y_pred_proba[:, i] >= threshold).astype(int)
            
            # Calcular métricas
            precision = precision_score(y_test[:, i], y_pred_binary, zero_division=0)
            recall = recall_score(y_test[:, i], y_pred_binary, zero_division=0)
            f1 = f1_score(y_test[:, i], y_pred_binary, zero_division=0)
            accuracy = accuracy_score(y_test[:, i], y_pred_binary)
            
            # Guardar resultados
            label_results['precision'].append(precision)
            label_results['recall'].append(recall)
            label_results['f1'].append(f1)
            label_results['accuracy'].append(accuracy)
        
        results[label] = label_results
    
    # Calcular métricas promedio
    avg_results = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': [],
        'threshold': thresholds
    }
    
    for threshold_idx, _ in enumerate(thresholds):
        avg_precision = np.mean([results[label]['precision'][threshold_idx] for label in mlb_classes])
        avg_recall = np.mean([results[label]['recall'][threshold_idx] for label in mlb_classes])
        avg_f1 = np.mean([results[label]['f1'][threshold_idx] for label in mlb_classes])
        avg_accuracy = np.mean([results[label]['accuracy'][threshold_idx] for label in mlb_classes])
        
        avg_results['precision'].append(avg_precision)
        avg_results['recall'].append(avg_recall)
        avg_results['f1'].append(avg_f1)
        avg_results['accuracy'].append(avg_accuracy)
    
    results['average'] = avg_results
    
    return results


def find_optimal_thresholds(y_test, y_pred_proba, mlb_classes, metric='f1', thresholds=None):
    """Encuentra los umbrales óptimos para cada etiqueta según una métrica específica.
    
    Args:
        y_test: Matriz de etiquetas reales.
        y_pred_proba: Matriz de probabilidades predichas.
        mlb_classes: Nombres de las etiquetas.
        metric: Métrica a optimizar ('precision', 'recall', 'f1', 'accuracy').
        thresholds: Lista de umbrales a evaluar. Si es None, se utilizan valores de 0.01 a 0.99.
        
    Returns:
        Diccionario con umbrales óptimos por etiqueta.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    optimal_thresholds = {}
    
    # Para cada etiqueta
    for i, label in enumerate(mlb_classes):
        best_score = 0
        best_threshold = 0.5  # Umbral por defecto
        
        # Para cada umbral
        for threshold in thresholds:
            # Aplicar umbral
            y_pred_binary = (y_pred_proba[:, i] >= threshold).astype(int)
            
            # Calcular métrica
            if metric == 'precision':
                score = precision_score(y_test[:, i], y_pred_binary, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_test[:, i], y_pred_binary, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_test[:, i], y_pred_binary)
            else:  # f1 por defecto
                score = f1_score(y_test[:, i], y_pred_binary, zero_division=0)
            
            # Actualizar mejor umbral
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds[label] = best_threshold
    
    return optimal_thresholds


def analyze_probability_distribution(y_test, y_pred_proba, mlb_classes):
    """Analiza la distribución de probabilidades predichas para cada clase.
    
    Args:
        y_test: Matriz de etiquetas reales.
        y_pred_proba: Matriz de probabilidades predichas.
        mlb_classes: Nombres de las etiquetas.
        
    Returns:
        Diccionario con distribuciones de probabilidad por etiqueta y clase real.
    """
    distributions = {}
    
    # Para cada etiqueta
    for i, label in enumerate(mlb_classes):
        # Separar probabilidades por clase real
        positive_probs = y_pred_proba[y_test[:, i] == 1, i]
        negative_probs = y_pred_proba[y_test[:, i] == 0, i]
        
        distributions[label] = {
            'positive_samples': positive_probs,
            'negative_samples': negative_probs
        }
    
    return distributions


def calculate_calibration_metrics(y_test, y_pred_proba, mlb_classes, n_bins=10):
    """Calcula métricas de calibración para cada etiqueta.
    
    Args:
        y_test: Matriz de etiquetas reales.
        y_pred_proba: Matriz de probabilidades predichas.
        mlb_classes: Nombres de las etiquetas.
        n_bins: Número de bins para el cálculo de calibración.
        
    Returns:
        Diccionario con métricas de calibración por etiqueta.
    """
    from sklearn.calibration import calibration_curve
    
    calibration_metrics = {}
    
    # Para cada etiqueta
    for i, label in enumerate(mlb_classes):
        # Calcular curva de calibración
        prob_true, prob_pred = calibration_curve(y_test[:, i], y_pred_proba[:, i], n_bins=n_bins)
        
        # Calcular error de calibración (ECE - Expected Calibration Error)
        # Dividir predicciones en bins
        bin_indices = np.minimum(n_bins - 1, np.floor(y_pred_proba[:, i] * n_bins).astype(int))
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_counts = np.maximum(bin_counts, 1)  # Evitar división por cero
        
        # Calcular precisión y confianza promedio por bin
        bin_sums = np.bincount(bin_indices, weights=y_test[:, i], minlength=n_bins)
        bin_true = bin_sums / bin_counts
        
        bin_proba_sums = np.bincount(bin_indices, weights=y_pred_proba[:, i], minlength=n_bins)
        bin_pred = bin_proba_sums / bin_counts
        
        # Calcular ECE
        ece = np.sum(np.abs(bin_true - bin_pred) * (bin_counts / len(y_test[:, i])))
        
        # Calcular Brier Score
        brier_score = np.mean((y_pred_proba[:, i] - y_test[:, i]) ** 2)
        
        calibration_metrics[label] = {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'ece': ece,
            'brier_score': brier_score
        }
    
    return calibration_metrics


def evaluate_model_with_optimal_thresholds(y_test, y_pred_proba, mlb_classes, thresholds):
    """Evalúa el modelo utilizando umbrales óptimos personalizados para cada etiqueta.
    
    Args:
        y_test: Matriz de etiquetas reales.
        y_pred_proba: Matriz de probabilidades predichas.
        mlb_classes: Nombres de las etiquetas.
        thresholds: Diccionario con umbrales por etiqueta.
        
    Returns:
        Diccionario con métricas por etiqueta.
    """
    # Crear matriz de predicciones binarias usando umbrales personalizados
    y_pred = np.zeros_like(y_test)
    
    for i, label in enumerate(mlb_classes):
        threshold = thresholds.get(label, 0.5)  # Usar 0.5 si no se proporciona umbral
        y_pred[:, i] = (y_pred_proba[:, i] >= threshold).astype(int)
    
    # Calcular métricas
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision_micro'] = precision_score(y_test, y_pred, average='micro')
    metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
    metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    metrics['recall_micro'] = recall_score(y_test, y_pred, average='micro')
    metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
    metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
    metrics['f1_micro'] = f1_score(y_test, y_pred, average='micro')
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    metrics['hamming_loss'] = hamming_loss(y_test, y_pred)
    metrics['jaccard_score'] = jaccard_score(y_test, y_pred, average='samples')
    
    # Calcular métricas por etiqueta
    report = classification_report(y_test, y_pred, target_names=mlb_classes, output_dict=True)
    metrics['per_label'] = report
    
    # Añadir información de umbrales utilizados
    metrics['thresholds_used'] = thresholds
    
    return metrics, y_pred


def analyze_error_patterns(y_test, y_pred, mlb_classes):
    """Analiza patrones de error en las predicciones.
    
    Args:
        y_test: Matriz de etiquetas reales.
        y_pred: Matriz de etiquetas predichas.
        mlb_classes: Nombres de las etiquetas.
        
    Returns:
        Diccionario con índices de errores por tipo y etiqueta.
    """
    error_patterns = {}
    
    # Para cada etiqueta
    for i, label in enumerate(mlb_classes):
        # Encontrar índices de falsos positivos y falsos negativos
        false_positives = np.where((y_test[:, i] == 0) & (y_pred[:, i] == 1))[0]
        false_negatives = np.where((y_test[:, i] == 1) & (y_pred[:, i] == 0))[0]
        
        error_patterns[label] = {
            'false_positives': false_positives.tolist(),
            'false_negatives': false_negatives.tolist()
        }
    
    # Analizar errores comunes entre etiquetas
    common_errors = {}
    for i, label1 in enumerate(mlb_classes):
        for j, label2 in enumerate(mlb_classes):
            if i < j:  # Evitar duplicados
                # Errores comunes entre dos etiquetas
                fp_intersection = set(error_patterns[label1]['false_positives']).intersection(
                    set(error_patterns[label2]['false_positives']))
                fn_intersection = set(error_patterns[label1]['false_negatives']).intersection(
                    set(error_patterns[label2]['false_negatives']))
                
                common_errors[f"{label1}_and_{label2}"] = {
                    'common_false_positives': list(fp_intersection),
                    'common_false_negatives': list(fn_intersection)
                }
    
    error_patterns['common_errors'] = common_errors
    
    return error_patterns


def perform_cross_validation(model, X, y, mlb_classes, cv=5, stratified=True):
    """Realiza validación cruzada estratificada para evaluar el modelo.
    
    Args:
        model: Modelo a evaluar.
        X: Características.
        y: Etiquetas.
        mlb_classes: Nombres de las etiquetas.
        cv: Número de folds para la validación cruzada.
        stratified: Si True, utiliza validación cruzada estratificada.
        
    Returns:
        Diccionario con métricas de validación cruzada.
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    import copy
    
    # Inicializar métricas
    cv_metrics = {
        'accuracy': [],
        'precision_micro': [],
        'precision_macro': [],
        'precision_weighted': [],
        'recall_micro': [],
        'recall_macro': [],
        'recall_weighted': [],
        'f1_micro': [],
        'f1_macro': [],
        'f1_weighted': [],
        'hamming_loss': [],
        'jaccard_score': []
    }
    
    # Inicializar validación cruzada
    if stratified:
        # Para validación cruzada estratificada, necesitamos una etiqueta única por muestra
        # Usamos la etiqueta más frecuente como estratificación
        most_common_label = np.argmax(np.sum(y, axis=0))
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        splits = kf.split(X, y[:, most_common_label])
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        splits = kf.split(X)
    
    # Realizar validación cruzada
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clonar y entrenar modelo
        model_clone = copy.deepcopy(model)
        model_clone.fit(X_train, y_train)
        
        # Predecir
        y_pred = model_clone.predict(X_test)
        
        # Calcular métricas
        cv_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_metrics['precision_micro'].append(precision_score(y_test, y_pred, average='micro'))
        cv_metrics['precision_macro'].append(precision_score(y_test, y_pred, average='macro'))
        cv_metrics['precision_weighted'].append(precision_score(y_test, y_pred, average='weighted'))
        cv_metrics['recall_micro'].append(recall_score(y_test, y_pred, average='micro'))
        cv_metrics['recall_macro'].append(recall_score(y_test, y_pred, average='macro'))
        cv_metrics['recall_weighted'].append(recall_score(y_test, y_pred, average='weighted'))
        cv_metrics['f1_micro'].append(f1_score(y_test, y_pred, average='micro'))
        cv_metrics['f1_macro'].append(f1_score(y_test, y_pred, average='macro'))
        cv_metrics['f1_weighted'].append(f1_score(y_test, y_pred, average='weighted'))
        cv_metrics['hamming_loss'].append(hamming_loss(y_test, y_pred))
        cv_metrics['jaccard_score'].append(jaccard_score(y_test, y_pred, average='samples'))
    
    # Calcular medias y desviaciones estándar
    cv_results = {}
    for metric, values in cv_metrics.items():
        cv_results[f'{metric}_mean'] = np.mean(values)
        cv_results[f'{metric}_std'] = np.std(values)
    
    return cv_results


def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)):
    """Genera curvas de aprendizaje para evaluar el rendimiento del modelo con diferentes tamaños de datos.
    
    Args:
        model: Modelo a evaluar.
        X: Características.
        y: Etiquetas.
        cv: Número de folds para la validación cruzada.
        train_sizes: Tamaños relativos de los conjuntos de entrenamiento.
        
    Returns:
        Diccionario con resultados de las curvas de aprendizaje.
    """
    from sklearn.model_selection import learning_curve
    
    # Calcular curvas de aprendizaje
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring='f1_weighted', n_jobs=-1)
    
    # Calcular medias y desviaciones estándar
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    return {
        'train_sizes': train_sizes_abs,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std
    }


def evaluate_model(model, vectorizer, mlb, X_test, y_test):
    """Evalúa el modelo con datos de prueba y calcula métricas de rendimiento.
    
    Args:
        model: Modelo entrenado para la clasificación.
        vectorizer: Vectorizador TF-IDF entrenado.
        mlb: MultiLabelBinarizer entrenado.
        X_test: Características de los datos de prueba.
        y_test: Etiquetas reales de los datos de prueba.
        
    Returns:
        dict: Diccionario con las métricas de rendimiento.
    """
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Intentar obtener probabilidades si el modelo lo soporta
    try:
        y_pred_proba = model.predict_proba(X_test)
    except:
        y_pred_proba = None
    
    # Calcular métricas
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision_micro'] = precision_score(y_test, y_pred, average='micro')
    metrics['precision_macro'] = precision_score(y_test, y_pred, average='macro')
    metrics['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    metrics['recall_micro'] = recall_score(y_test, y_pred, average='micro')
    metrics['recall_macro'] = recall_score(y_test, y_pred, average='macro')
    metrics['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
    metrics['f1_micro'] = f1_score(y_test, y_pred, average='micro')
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    metrics['hamming_loss'] = hamming_loss(y_test, y_pred)
    metrics['jaccard_score'] = jaccard_score(y_test, y_pred, average='samples')
    
    # Calcular métricas por etiqueta
    report = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True)
    metrics['per_label'] = report
    
    # Análisis avanzado si hay probabilidades disponibles
    if y_pred_proba is not None:
        # Análisis de umbrales
        metrics['threshold_analysis'] = analyze_threshold_impact(y_test, y_pred_proba, mlb.classes_)
        
        # Umbrales óptimos
        metrics['optimal_thresholds'] = find_optimal_thresholds(y_test, y_pred_proba, mlb.classes_)
        
        # Evaluación con umbrales óptimos
        metrics['optimal_threshold_metrics'], y_pred_optimal = evaluate_model_with_optimal_thresholds(
            y_test, y_pred_proba, mlb.classes_, metrics['optimal_thresholds'])
        
        # Distribución de probabilidades
        metrics['probability_distributions'] = analyze_probability_distribution(y_test, y_pred_proba, mlb.classes_)
        
        # Métricas de calibración
        metrics['calibration_metrics'] = calculate_calibration_metrics(y_test, y_pred_proba, mlb.classes_)
    
    # Análisis de patrones de error
    metrics['error_patterns'] = analyze_error_patterns(y_test, y_pred, mlb.classes_)
    
    return metrics, y_pred, y_pred_proba if y_pred_proba is not None else None


def display_metrics(metrics):
    """Muestra las métricas de rendimiento del modelo.
    
    Args:
        metrics (dict): Diccionario con las métricas de rendimiento.
    """
    print("\n=== MÉTRICAS DE RENDIMIENTO ===\n")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (micro): {metrics['precision_micro']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (micro): {metrics['recall_micro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-score (micro): {metrics['f1_micro']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Jaccard Score: {metrics['jaccard_score']:.4f}")
    
    # Mostrar información de umbrales óptimos si está disponible
    if 'optimal_thresholds' in metrics:
        print("\n=== UMBRALES ÓPTIMOS POR CATEGORÍA ===\n")
        for label, threshold in metrics['optimal_thresholds'].items():
            print(f"{label}: {threshold:.2f}")
    
    print("\n=== MÉTRICAS POR CATEGORÍA ===\n")
    per_label = metrics['per_label']
    
    # Crear tabla de métricas por categoría
    categories = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for category, values in per_label.items():
        if category not in ['accuracy', 'macro avg', 'micro avg', 'weighted avg']:
            categories.append(category)
            precisions.append(values['precision'])
            recalls.append(values['recall'])
            f1_scores.append(values['f1-score'])
            supports.append(values['support'])
    
    # Crear DataFrame para mostrar las métricas por categoría
    df_metrics = pd.DataFrame({
        'Categoría': categories,
        'Precision': precisions,
        'Recall': recalls,
        'F1-score': f1_scores,
        'Support': supports
    })
    
    print(df_metrics.to_string(index=False))
    
    # Mostrar análisis de errores
    if 'error_patterns' in metrics:
        print("\n=== ANÁLISIS DE ERRORES ===\n")
        error_patterns = metrics['error_patterns']
        for label, errors in error_patterns.items():
            if label != 'common_errors':
                print(f"{label}:")
                print(f"  Falsos Positivos: {len(errors['false_positives'])}")
                print(f"  Falsos Negativos: {len(errors['false_negatives'])}")


def visualize_results(metrics, y_test, y_pred, y_pred_proba, mlb):
    """Visualiza los resultados de la evaluación del modelo.
    
    Args:
        metrics (dict): Diccionario con las métricas de rendimiento.
        y_test: Etiquetas reales de los datos de prueba.
        y_pred: Etiquetas predichas por el modelo.
        y_pred_proba: Probabilidades predichas por el modelo (puede ser None).
        mlb: MultiLabelBinarizer entrenado.
    """
    # Crear directorio para guardar las visualizaciones si no existe
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualizar métricas con gráfico de radar
    radar_metrics = {
        'Precision': metrics['precision_weighted'],
        'Recall': metrics['recall_weighted'],
        'F1-score': metrics['f1_weighted'],
        'Accuracy': metrics['accuracy'],
        'Jaccard': metrics['jaccard_score']
    }
    
    plot_metrics_radar(radar_metrics, title="Métricas del Modelo")
    plt.savefig(os.path.join(output_dir, 'model_metrics_radar.png'))
    plt.close()
    
    # Visualizar matriz de confusión para cada categoría
    for i, category in enumerate(mlb.classes_):
        plot_confusion_matrix(
            y_test[:, i], 
            y_pred[:, i], 
            classes=['No ' + category, category],
            title=f'Matriz de Confusión - {category}'
        )
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{category}.png'))
        plt.close()
    
    # Visualizar análisis de umbrales si hay probabilidades disponibles
    if y_pred_proba is not None and 'threshold_analysis' in metrics:
        # Visualizar impacto de umbrales en F1-score para cada categoría
        plt.figure(figsize=(12, 8))
        for i, category in enumerate(mlb.classes_):
            threshold_results = metrics['threshold_analysis'][category]
            plt.plot(threshold_results['threshold'], threshold_results['f1'], 
                     label=f"{category} (óptimo: {metrics['optimal_thresholds'][category]:.2f})")
        
        plt.title('Impacto de Umbrales en F1-score por Categoría')
        plt.xlabel('Umbral')
        plt.ylabel('F1-score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_impact_f1.png'))
        plt.close()
        
        # Visualizar curvas de precisión-recall para cada categoría
        plt.figure(figsize=(12, 8))
        for i, category in enumerate(mlb.classes_):
            threshold_results = metrics['threshold_analysis'][category]
            plt.plot(threshold_results['recall'], threshold_results['precision'], 
                     label=category, marker='o', markersize=3)
        
        plt.title('Curvas de Precisión-Recall por Categoría')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
        plt.close()
        
        # Visualizar distribución de probabilidades para una categoría de ejemplo
        if len(mlb.classes_) > 0:
            example_category = mlb.classes_[0]
            plt.figure(figsize=(10, 6))
            
            # Obtener distribuciones
            dist = metrics['probability_distributions'][example_category]
            
            # Histograma para muestras positivas
            if len(dist['positive_samples']) > 0:
                plt.hist(dist['positive_samples'], bins=20, alpha=0.7, 
                         label='Muestras Positivas', color='green')
            
            # Histograma para muestras negativas
            if len(dist['negative_samples']) > 0:
                plt.hist(dist['negative_samples'], bins=20, alpha=0.7, 
                         label='Muestras Negativas', color='red')
            
            plt.axvline(x=metrics['optimal_thresholds'][example_category], 
                       color='black', linestyle='--', 
                       label=f'Umbral Óptimo: {metrics["optimal_thresholds"][example_category]:.2f}')
            
            plt.title(f'Distribución de Probabilidades - {example_category}')
            plt.xlabel('Probabilidad Predicha')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'probability_distribution_{example_category}.png'))
            plt.close()
    
    print(f"\nVisualizaciones guardadas en {output_dir}")


def main():
    """Función principal para evaluar un modelo con datos de prueba."""
    parser = argparse.ArgumentParser(description='Evaluar un modelo de clasificación de literatura médica')
    parser.add_argument('--model', type=str, help='Ruta al archivo del modelo entrenado')
    parser.add_argument('--data', type=str, help='Ruta al archivo CSV con datos de prueba')
    parser.add_argument('--cross-validation', action='store_true', help='Realizar validación cruzada')
    parser.add_argument('--cv-folds', type=int, default=5, help='Número de folds para validación cruzada')
    parser.add_argument('--learning-curve', action='store_true', help='Generar curva de aprendizaje')
    args = parser.parse_args()
    
    # Si no se proporcionan argumentos, usar valores por defecto o solicitar al usuario
    if args.model is None:
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        available_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not available_models:
            print("No se encontraron modelos entrenados en la carpeta 'models/'")
            print("Por favor, entrene un modelo primero usando train_model.py")
            return
        
        print("Modelos disponibles:")
        for i, model_file in enumerate(available_models):
            print(f"  {i+1}. {model_file}")
        
        try:
            model_idx = int(input("\nSeleccione un modelo por número: ")) - 1
            if model_idx < 0 or model_idx >= len(available_models):
                print("Selección inválida")
                return
            model_path = os.path.join(models_dir, available_models[model_idx])
        except ValueError:
            print("Por favor, ingrese un número válido")
            return
    else:
        model_path = args.model
    
    # Cargar el modelo
    model, vectorizer, mlb = load_model(model_path)
    if model is None:
        return
    
    # Cargar datos de prueba
    if args.data is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        test_data_path = os.path.join(data_dir, 'processed', 'test_data.csv')
        if not os.path.exists(test_data_path):
            test_data_path = os.path.join(data_dir, 'raw', 'challenge_data-18-ago.csv')
    else:
        test_data_path = args.data
    
    df_test = load_test_data(test_data_path)
    if df_test is None:
        return
    
    # Preprocesar datos de prueba
    df_processed = preprocess_dataset(df_test)
    
    # Extraer características
    print("Extrayendo características...")
    X_test = vectorizer.transform(df_processed['text_processed'])
    
    # Preparar etiquetas
    print("Preparando etiquetas...")
    y_test = mlb.transform(df_processed['group'].str.split('|'))
    
    # Evaluar modelo
    print("Evaluando modelo...")
    metrics, y_pred, y_pred_proba = evaluate_model(model, vectorizer, mlb, X_test, y_test)
    
    # Mostrar métricas
    display_metrics(metrics)
    
    # Visualizar resultados
    visualize_results(metrics, y_test, y_pred, y_pred_proba, mlb)
    
    # Realizar validación cruzada si se solicita
    if args.cross_validation:
        print("\n=== VALIDACIÓN CRUZADA ===\n")
        cv_results = perform_cross_validation(model, X_test.toarray(), y_test, mlb.classes_, cv=args.cv_folds)
        
        print(f"Resultados de validación cruzada ({args.cv_folds} folds):")
        for metric, value in cv_results.items():
            if metric.endswith('_mean'):
                base_metric = metric.replace('_mean', '')
                print(f"{base_metric}: {value:.4f} ± {cv_results[base_metric + '_std']:.4f}")
    
    # Generar curva de aprendizaje si se solicita
    if args.learning_curve:
        print("\n=== CURVA DE APRENDIZAJE ===\n")
        print("Generando curva de aprendizaje... (esto puede tardar un poco)")
        
        # Convertir a array denso para la curva de aprendizaje
        X_dense = X_test.toarray()
        
        # Calcular curva de aprendizaje
        lc_results = plot_learning_curve(model, X_dense, y_test)
        
        # Visualizar curva de aprendizaje
        plt.figure(figsize=(10, 6))
        plt.plot(lc_results['train_sizes'], lc_results['train_mean'], 'o-', color='r', label='Entrenamiento')
        plt.fill_between(lc_results['train_sizes'], 
                        lc_results['train_mean'] - lc_results['train_std'],
                        lc_results['train_mean'] + lc_results['train_std'], 
                        alpha=0.1, color='r')
        
        plt.plot(lc_results['train_sizes'], lc_results['test_mean'], 'o-', color='g', label='Validación')
        plt.fill_between(lc_results['train_sizes'], 
                        lc_results['test_mean'] - lc_results['test_std'],
                        lc_results['test_mean'] + lc_results['test_std'], 
                        alpha=0.1, color='g')
        
        plt.title('Curva de Aprendizaje')
        plt.xlabel('Tamaño del Conjunto de Entrenamiento')
        plt.ylabel('F1-score Ponderado')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures', 'learning_curve.png'))
        plt.close()
        
        print("Curva de aprendizaje generada y guardada.")


if __name__ == "__main__":
    main()