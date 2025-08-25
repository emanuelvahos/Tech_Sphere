# -*- coding: utf-8 -*-
"""
Módulo para el entrenamiento y evaluación de modelos de clasificación multi-etiqueta.

Este módulo contiene funciones para entrenar, evaluar y optimizar modelos
de clasificación multi-etiqueta para literatura médica.
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
from typing import List, Dict, Union, Tuple, Optional, Any, Callable

# Importaciones para modelos de machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, hamming_loss, jaccard_score

# Para manejo de desbalance de clases
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def load_features_and_labels(features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga las características y etiquetas desde archivos.
    
    Args:
        features_path: Ruta al archivo de características.
        labels_path: Ruta al archivo de etiquetas.
        
    Returns:
        Tupla con (matriz de características, matriz de etiquetas).
    """
    X = np.load(features_path)
    y = np.load(labels_path)
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
              val_size: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        X: Matriz de características.
        y: Matriz de etiquetas.
        test_size: Proporción del conjunto de prueba.
        val_size: Proporción del conjunto de validación.
        random_state: Semilla para reproducibilidad.
        
    Returns:
        Tupla con (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Primero separamos el conjunto de prueba
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y.sum(axis=1)
    )
    
    # Luego separamos el conjunto de validación del conjunto temporal
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp.sum(axis=1)
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_model(model_type: str, **kwargs) -> Any:
    """
    Crea un modelo de clasificación multi-etiqueta.
    
    Args:
        model_type: Tipo de modelo ('logistic', 'svm', 'rf', 'gb', 'mlp').
        **kwargs: Parámetros adicionales para el modelo.
        
    Returns:
        Modelo de clasificación multi-etiqueta.
    """
    # Configurar el modelo base según el tipo
    if model_type.lower() == 'logistic':
        base_model = LogisticRegression(
            C=kwargs.get('C', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            class_weight=kwargs.get('class_weight', 'balanced')
        )
    elif model_type.lower() == 'svm':
        base_model = LinearSVC(
            C=kwargs.get('C', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            class_weight=kwargs.get('class_weight', 'balanced')
        )
    elif model_type.lower() == 'rf':
        base_model = RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            random_state=kwargs.get('random_state', 42),
            class_weight=kwargs.get('class_weight', 'balanced')
        )
    elif model_type.lower() == 'gb':
        base_model = GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 3),
            random_state=kwargs.get('random_state', 42)
        )
    elif model_type.lower() == 'mlp':
        base_model = MLPClassifier(
            hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100,)),
            activation=kwargs.get('activation', 'relu'),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    # Envolver en OneVsRestClassifier para clasificación multi-etiqueta
    model = OneVsRestClassifier(base_model)
    
    return model


def handle_class_imbalance(X: np.ndarray, y: np.ndarray, method: str = 'smote',
                         sampling_strategy: Union[str, Dict] = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Maneja el desbalance de clases en los datos.
    
    Args:
        X: Matriz de características.
        y: Matriz de etiquetas.
        method: Método para manejar el desbalance ('smote', 'undersample', 'none').
        sampling_strategy: Estrategia de muestreo.
        
    Returns:
        Tupla con (X_resampled, y_resampled).
    """
    if method.lower() == 'none':
        return X, y
    
    # Para clasificación multi-etiqueta, aplicamos el método a cada etiqueta por separado
    X_resampled, y_resampled = X.copy(), y.copy()
    
    for i in range(y.shape[1]):
        if method.lower() == 'smote':
            resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        elif method.lower() == 'undersample':
            resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        else:
            raise ValueError(f"Método de manejo de desbalance no soportado: {method}")
        
        # Aplicar el método de resampling a esta etiqueta
        X_temp, y_temp = resampler.fit_resample(X, y[:, i])
        
        # Si es la primera etiqueta, inicializamos los arrays
        if i == 0:
            X_resampled = X_temp
            y_resampled = np.zeros((X_temp.shape[0], y.shape[1]))
        
        # Actualizar la columna correspondiente en y_resampled
        y_resampled[:, i] = y_temp
    
    return X_resampled, y_resampled


def train_and_evaluate(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    """
    Entrena y evalúa un modelo.
    
    Args:
        model: Modelo a entrenar.
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_val: Características de validación.
        y_val: Etiquetas de validación.
        
    Returns:
        Diccionario con métricas de evaluación.
    """
    # Registrar tiempo de inicio
    start_time = time.time()
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Tiempo de entrenamiento
    train_time = time.time() - start_time
    
    # Predecir en conjunto de validación
    y_pred = model.predict(X_val)
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision_micro': precision_score(y_val, y_pred, average='micro'),
        'precision_macro': precision_score(y_val, y_pred, average='macro'),
        'precision_weighted': precision_score(y_val, y_pred, average='weighted'),
        'recall_micro': recall_score(y_val, y_pred, average='micro'),
        'recall_macro': recall_score(y_val, y_pred, average='macro'),
        'recall_weighted': recall_score(y_val, y_pred, average='weighted'),
        'f1_micro': f1_score(y_val, y_pred, average='micro'),
        'f1_macro': f1_score(y_val, y_pred, average='macro'),
        'f1_weighted': f1_score(y_val, y_pred, average='weighted'),
        'hamming_loss': hamming_loss(y_val, y_pred),
        'jaccard_micro': jaccard_score(y_val, y_pred, average='micro'),
        'jaccard_macro': jaccard_score(y_val, y_pred, average='macro'),
        'jaccard_weighted': jaccard_score(y_val, y_pred, average='weighted'),
        'train_time': train_time
    }
    
    return metrics


def optimize_hyperparameters(model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                           param_grid: Dict[str, List], cv: int = 3) -> Tuple[Any, Dict[str, Any]]:
    """
    Optimiza los hiperparámetros de un modelo.
    
    Args:
        model_type: Tipo de modelo.
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        param_grid: Cuadrícula de parámetros a probar.
        cv: Número de folds para validación cruzada.
        
    Returns:
        Tupla con (mejor modelo, mejores parámetros).
    """
    # Crear modelo base
    base_model = create_model(model_type)
    
    # Configurar búsqueda de hiperparámetros
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    # Ejecutar búsqueda
    grid_search.fit(X_train, y_train)
    
    # Obtener mejor modelo y parámetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    return best_model, best_params


def evaluate_final_model(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                       label_names: List[str]) -> Dict[str, Any]:
    """
    Evalúa el modelo final en el conjunto de prueba.
    
    Args:
        model: Modelo entrenado.
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        label_names: Nombres de las etiquetas.
        
    Returns:
        Diccionario con métricas y resultados detallados.
    """
    # Predecir en conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas generales
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_micro': precision_score(y_test, y_pred, average='micro'),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_micro': recall_score(y_test, y_pred, average='micro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_micro': f1_score(y_test, y_pred, average='micro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'hamming_loss': hamming_loss(y_test, y_pred),
        'jaccard_micro': jaccard_score(y_test, y_pred, average='micro'),
        'jaccard_macro': jaccard_score(y_test, y_pred, average='macro'),
        'jaccard_weighted': jaccard_score(y_test, y_pred, average='weighted')
    }
    
    # Calcular métricas por etiqueta
    per_label_metrics = {}
    for i, label in enumerate(label_names):
        per_label_metrics[label] = {
            'precision': precision_score(y_test[:, i], y_pred[:, i]),
            'recall': recall_score(y_test[:, i], y_pred[:, i]),
            'f1': f1_score(y_test[:, i], y_pred[:, i]),
            'support': np.sum(y_test[:, i])
        }
    
    # Calcular matrices de confusión por etiqueta
    confusion_matrices = {}
    for i, label in enumerate(label_names):
        confusion_matrices[label] = confusion_matrix(y_test[:, i], y_pred[:, i])
    
    # Reporte de clasificación detallado
    classification_rep = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    # Compilar todos los resultados
    results = {
        'metrics': metrics,
        'per_label_metrics': per_label_metrics,
        'confusion_matrices': confusion_matrices,
        'classification_report': classification_rep,
        'y_pred': y_pred
    }
    
    return results


def save_model_and_results(model: Any, results: Dict[str, Any], output_dir: str,
                         model_name: str) -> None:
    """
    Guarda el modelo entrenado y los resultados de evaluación.
    
    Args:
        model: Modelo entrenado.
        results: Resultados de evaluación.
        output_dir: Directorio de salida.
        model_name: Nombre del modelo.
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar modelo
    with open(os.path.join(output_dir, f"{model_name}.pkl"), 'wb') as f:
        pickle.dump(model, f)
    
    # Guardar resultados
    with open(os.path.join(output_dir, f"{model_name}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    # Guardar métricas principales en formato CSV para fácil acceso
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)
    
    # Guardar métricas por etiqueta en formato CSV
    per_label_df = pd.DataFrame.from_dict(results['per_label_metrics'], orient='index')
    per_label_df.to_csv(os.path.join(output_dir, f"{model_name}_per_label_metrics.csv"))


def main():
    """
    Función principal para ejecutar el entrenamiento y evaluación desde la línea de comandos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar y evaluar modelos de clasificación multi-etiqueta')
    parser.add_argument('--features', type=str, required=True, help='Ruta al archivo de características')
    parser.add_argument('--labels', type=str, required=True, help='Ruta al archivo de etiquetas')
    parser.add_argument('--output-dir', type=str, required=True, help='Directorio para guardar modelos y resultados')
    parser.add_argument('--model-type', type=str, default='logistic', 
                       choices=['logistic', 'svm', 'rf', 'gb', 'mlp'],
                       help='Tipo de modelo a entrenar')
    parser.add_argument('--optimize', action='store_true', help='Optimizar hiperparámetros')
    parser.add_argument('--balance-method', type=str, default='none',
                       choices=['none', 'smote', 'undersample'],
                       help='Método para manejar desbalance de clases')
    
    args = parser.parse_args()
    
    # Cargar características y etiquetas
    print(f"Cargando características y etiquetas...")
    X, y = load_features_and_labels(args.features, args.labels)
    
    # Dividir datos
    print("Dividiendo datos en conjuntos de entrenamiento, validación y prueba...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Manejar desbalance de clases si se solicita
    if args.balance_method != 'none':
        print(f"Aplicando método de balance de clases: {args.balance_method}...")
        X_train, y_train = handle_class_imbalance(X_train, y_train, method=args.balance_method)
    
    # Nombres de etiquetas
    label_names = ['cardiovascular', 'hepatorenal', 'neurological', 'oncological']
    
    # Optimizar hiperparámetros si se solicita
    if args.optimize:
        print(f"Optimizando hiperparámetros para modelo {args.model_type}...")
        
        # Definir cuadrícula de parámetros según el tipo de modelo
        if args.model_type == 'logistic':
            param_grid = {
                'estimator__C': [0.1, 1.0, 10.0],
                'estimator__class_weight': ['balanced', None]
            }
        elif args.model_type == 'svm':
            param_grid = {
                'estimator__C': [0.1, 1.0, 10.0],
                'estimator__class_weight': ['balanced', None]
            }
        elif args.model_type == 'rf':
            param_grid = {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [None, 10, 20],
                'estimator__min_samples_split': [2, 5, 10]
            }
        elif args.model_type == 'gb':
            param_grid = {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__learning_rate': [0.01, 0.1, 0.2],
                'estimator__max_depth': [3, 5, 7]
            }
        elif args.model_type == 'mlp':
            param_grid = {
                'estimator__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'estimator__activation': ['relu', 'tanh'],
                'estimator__alpha': [0.0001, 0.001, 0.01]
            }
        
        # Ejecutar optimización
        model, best_params = optimize_hyperparameters(
            args.model_type, X_train, y_train, param_grid, cv=3
        )
        
        print(f"Mejores parámetros encontrados: {best_params}")
    else:
        # Crear modelo con parámetros por defecto
        print(f"Creando modelo {args.model_type} con parámetros por defecto...")
        model = create_model(args.model_type)
    
    # Entrenar y evaluar en conjunto de validación
    print("Entrenando y evaluando modelo en conjunto de validación...")
    val_metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val)
    
    print("Métricas en conjunto de validación:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Evaluar modelo final en conjunto de prueba
    print("Evaluando modelo final en conjunto de prueba...")
    test_results = evaluate_final_model(model, X_test, y_test, label_names)
    
    print("Métricas en conjunto de prueba:")
    for metric, value in test_results['metrics'].items():
        if metric != 'confusion_matrices':
            print(f"  {metric}: {value:.4f}")
    
    # Guardar modelo y resultados
    model_name = f"{args.model_type}_model"
    print(f"Guardando modelo y resultados en {args.output_dir}...")
    save_model_and_results(model, test_results, args.output_dir, model_name)
    
    print("¡Entrenamiento y evaluación completados!")


if __name__ == "__main__":
    main()