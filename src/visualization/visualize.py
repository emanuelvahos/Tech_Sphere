# -*- coding: utf-8 -*-
"""
Módulo para la visualización de resultados de los modelos de clasificación.

Este módulo contiene funciones para generar visualizaciones de los resultados
de los modelos de clasificación multi-etiqueta para literatura médica.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import confusion_matrix
import itertools

# Configuración de estilo para las visualizaciones
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], normalize: bool = False,
                        title: str = 'Matriz de Confusión', cmap: Any = plt.cm.Blues,
                        figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Genera una visualización de la matriz de confusión.
    
    Args:
        cm: Matriz de confusión.
        classes: Nombres de las clases.
        normalize: Si se debe normalizar la matriz.
        title: Título del gráfico.
        cmap: Mapa de colores.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Valor Real')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                          metric_name: str = 'f1_weighted',
                          title: str = 'Comparación de F1 Ponderado',
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Compara una métrica específica entre diferentes modelos.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo.
        metric_name: Nombre de la métrica a comparar.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    models = list(metrics_dict.keys())
    values = [metrics[metric_name] for metrics in metrics_dict.values()]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(models, values, color=sns.color_palette('viridis', len(models)))
    
    # Añadir etiquetas con valores
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylim(0, max(values) * 1.15)  # Dar espacio para las etiquetas
    ax.set_xlabel('Modelo')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_per_label_metrics(per_label_metrics: Dict[str, Dict[str, float]],
                          metric_name: str = 'f1',
                          title: str = 'F1-Score por Categoría',
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Visualiza una métrica específica para cada etiqueta.
    
    Args:
        per_label_metrics: Métricas por etiqueta.
        metric_name: Nombre de la métrica a visualizar.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    labels = list(per_label_metrics.keys())
    values = [metrics[metric_name] for metrics in per_label_metrics.values()]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=sns.color_palette('viridis', len(labels)))
    
    # Añadir etiquetas con valores
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylim(0, max(values) * 1.15)  # Dar espacio para las etiquetas
    ax.set_xlabel('Categoría')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def plot_metrics_radar(metrics: Dict[str, float], 
                     metrics_to_plot: List[str] = None,
                     title: str = 'Métricas del Modelo',
                     figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Genera un gráfico de radar para visualizar múltiples métricas.
    
    Args:
        metrics: Diccionario con métricas.
        metrics_to_plot: Lista de métricas a incluir.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['precision_weighted', 'recall_weighted', 'f1_weighted', 
                          'accuracy', 'jaccard_weighted']
    
    # Filtrar métricas disponibles
    metrics_to_plot = [m for m in metrics_to_plot if m in metrics]
    
    # Obtener valores
    values = [metrics[m] for m in metrics_to_plot]
    
    # Número de variables
    N = len(metrics_to_plot)
    
    # Ángulos para cada eje
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el polígono
    
    # Valores para cada eje
    values += values[:1]  # Cerrar el polígono
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Dibujar polígono
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Etiquetas
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    
    # Ajustar límites del eje y
    ax.set_ylim(0, 1)
    
    # Añadir título
    plt.title(title, size=15, y=1.1)
    
    return fig


def plot_label_distribution(y: np.ndarray, label_names: List[str],
                          title: str = 'Distribución de Etiquetas',
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Visualiza la distribución de etiquetas en el conjunto de datos.
    
    Args:
        y: Matriz de etiquetas.
        label_names: Nombres de las etiquetas.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    # Contar ocurrencias de cada etiqueta
    label_counts = y.sum(axis=0)
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(label_names, label_counts, color=sns.color_palette('viridis', len(label_names)))
    
    # Añadir etiquetas con valores
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Categoría')
    ax.set_ylabel('Número de Documentos')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def plot_label_co_occurrence(y: np.ndarray, label_names: List[str],
                           title: str = 'Matriz de Co-ocurrencia de Etiquetas',
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualiza la co-ocurrencia de etiquetas en el conjunto de datos.
    
    Args:
        y: Matriz de etiquetas.
        label_names: Nombres de las etiquetas.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    # Calcular matriz de co-ocurrencia
    co_occurrence = np.zeros((len(label_names), len(label_names)))
    
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            # Contar documentos que tienen ambas etiquetas
            co_occurrence[i, j] = np.sum(np.logical_and(y[:, i], y[:, j]))
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='viridis',
               xticklabels=label_names, yticklabels=label_names, ax=ax)
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def plot_learning_curve(train_sizes: np.ndarray, train_scores: np.ndarray, 
                      test_scores: np.ndarray, title: str = 'Curva de Aprendizaje',
                      figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Visualiza la curva de aprendizaje de un modelo.
    
    Args:
        train_sizes: Tamaños de conjunto de entrenamiento.
        train_scores: Puntuaciones en entrenamiento.
        test_scores: Puntuaciones en prueba.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar puntuaciones medias
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Entrenamiento')
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Validación')
    
    # Graficar bandas de desviación estándar
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                   alpha=0.1, color='r')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                   alpha=0.1, color='g')
    
    ax.set_xlabel('Tamaño del Conjunto de Entrenamiento')
    ax.set_ylabel('Puntuación')
    ax.set_title(title)
    ax.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    return fig


def plot_feature_importance(feature_importance: np.ndarray, feature_names: List[str],
                          top_n: int = 20, title: str = 'Importancia de Características',
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Visualiza la importancia de las características.
    
    Args:
        feature_importance: Array con importancia de características.
        feature_names: Nombres de las características.
        top_n: Número de características principales a mostrar.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    # Crear DataFrame para facilitar la ordenación
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    
    # Ordenar por importancia y seleccionar top_n
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Característica')
    plt.tight_layout()
    
    return fig


def plot_roc_curve(fpr: Dict[str, np.ndarray], tpr: Dict[str, np.ndarray], 
                 roc_auc: Dict[str, float], title: str = 'Curva ROC',
                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualiza curvas ROC para clasificación multi-etiqueta.
    
    Args:
        fpr: Diccionario con tasas de falsos positivos por etiqueta.
        tpr: Diccionario con tasas de verdaderos positivos por etiqueta.
        roc_auc: Diccionario con áreas bajo la curva ROC por etiqueta.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar curva ROC para cada etiqueta
    for label, label_fpr in fpr.items():
        if label == 'micro':
            ax.plot(fpr[label], tpr[label], 
                   label=f'ROC micro-promedio (AUC = {roc_auc[label]:.2f})',
                   color='deeppink', linestyle=':', linewidth=4)
        elif label == 'macro':
            ax.plot(fpr[label], tpr[label],
                   label=f'ROC macro-promedio (AUC = {roc_auc[label]:.2f})',
                   color='navy', linestyle=':', linewidth=4)
        else:
            ax.plot(fpr[label], tpr[label],
                   label=f'ROC {label} (AUC = {roc_auc[label]:.2f})')
    
    # Graficar línea diagonal
    ax.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos')
    ax.set_ylabel('Tasa de Verdaderos Positivos')
    ax.set_title(title)
    ax.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_precision_recall_curve(precision: Dict[str, np.ndarray], recall: Dict[str, np.ndarray],
                              average_precision: Dict[str, float],
                              title: str = 'Curva Precisión-Exhaustividad',
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualiza curvas de precisión-exhaustividad para clasificación multi-etiqueta.
    
    Args:
        precision: Diccionario con precisión por etiqueta.
        recall: Diccionario con exhaustividad por etiqueta.
        average_precision: Diccionario con precisión promedio por etiqueta.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar curva para cada etiqueta
    for label, label_precision in precision.items():
        if label == 'micro':
            ax.plot(recall[label], precision[label],
                   label=f'Micro-promedio (AP = {average_precision[label]:.2f})',
                   color='deeppink', linestyle=':', linewidth=4)
        elif label == 'macro':
            ax.plot(recall[label], precision[label],
                   label=f'Macro-promedio (AP = {average_precision[label]:.2f})',
                   color='navy', linestyle=':', linewidth=4)
        else:
            ax.plot(recall[label], precision[label],
                   label=f'{label} (AP = {average_precision[label]:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Exhaustividad')
    ax.set_ylabel('Precisión')
    ax.set_title(title)
    ax.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                               label_names: List[str],
                               title: str = 'Distribución de Predicciones vs. Valores Reales',
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Compara la distribución de etiquetas reales y predichas.
    
    Args:
        y_true: Matriz de etiquetas reales.
        y_pred: Matriz de etiquetas predichas.
        label_names: Nombres de las etiquetas.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    # Contar ocurrencias de cada etiqueta
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)
    
    # Crear DataFrame para facilitar la visualización
    df = pd.DataFrame({
        'Etiqueta': label_names * 2,
        'Conteo': np.concatenate([true_counts, pred_counts]),
        'Tipo': ['Real'] * len(label_names) + ['Predicción'] * len(label_names)
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='Etiqueta', y='Conteo', hue='Tipo', data=df, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Categoría')
    ax.set_ylabel('Número de Documentos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def plot_multilabel_counts(y: np.ndarray, title: str = 'Distribución de Número de Etiquetas por Documento',
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Visualiza la distribución del número de etiquetas por documento.
    
    Args:
        y: Matriz de etiquetas.
        title: Título del gráfico.
        figsize: Tamaño de la figura.
        
    Returns:
        Figura de matplotlib.
    """
    # Contar número de etiquetas por documento
    label_counts = y.sum(axis=1)
    
    # Contar frecuencia de cada número de etiquetas
    count_distribution = np.bincount(label_counts.astype(int))
    
    # Crear índices para el gráfico
    indices = np.arange(len(count_distribution))
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(indices, count_distribution, color=sns.color_palette('viridis', len(count_distribution)))
    
    # Añadir etiquetas con valores
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Solo mostrar etiquetas para barras con altura > 0
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Número de Etiquetas')
    ax.set_ylabel('Número de Documentos')
    ax.set_title(title)
    ax.set_xticks(indices)
    plt.tight_layout()
    
    return fig


def save_figures(figures: Dict[str, plt.Figure], output_dir: str) -> None:
    """
    Guarda múltiples figuras en un directorio.
    
    Args:
        figures: Diccionario con nombres y figuras.
        output_dir: Directorio de salida.
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar cada figura
    for name, fig in figures.items():
        fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Carga resultados de evaluación desde un archivo pickle.
    
    Args:
        results_path: Ruta al archivo de resultados.
        
    Returns:
        Diccionario con resultados.
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def main():
    """
    Función principal para ejecutar visualizaciones desde la línea de comandos.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar visualizaciones para resultados de clasificación')
    parser.add_argument('--results', type=str, required=True, help='Ruta al archivo de resultados')
    parser.add_argument('--output-dir', type=str, required=True, help='Directorio para guardar visualizaciones')
    parser.add_argument('--model-name', type=str, default='model', help='Nombre del modelo para etiquetas')
    
    args = parser.parse_args()
    
    # Cargar resultados
    print(f"Cargando resultados desde {args.results}...")
    results = load_results(args.results)
    
    # Extraer métricas y datos necesarios
    metrics = results['metrics']
    per_label_metrics = results.get('per_label_metrics', {})
    confusion_matrices = results.get('confusion_matrices', {})
    y_pred = results.get('y_pred', None)
    
    # Nombres de etiquetas
    label_names = list(per_label_metrics.keys()) if per_label_metrics else ['Label 1', 'Label 2', 'Label 3', 'Label 4']
    
    # Crear figuras
    figures = {}
    
    # Métricas generales
    figures['metrics_radar'] = plot_metrics_radar(metrics, title=f'Métricas del Modelo {args.model_name}')
    
    # Métricas por etiqueta
    if per_label_metrics:
        figures['per_label_f1'] = plot_per_label_metrics(per_label_metrics, metric_name='f1',
                                                       title=f'F1-Score por Categoría - {args.model_name}')
        figures['per_label_precision'] = plot_per_label_metrics(per_label_metrics, metric_name='precision',
                                                             title=f'Precisión por Categoría - {args.model_name}')
        figures['per_label_recall'] = plot_per_label_metrics(per_label_metrics, metric_name='recall',
                                                          title=f'Exhaustividad por Categoría - {args.model_name}')
    
    # Matrices de confusión
    for label, cm in confusion_matrices.items():
        figures[f'confusion_matrix_{label}'] = plot_confusion_matrix(
            cm, classes=['Negativo', 'Positivo'],
            title=f'Matriz de Confusión - {label} - {args.model_name}'
        )
    
    # Guardar figuras
    print(f"Guardando {len(figures)} visualizaciones en {args.output_dir}...")
    save_figures(figures, args.output_dir)
    
    print("¡Visualizaciones generadas con éxito!")


if __name__ == "__main__":
    main()