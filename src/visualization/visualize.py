#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizaciones avanzadas e interactivas para análisis de resultados de modelos de clasificación multi-etiqueta.

Este módulo proporciona funciones para visualizar los resultados de modelos de clasificación
multi-etiqueta, incluyendo matrices de confusión, métricas por etiqueta, comparaciones
de métricas entre modelos, y visualizaciones interactivas para análisis exploratorio.
"""

import os
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import networkx as nx
from collections import Counter
import re
from nltk.util import ngrams


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: List[str] = None, normalize: bool = False,
                         title: str = 'Matriz de Confusión', 
                         cmap: str = 'Blues',
                         figsize: Tuple[int, int] = (10, 8),
                         interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Genera una matriz de confusión para cada etiqueta en un problema de clasificación multi-etiqueta.
    
    Args:
        y_true: Etiquetas verdaderas (one-hot encoding)
        y_pred: Etiquetas predichas (one-hot encoding)
        labels: Nombres de las etiquetas
        normalize: Si es True, normaliza los valores
        title: Título del gráfico
        cmap: Mapa de colores para la matriz
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    if interactive:
        # Crear una figura interactiva con Plotly
        n_labels = y_true.shape[1]
        fig = make_subplots(rows=int(np.ceil(n_labels/3)), cols=min(n_labels, 3),
                           subplot_titles=[labels[i] if labels else f"Label {i}" for i in range(n_labels)])
        
        # Crear una matriz de confusión para cada etiqueta
        for i in range(n_labels):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm = np.round(cm, 2)
            
            # Calcular posición en la cuadrícula
            row = i // 3 + 1
            col = i % 3 + 1
            
            # Crear heatmap
            heatmap = go.Heatmap(
                z=cm,
                x=['Negativo', 'Positivo'],
                y=['Negativo', 'Positivo'],
                colorscale=cmap.lower(),
                showscale=False,
                text=[[str(cm[i][j]) for j in range(2)] for i in range(2)],
                hoverinfo='text'
            )
            
            fig.add_trace(heatmap, row=row, col=col)
            
            # Configurar ejes
            fig.update_xaxes(title_text="Predicho", row=row, col=col)
            fig.update_yaxes(title_text="Real", row=row, col=col)
        
        # Ajustar diseño
        fig.update_layout(
            title_text=title,
            height=300 * int(np.ceil(n_labels/3)),
            width=900,
            showlegend=False
        )
        
        return fig
    else:
        # Versión estática con Matplotlib
        n_labels = y_true.shape[1]
        n_cols = min(3, n_labels)
        n_rows = int(np.ceil(n_labels / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_labels == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i in range(n_labels):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm = np.round(cm, 2)
            
            sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                      cmap=cmap, ax=axes[i], cbar=False)
            
            axes[i].set_title(labels[i] if labels else f"Label {i}")
            axes[i].set_xlabel('Predicho')
            axes[i].set_ylabel('Real')
            axes[i].set_xticklabels(['Negativo', 'Positivo'])
            axes[i].set_yticklabels(['Negativo', 'Positivo'])
        
        # Ocultar ejes no utilizados
        for i in range(n_labels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                           title: str = 'Comparación de Métricas entre Modelos',
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Compara métricas entre diferentes modelos.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
                     {nombre_modelo: {métrica: valor, ...}, ...}
        title: Título del gráfico
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        values = [metrics_dict[model][metric] for metric in metrics]
        ax.bar(x + i * width - width * (len(models) - 1) / 2, values, width, label=model)
    
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Valor')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_per_label_metrics(per_label_metrics: Dict[str, Dict[str, float]],
                          title: str = 'Métricas por Etiqueta',
                          figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Visualiza métricas para cada etiqueta.
    
    Args:
        per_label_metrics: Diccionario con métricas por etiqueta
                          {etiqueta: {métrica: valor, ...}, ...}
        title: Título del gráfico
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    labels = list(per_label_metrics.keys())
    metrics = list(per_label_metrics[labels[0]].keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        values = [per_label_metrics[label][metric] for label in labels]
        ax.bar(x + i * width - width * (len(metrics) - 1) / 2, values, width, label=metric)
    
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Valor')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_metrics_radar(metrics_dict: Dict[str, Dict[str, float]],
                      title: str = 'Comparación de Modelos',
                      figsize: Tuple[int, int] = (10, 10)) -> plt.Figure:
    """
    Genera un gráfico de radar para comparar métricas entre diferentes modelos.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
                     {nombre_modelo: {métrica: valor, ...}, ...}
        title: Título del gráfico
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())
    
    # Número de variables
    N = len(metrics)
    
    # Ángulos para cada eje
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Cerrar el polígono
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Añadir cada modelo
    for model in models:
        values = [metrics_dict[model][metric] for metric in metrics]
        values += values[:1]  # Cerrar el polígono
        
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Configurar ejes y etiquetas
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    
    # Añadir título y leyenda
    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig


def plot_label_distribution(y: np.ndarray, labels: List[str] = None,
                           title: str = 'Distribución de Etiquetas',
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Visualiza la distribución de etiquetas en el conjunto de datos.
    
    Args:
        y: Matriz de etiquetas (one-hot encoding)
        labels: Nombres de las etiquetas
        title: Título del gráfico
        figsize: Tamaño de la figura
        
    Returns:
        Figura de matplotlib
    """
    if labels is None:
        labels = [f"Label {i}" for i in range(y.shape[1])]
    
    # Contar ocurrencias de cada etiqueta
    label_counts = y.sum(axis=0)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=figsize)
    
    # Graficar barras
    ax.bar(labels, label_counts)
    
    # Configurar gráfico
    ax.set_title(title)
    ax.set_xlabel('Etiquetas')
    ax.set_ylabel('Frecuencia')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_word_cloud(texts: List[str], stopwords: List[str] = None, 
                   max_words: int = 200, background_color: str = 'white',
                   title: str = 'Nube de Palabras', figsize: Tuple[int, int] = (12, 8),
                   interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Genera una nube de palabras a partir de una lista de textos.
    
    Args:
        texts: Lista de textos
        stopwords: Lista de palabras a excluir
        max_words: Número máximo de palabras a incluir
        background_color: Color de fondo
        title: Título del gráfico
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    # Unir todos los textos
    text = ' '.join(texts)
    
    # Crear nube de palabras
    wordcloud = WordCloud(
        max_words=max_words,
        background_color=background_color,
        width=800,
        height=400,
        stopwords=stopwords
    ).generate(text)
    
    # Obtener palabras y frecuencias
    word_freqs = {word: freq for word, freq in wordcloud.words_.items()}
    
    if interactive:
        # Crear versión interactiva con Plotly
        words = list(word_freqs.keys())
        freqs = list(word_freqs.values())
        
        # Normalizar frecuencias para tamaño de texto
        sizes = [20 + 80 * (freq / max(freqs)) for freq in freqs]
        
        # Crear figura
        fig = go.Figure()
        
        # Añadir palabras como anotaciones
        for i, (word, size) in enumerate(zip(words[:100], sizes[:100])):
            fig.add_annotation(
                x=np.random.uniform(0, 1),
                y=np.random.uniform(0, 1),
                text=word,
                showarrow=False,
                font=dict(size=size, color=plt.cm.viridis(np.random.rand()))
            )
        
        # Configurar diseño
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=background_color,
            width=figsize[0]*100,
            height=figsize[1]*100
        )
        
        return fig
    else:
        # Versión estática con Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        plt.tight_layout()
        
        return fig


def plot_tsne_visualization(X: np.ndarray, categories: List[str], 
                           n_components: int = 2, perplexity: int = 30,
                           title: str = 'Visualización t-SNE', 
                           figsize: Tuple[int, int] = (12, 10),
                           interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Genera una visualización t-SNE para datos de alta dimensionalidad.
    
    Args:
        X: Matriz de características
        categories: Lista de categorías para cada muestra
        n_components: Número de componentes para t-SNE
        perplexity: Perplexidad para t-SNE
        title: Título del gráfico
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    # Aplicar t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Crear DataFrame para facilitar la visualización
    df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1] if n_components >= 2 else np.zeros(X_tsne.shape[0]),
        'z': X_tsne[:, 2] if n_components >= 3 else np.zeros(X_tsne.shape[0]),
        'category': categories
    })
    
    if interactive:
        # Versión interactiva con Plotly
        if n_components == 3:
            fig = px.scatter_3d(
                df, x='x', y='y', z='z',
                color='category',
                title=title,
                labels={'category': 'Categoría'},
                opacity=0.7,
                height=figsize[1]*80,
                width=figsize[0]*80
            )
        else:
            fig = px.scatter(
                df, x='x', y='y',
                color='category',
                title=title,
                labels={'category': 'Categoría'},
                opacity=0.7,
                height=figsize[1]*80,
                width=figsize[0]*80
            )
            
        # Añadir filtro de categorías
        unique_categories = df['category'].unique()
        buttons = []
        
        # Botón para mostrar todas las categorías
        buttons.append(dict(
            label="Todas",
            method="update",
            args=[{"visible": [True] * len(unique_categories)}]
        ))
        
        # Botones para cada categoría
        for i, cat in enumerate(unique_categories):
            visible = [False] * len(unique_categories)
            visible[i] = True
            buttons.append(dict(
                label=cat,
                method="update",
                args=[{"visible": visible}]
            ))
        
        # Añadir menú de filtros
        fig.update_layout(
            updatemenus=[dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )]
        )
        
        return fig
    else:
        # Versión estática con Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Obtener categorías únicas
        unique_categories = np.unique(categories)
        
        # Crear un color para cada categoría
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
        
        # Graficar cada categoría
        for i, category in enumerate(unique_categories):
            mask = df['category'] == category
            ax.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], 
                      c=[colors[i]], label=category, alpha=0.7)
        
        # Configurar gráfico
        ax.set_title(title)
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.grid(linestyle='--', alpha=0.7)
        
        # Añadir leyenda (si no hay demasiadas categorías)
        if len(unique_categories) <= 20:
            ax.legend()
        
        plt.tight_layout()
        
        return fig


def plot_learning_curves(train_sizes: np.ndarray, train_scores: np.ndarray, 
                        test_scores: np.ndarray, title: str = 'Curvas de Aprendizaje',
                        figsize: Tuple[int, int] = (10, 6),
                        interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Visualiza curvas de aprendizaje para evaluar el rendimiento del modelo.
    
    Args:
        train_sizes: Tamaños de conjuntos de entrenamiento
        train_scores: Puntuaciones en conjuntos de entrenamiento
        test_scores: Puntuaciones en conjuntos de prueba
        title: Título del gráfico
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    if interactive:
        # Versión interactiva con Plotly
        fig = go.Figure()
        
        # Añadir curva de entrenamiento
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Entrenamiento',
            line=dict(color='blue'),
            error_y=dict(
                type='data',
                array=train_std,
                visible=True,
                color='blue'
            )
        ))
        
        # Añadir curva de prueba
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=test_mean,
            mode='lines+markers',
            name='Validación',
            line=dict(color='red'),
            error_y=dict(
                type='data',
                array=test_std,
                visible=True,
                color='red'
            )
        ))
        
        # Configurar diseño
        fig.update_layout(
            title=title,
            xaxis_title='Tamaño del Conjunto de Entrenamiento',
            yaxis_title='Puntuación',
            legend=dict(x=0.02, y=0.98),
            template='plotly_white',
            height=figsize[1]*80,
            width=figsize[0]*80
        )
        
        return fig
    else:
        # Versión estática con Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Graficar curva de entrenamiento
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.1, color='blue')
        
        # Graficar curva de prueba
        ax.plot(train_sizes, test_mean, 'o-', color='red', label='Validación')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                       alpha=0.1, color='red')
        
        # Configurar gráfico
        ax.set_title(title)
        ax.set_xlabel('Tamaño del Conjunto de Entrenamiento')
        ax.set_ylabel('Puntuación')
        ax.grid(linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        return fig


def plot_feature_correlation(X: pd.DataFrame, feature_names: List[str] = None,
                            threshold: float = 0.7, title: str = 'Correlación de Características',
                            figsize: Tuple[int, int] = (12, 10),
                            interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Visualiza la correlación entre características.
    
    Args:
        X: DataFrame o matriz de características
        feature_names: Nombres de las características
        threshold: Umbral para resaltar correlaciones altas
        title: Título del gráfico
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    # Convertir a DataFrame si es necesario
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names if feature_names else 
                        [f'Feature {i}' for i in range(X.shape[1])])
    
    # Calcular matriz de correlación
    corr_matrix = X.corr()
    
    if interactive:
        # Versión interactiva con Plotly
        mask = np.abs(corr_matrix) > threshold
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Características", y="Características", color="Correlación"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title=title,
            width=figsize[0]*80,
            height=figsize[1]*80
        )
        
        # Resaltar correlaciones altas
        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, val in enumerate(row):
                if i != j and abs(val) > threshold:
                    annotations.append(dict(
                        x=j, y=i,
                        text=f"{val:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(val) > 0.8 else "black")
                    ))
        
        fig.update_layout(annotations=annotations)
        
        return fig
    else:
        # Versión estática con Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear mapa de calor
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5},
                   annot=True, fmt='.2f', ax=ax)
        
        ax.set_title(title)
        
        plt.tight_layout()
        
        return fig


def plot_text_length_distribution(text_lengths: List[int], bins: int = 30,
                                title: str = 'Distribución de Longitud de Textos',
                                figsize: Tuple[int, int] = (12, 6),
                                interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Visualiza la distribución de longitudes de texto.
    
    Args:
        text_lengths: Lista de longitudes de texto
        bins: Número de bins para el histograma
        title: Título del gráfico
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    # Calcular estadísticas
    mean_length = np.mean(text_lengths)
    median_length = np.median(text_lengths)
    std_length = np.std(text_lengths)
    
    if interactive:
        # Versión interactiva con Plotly
        fig = go.Figure()
        
        # Añadir histograma
        fig.add_trace(go.Histogram(
            x=text_lengths,
            nbinsx=bins,
            name='Frecuencia',
            opacity=0.7
        ))
        
        # Añadir línea de densidad
        hist_data, bin_edges = np.histogram(text_lengths, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=hist_data,
            mode='lines',
            name='Densidad',
            line=dict(color='red', width=2)
        ))
        
        # Añadir líneas para estadísticas
        fig.add_vline(x=mean_length, line_dash="dash", line_color="green",
                     annotation_text=f"Media: {mean_length:.1f}")
        fig.add_vline(x=median_length, line_dash="dash", line_color="blue",
                     annotation_text=f"Mediana: {median_length:.1f}")
        
        # Configurar diseño
        fig.update_layout(
            title=title,
            xaxis_title='Longitud del Texto',
            yaxis_title='Frecuencia',
            template='plotly_white',
            height=figsize[1]*80,
            width=figsize[0]*80,
            annotations=[
                dict(
                    x=0.02, y=0.95,
                    xref="paper", yref="paper",
                    text=f"Desviación Estándar: {std_length:.1f}",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
            ]
        )
        
        return fig
    else:
        # Versión estática con Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear histograma
        n, bins, patches = ax.hist(text_lengths, bins=bins, alpha=0.7, density=True)
        
        # Añadir línea de densidad
        density = sns.kdeplot(text_lengths, ax=ax, color='red')
        
        # Añadir líneas para estadísticas
        ax.axvline(mean_length, color='green', linestyle='dashed', linewidth=1,
                  label=f'Media: {mean_length:.1f}')
        ax.axvline(median_length, color='blue', linestyle='dashed', linewidth=1,
                  label=f'Mediana: {median_length:.1f}')
        
        # Configurar gráfico
        ax.set_title(title)
        ax.set_xlabel('Longitud del Texto')
        ax.set_ylabel('Densidad')
        ax.text(0.02, 0.95, f'Desviación Estándar: {std_length:.1f}',
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax.legend()
        ax.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig


def plot_ngram_frequency(texts: List[str], n: int = 2, top_n: int = 20,
                        title: str = 'Frecuencia de N-gramas',
                        figsize: Tuple[int, int] = (14, 8),
                        interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Visualiza la frecuencia de n-gramas en los textos.
    
    Args:
        texts: Lista de textos
        n: Tamaño del n-grama (1 para unigramas, 2 para bigramas, etc.)
        top_n: Número de n-gramas más frecuentes a mostrar
        title: Título del gráfico
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    # Preprocesar textos
    processed_texts = []
    for text in texts:
        # Convertir a minúsculas y eliminar caracteres especiales
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Dividir en palabras
        words = text.split()
        processed_texts.append(words)
    
    # Extraer n-gramas
    all_ngrams = []
    for words in processed_texts:
        all_ngrams.extend(list(ngrams(words, n)))
    
    # Contar frecuencia
    ngram_freq = Counter(all_ngrams)
    
    # Obtener los n-gramas más frecuentes
    top_ngrams = ngram_freq.most_common(top_n)
    labels = [' '.join(ngram) for ngram, _ in top_ngrams]
    values = [freq for _, freq in top_ngrams]
    
    if interactive:
        # Versión interactiva con Plotly
        fig = px.bar(
            x=labels, y=values,
            labels={'x': f'{n}-gramas', 'y': 'Frecuencia'},
            title=title,
            height=figsize[1]*80,
            width=figsize[0]*80
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            template='plotly_white'
        )
        
        return fig
    else:
        # Versión estática con Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear gráfico de barras
        ax.bar(labels, values)
        
        # Configurar gráfico
        ax.set_title(title)
        ax.set_xlabel(f'{n}-gramas')
        ax.set_ylabel('Frecuencia')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig


def plot_network_analysis(label_lists: List[List[str]], min_weight: int = 2,
                         title: str = 'Red de Co-ocurrencia de Etiquetas',
                         figsize: Tuple[int, int] = (12, 12),
                         interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Visualiza la red de co-ocurrencia de etiquetas.
    
    Args:
        label_lists: Lista de listas de etiquetas
        min_weight: Peso mínimo para mostrar una conexión
        title: Título del gráfico
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    # Crear grafo
    G = nx.Graph()
    
    # Añadir nodos y aristas
    for labels in label_lists:
        # Añadir nodos
        for label in labels:
            if label not in G.nodes():
                G.add_node(label)
        
        # Añadir aristas para cada par de etiquetas co-ocurrentes
        for i, label1 in enumerate(labels):
            for label2 in labels[i+1:]:
                if G.has_edge(label1, label2):
                    G[label1][label2]['weight'] += 1
                else:
                    G.add_edge(label1, label2, weight=1)
    
    # Filtrar aristas por peso mínimo
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < min_weight]
    G.remove_edges_from(edges_to_remove)
    
    # Eliminar nodos aislados
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    if len(G.nodes()) == 0:
        print("No hay suficientes co-ocurrencias para visualizar la red.")
        return None
    
    # Calcular layout
    pos = nx.spring_layout(G, seed=42)
    
    # Obtener pesos de aristas
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [w / max_weight for w in edge_weights]
    
    # Calcular centralidad de los nodos
    centrality = nx.degree_centrality(G)
    node_sizes = [centrality[node] * 1000 + 100 for node in G.nodes()]
    
    if interactive:
        # Versión interactiva con Plotly
        edge_x = []
        edge_y = []
        edge_trace = []
        
        # Crear trazas para aristas
        for i, (u, v) in enumerate(G.edges()):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = G[u][v]['weight']
            width = 1 + 4 * normalized_weights[i]
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=width, color='rgba(150,150,150,0.7)'),
                hoverinfo='text',
                text=f"{u} - {v}: {weight}",
                mode='lines'
            ))
        
        # Crear traza para nodos
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_sizes,
                color=list(centrality.values()),
                colorbar=dict(
                    thickness=15,
                    title='Centralidad',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            ),
            hoverinfo='text',
            hovertext=[f"{node}: {centrality[node]:.2f}" for node in G.nodes()]
        )
        
        # Crear figura
        fig = go.Figure(data=edge_trace + [node_trace])
        
        # Configurar diseño
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=figsize[0]*80,
            width=figsize[1]*80
        )
        
        return fig
    else:
        # Versión estática con Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Dibujar aristas
        nx.draw_networkx_edges(
            G, pos, alpha=0.7,
            width=[1 + 4 * w for w in normalized_weights],
            edge_color='gray'
        )
        
        # Dibujar nodos
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=list(centrality.values()),
            cmap=plt.cm.YlGnBu,
            alpha=0.8
        )
        
        # Añadir etiquetas
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_family='sans-serif'
        )
        
        # Configurar gráfico
        ax.set_title(title)
        ax.axis('off')
        
        # Añadir barra de color para centralidad
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlGnBu, norm=plt.Normalize(vmin=0, vmax=max(centrality.values())))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Centralidad')
        
        plt.tight_layout()
        
        return fig


def plot_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                       categories: List[str] = None,
                       title: str = 'Análisis de Errores por Categoría',
                       figsize: Tuple[int, int] = (14, 8),
                       interactive: bool = False) -> Union[plt.Figure, go.Figure]:
    """
    Visualiza el análisis de errores por categoría.
    
    Args:
        y_true: Etiquetas verdaderas (one-hot encoding)
        y_pred: Etiquetas predichas (one-hot encoding)
        categories: Nombres de las categorías
        title: Título del gráfico
        figsize: Tamaño de la figura
        interactive: Si es True, genera una visualización interactiva con Plotly
        
    Returns:
        Figura de matplotlib o figura de Plotly (si interactive=True)
    """
    if categories is None:
        categories = [f"Categoría {i}" for i in range(y_true.shape[1])]
    
    # Calcular falsos positivos y falsos negativos por categoría
    false_positives = []
    false_negatives = []
    
    for i in range(y_true.shape[1]):
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        false_positives.append(fp)
        false_negatives.append(fn)
    
    if interactive:
        # Versión interactiva con Plotly
        fig = go.Figure()
        
        # Añadir barras para falsos positivos
        fig.add_trace(go.Bar(
            x=categories,
            y=false_positives,
            name='Falsos Positivos',
            marker_color='indianred'
        ))
        
        # Añadir barras para falsos negativos
        fig.add_trace(go.Bar(
            x=categories,
            y=false_negatives,
            name='Falsos Negativos',
            marker_color='royalblue'
        ))
        
        # Configurar diseño
        fig.update_layout(
            title=title,
            xaxis_title='Categoría',
            yaxis_title='Número de Errores',
            barmode='group',
            template='plotly_white',
            height=figsize[1]*80,
            width=figsize[0]*80
        )
        
        # Rotar etiquetas si hay muchas categorías
        if len(categories) > 10:
            fig.update_layout(xaxis_tickangle=-45)
        
        return fig
    else:
        # Versión estática con Matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        
        # Configurar posiciones de barras
        x = np.arange(len(categories))
        width = 0.35
        
        # Crear barras
        ax.bar(x - width/2, false_positives, width, label='Falsos Positivos', color='indianred')
        ax.bar(x + width/2, false_negatives, width, label='Falsos Negativos', color='royalblue')
        
        # Configurar gráfico
        ax.set_title(title)
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Número de Errores')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig


def save_figures(figures: Dict[str, plt.Figure], output_dir: str = 'reports/figures'):
    """
    Guarda un conjunto de figuras en el directorio especificado.
    
    Args:
        figures: Diccionario de figuras {nombre: figura}
        output_dir: Directorio de salida
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
        results_path: Ruta al archivo pickle con los resultados.
        
    Returns:
        Dict con los resultados cargados.
    """
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """
    Guarda resultados de evaluación en un archivo pickle.
    
    Args:
        results: Diccionario con resultados
        output_path: Ruta de salida para el archivo pickle
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar resultados
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)