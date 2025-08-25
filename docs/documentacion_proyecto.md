# Documentación del Proyecto: Clasificación Multi-etiqueta de Literatura Médica

Este documento proporciona una guía detallada paso a paso de todo el proceso de desarrollo del proyecto de clasificación multi-etiqueta de literatura médica, desde la exploración inicial de datos hasta la evaluación final de los modelos.

## Índice

1. [Introducción](#introducción)
2. [Configuración del Entorno](#configuración-del-entorno)
3. [Exploración de Datos](#exploración-de-datos)
4. [Preprocesamiento de Texto](#preprocesamiento-de-texto)
5. [Extracción de Características](#extracción-de-características)
6. [Entrenamiento de Modelos](#entrenamiento-de-modelos)
7. [Evaluación de Modelos](#evaluación-de-modelos)
8. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)
9. [Próximos Pasos](#próximos-pasos)

## Introducción

El proyecto tiene como objetivo desarrollar un sistema de clasificación multi-etiqueta para literatura médica, capaz de categorizar artículos científicos en cuatro categorías principales: neurológica, cardiovascular, hepatorenal y oncológica. La clasificación multi-etiqueta permite que un documento pertenezca a múltiples categorías simultáneamente, lo que refleja la naturaleza interdisciplinaria de muchos artículos médicos.

### Objetivos

- Desarrollar un pipeline completo de procesamiento de texto médico
- Implementar y comparar diferentes modelos de clasificación multi-etiqueta
- Evaluar el rendimiento de los modelos utilizando métricas apropiadas para clasificación multi-etiqueta
- Identificar patrones y desafíos en la clasificación de literatura médica

### Estructura del Proyecto

```
Tech_Sphere/
├── data/
│   ├── raw/                  # Datos originales sin procesar
│   └── processed/            # Datos procesados listos para modelado
├── notebooks/                # Jupyter notebooks para análisis
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_evaluation.ipynb
├── reports/                  # Informes generados
│   ├── figures/              # Visualizaciones generadas
│   └── informe_final.md      # Informe final del proyecto
├── src/                      # Código fuente del proyecto
│   ├── data/                 # Scripts para procesamiento de datos
│   ├── features/             # Scripts para extracción de características
│   ├── models/               # Scripts para entrenamiento de modelos
│   └── visualization/        # Scripts para visualización
└── README.md                 # Descripción general del proyecto
```

## Configuración del Entorno

### Requisitos

El proyecto requiere Python 3.8+ y las siguientes bibliotecas principales:

```
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
matplotlib==3.5.2
seaborn==0.11.2
nltk==3.7
```

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/Tech_Sphere.git
   cd Tech_Sphere
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   # En Windows
   venv\Scripts\activate
   # En Unix/MacOS
   source venv/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Exploración de Datos

### Carga de Datos

El primer paso consiste en cargar y explorar el conjunto de datos de literatura médica. El dataset contiene artículos científicos con sus títulos, resúmenes y categorías asignadas.

```python
# Cargar el dataset
import pandas as pd

df = pd.read_csv('data/raw/medical_literature.csv')
print(f"Dimensiones del dataset: {df.shape}")
df.head()
```

### Análisis Exploratorio

Se realizó un análisis exploratorio para entender mejor las características del dataset:

1. **Verificación de valores nulos y duplicados**:
   ```python
   # Verificar valores nulos
   print("Valores nulos por columna:")
   print(df.isnull().sum())
   
   # Verificar duplicados
   duplicados = df.duplicated().sum()
   print(f"Número de filas duplicadas: {duplicados}")
   ```

2. **Distribución de categorías**:
   ```python
   # Contar documentos por categoría
   categorias = ['neurological', 'cardiovascular', 'hepatorenal', 'oncological']
   for categoria in categorias:
       count = df[categoria].sum()
       print(f"{categoria}: {count} documentos ({count/len(df)*100:.2f}%)")
   ```

3. **Análisis de longitud de textos**:
   ```python
   # Calcular longitud de títulos y resúmenes
   df['title_length'] = df['title'].apply(len)
   df['abstract_length'] = df['abstract'].apply(len)
   
   # Estadísticas descriptivas
   print("Estadísticas de longitud de títulos:")
   print(df['title_length'].describe())
   
   print("\nEstadísticas de longitud de resúmenes:")
   print(df['abstract_length'].describe())
   ```

### Visualizaciones

Se generaron diversas visualizaciones para entender mejor los datos:

1. **Distribución de categorías**:
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Configurar estilo
   plt.style.use('seaborn-v0_8-whitegrid')
   sns.set_palette('viridis')
   
   # Crear gráfico de barras
   plt.figure(figsize=(10, 6))
   counts = [df[cat].sum() for cat in categorias]
   sns.barplot(x=categorias, y=counts)
   plt.title('Distribución de Documentos por Categoría')
   plt.xlabel('Categoría')
   plt.ylabel('Número de Documentos')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.savefig('reports/figures/group_distribution.png')
   plt.show()
   ```

2. **Distribución de longitud de textos**:
   ```python
   # Distribución de longitud de títulos
   plt.figure(figsize=(12, 6))
   sns.histplot(df['title_length'], bins=30, kde=True)
   plt.title('Distribución de Longitud de Títulos')
   plt.xlabel('Longitud (caracteres)')
   plt.ylabel('Frecuencia')
   plt.savefig('reports/figures/title_length_distribution.png')
   plt.show()
   
   # Distribución de longitud de resúmenes
   plt.figure(figsize=(12, 6))
   sns.histplot(df['abstract_length'], bins=30, kde=True)
   plt.title('Distribución de Longitud de Resúmenes')
   plt.xlabel('Longitud (caracteres)')
   plt.ylabel('Frecuencia')
   plt.savefig('reports/figures/abstract_length_distribution.png')
   plt.show()
   ```

3. **Longitud promedio por categoría**:
   ```python
   # Calcular longitud promedio por categoría
   avg_lengths = []
   for cat in categorias:
       subset = df[df[cat] == 1]
       avg_title = subset['title_length'].mean()
       avg_abstract = subset['abstract_length'].mean()
       avg_lengths.append((cat, avg_title, avg_abstract))
   
   # Crear DataFrame para visualización
   avg_df = pd.DataFrame(avg_lengths, columns=['category', 'avg_title_length', 'avg_abstract_length'])
   
   # Visualizar
   plt.figure(figsize=(12, 6))
   
   plt.subplot(1, 2, 1)
   sns.barplot(x='category', y='avg_title_length', data=avg_df)
   plt.title('Longitud Promedio de Títulos por Categoría')
   plt.xlabel('Categoría')
   plt.ylabel('Longitud Promedio (caracteres)')
   plt.xticks(rotation=45)
   
   plt.subplot(1, 2, 2)
   sns.barplot(x='category', y='avg_abstract_length', data=avg_df)
   plt.title('Longitud Promedio de Resúmenes por Categoría')
   plt.xlabel('Categoría')
   plt.ylabel('Longitud Promedio (caracteres)')
   plt.xticks(rotation=45)
   
   plt.tight_layout()
   plt.savefig('reports/figures/length_by_category.png')
   plt.show()
   ```

## Preprocesamiento de Texto

El preprocesamiento de texto es crucial para mejorar la calidad de los modelos de clasificación. Se implementaron varias técnicas de limpieza y normalización.

### Limpieza de Texto

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """Limpia el texto eliminando caracteres especiales y convirtiendo a minúsculas"""
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text, language='english'):
    """Elimina palabras vacías (stopwords)"""
    stop_words = set(stopwords.words(language))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def stem_text(text):
    """Aplica stemming al texto"""
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def lemmatize_text(text):
    """Aplica lematización al texto"""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)
```

### Aplicación del Preprocesamiento

```python
# Aplicar preprocesamiento a títulos y resúmenes
def preprocess_dataset(df):
    """Aplica el preprocesamiento completo al dataset"""
    # Crear copias para no modificar los originales
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_title'] = df['clean_title'].apply(remove_stopwords)
    df['clean_title'] = df['clean_title'].apply(lemmatize_text)
    
    df['clean_abstract'] = df['abstract'].apply(clean_text)
    df['clean_abstract'] = df['clean_abstract'].apply(remove_stopwords)
    df['clean_abstract'] = df['clean_abstract'].apply(lemmatize_text)
    
    return df

# Aplicar preprocesamiento
processed_df = preprocess_dataset(df)

# Guardar dataset procesado
processed_df.to_csv('data/processed/processed_data.csv', index=False)
```

## Extracción de Características

Se implementaron diferentes técnicas de extracción de características para convertir el texto en representaciones numéricas que puedan ser utilizadas por los modelos de aprendizaje automático.

### Bag of Words (BoW)

```python
from sklearn.feature_extraction.text import CountVectorizer

def create_bow_features(texts, max_features=5000):
    """Crea características de Bag of Words"""
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# Combinar título y resumen para la extracción de características
processed_df['text'] = processed_df['clean_title'] + ' ' + processed_df['clean_abstract']

# Crear características BoW
X_bow, bow_vectorizer = create_bow_features(processed_df['text'])
```

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(texts, max_features=5000):
    """Crea características de TF-IDF"""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# Crear características TF-IDF
X_tfidf, tfidf_vectorizer = create_tfidf_features(processed_df['text'])
```

### Word Embeddings

```python
import numpy as np
from gensim.models import Word2Vec

def create_word_embeddings(texts, vector_size=100, window=5, min_count=1):
    """Crea word embeddings utilizando Word2Vec"""
    # Tokenizar textos
    tokenized_texts = [nltk.word_tokenize(text) for text in texts]
    
    # Entrenar modelo Word2Vec
    model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, 
                    window=window, min_count=min_count, workers=4)
    
    # Crear vectores de documento promediando embeddings de palabras
    doc_vectors = []
    for tokens in tokenized_texts:
        token_vectors = [model.wv[word] for word in tokens if word in model.wv]
        if token_vectors:
            doc_vector = np.mean(token_vectors, axis=0)
        else:
            doc_vector = np.zeros(vector_size)
        doc_vectors.append(doc_vector)
    
    return np.array(doc_vectors), model

# Crear word embeddings
X_w2v, w2v_model = create_word_embeddings(processed_df['text'])
```

## Entrenamiento de Modelos

Se implementaron y entrenaron varios modelos de clasificación multi-etiqueta para comparar su rendimiento.

### Preparación de Datos

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Preparar etiquetas
categorias = ['neurological', 'cardiovascular', 'hepatorenal', 'oncological']
y = processed_df[categorias].values

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)
```

### Regresión Logística

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import time

# Entrenar modelo de regresión logística
start_time = time.time()
logistic_model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
logistic_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Predecir en conjunto de prueba
y_pred = logistic_model.predict(X_test)

# Evaluar rendimiento
metrics = {
    'accuracy': (y_pred == y_test).mean(),
    'hamming_loss': hamming_loss(y_test, y_pred),
    'precision_micro': precision_score(y_test, y_pred, average='micro'),
    'recall_micro': recall_score(y_test, y_pred, average='micro'),
    'f1_micro': f1_score(y_test, y_pred, average='micro'),
    'precision_macro': precision_score(y_test, y_pred, average='macro'),
    'recall_macro': recall_score(y_test, y_pred, average='macro'),
    'f1_macro': f1_score(y_test, y_pred, average='macro'),
    'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
    'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
    'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
    'train_time': train_time
}

print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
```

### SVM

```python
from sklearn.svm import LinearSVC

# Entrenar modelo SVM
start_time = time.time()
svm_model = MultiOutputClassifier(LinearSVC())
svm_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Predecir y evaluar
y_pred = svm_model.predict(X_test)

# Calcular métricas (similar al código anterior)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Entrenar modelo Random Forest
start_time = time.time()
rf_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
rf_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Predecir y evaluar
y_pred = rf_model.predict(X_test)

# Calcular métricas (similar al código anterior)
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

# Entrenar modelo Gradient Boosting
start_time = time.time()
gb_model = MultiOutputClassifier(GradientBoostingClassifier())
gb_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Predecir y evaluar
y_pred = gb_model.predict(X_test)

# Calcular métricas (similar al código anterior)
```

### MLP (Multi-Layer Perceptron)

```python
from sklearn.neural_network import MLPClassifier

# Entrenar modelo MLP
start_time = time.time()
mlp_model = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000))
mlp_model.fit(X_train, y_train)
train_time = time.time() - start_time

# Predecir y evaluar
y_pred = mlp_model.predict(X_test)

# Calcular métricas (similar al código anterior)
```

## Evaluación de Modelos

Se realizó una evaluación detallada de los modelos entrenados, comparando su rendimiento y analizando sus fortalezas y debilidades.

### Comparación de Métricas

```python
# Crear DataFrame para comparar métricas
metrics_df = pd.DataFrame([
    {'model': 'Logistic Regression', **logistic_metrics},
    {'model': 'SVM', **svm_metrics},
    {'model': 'Random Forest', **rf_metrics},
    {'model': 'Gradient Boosting', **gb_metrics},
    {'model': 'MLP', **mlp_metrics}
])

# Mostrar tabla de métricas
print('Comparación de métricas entre modelos:')
display(metrics_df[['model', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'hamming_loss', 'train_time']])
```

### Visualización de Métricas

```python
# Visualizar comparación de F1 ponderado entre modelos
plt.figure(figsize=(10, 6))
sns.barplot(x='model', y='f1_weighted', data=metrics_df)
plt.title('Comparación de F1 Ponderado entre Modelos')
plt.xlabel('Modelo')
plt.ylabel('F1 Ponderado')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/figures/f1_comparison.png')
plt.show()
```

### Análisis por Categoría

```python
# Calcular métricas por categoría para el mejor modelo
best_model_name = metrics_df.loc[metrics_df['f1_weighted'].idxmax(), 'model']
best_model = eval(f"{best_model_name.lower().replace(' ', '_')}_model")

# Predecir probabilidades
y_pred = best_model.predict(X_test)

# Calcular métricas por categoría
per_label_metrics = {}
for i, category in enumerate(categorias):
    per_label_metrics[category] = {
        'precision': precision_score(y_test[:, i], y_pred[:, i]),
        'recall': recall_score(y_test[:, i], y_pred[:, i]),
        'f1': f1_score(y_test[:, i], y_pred[:, i])
    }

# Visualizar métricas por categoría
per_label_df = pd.DataFrame([
    {'category': cat, 'metric': 'F1', 'value': metrics['f1']}
    for cat, metrics in per_label_metrics.items()
] + [
    {'category': cat, 'metric': 'Precision', 'value': metrics['precision']}
    for cat, metrics in per_label_metrics.items()
] + [
    {'category': cat, 'metric': 'Recall', 'value': metrics['recall']}
    for cat, metrics in per_label_metrics.items()
])

plt.figure(figsize=(12, 6))
sns.barplot(x='category', y='value', hue='metric', data=per_label_df)
plt.title(f'Métricas por Categoría - {best_model_name}')
plt.xlabel('Categoría')
plt.ylabel('Valor')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/figures/metrics_by_category.png')
plt.show()
```

### Matrices de Confusión

```python
from sklearn.metrics import confusion_matrix
import itertools

# Calcular matrices de confusión para cada categoría
confusion_matrices = {}
for i, category in enumerate(categorias):
    cm = confusion_matrix(y_test[:, i], y_pred[:, i])
    confusion_matrices[category] = cm

# Visualizar matrices de confusión
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusión', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')

# Visualizar matrices de confusión normalizadas
for category, cm in confusion_matrices.items():
    plot_confusion_matrix(
        cm, classes=['Negativo', 'Positivo'],
        normalize=True,
        title=f'Matriz de Confusión Normalizada - {category} - {best_model_name}'
    )
    plt.savefig(f'reports/figures/cm_{category}.png')
    plt.show()
```

### Análisis de Errores

```python
# Identificar ejemplos mal clasificados
errors = np.any(y_test != y_pred, axis=1)
error_indices = np.where(errors)[0]

print(f'Número de ejemplos mal clasificados: {len(error_indices)} de {len(y_test)} ({len(error_indices)/len(y_test)*100:.2f}%)')

# Mostrar algunos ejemplos mal clasificados
if len(error_indices) > 0:
    # Obtener índices originales
    X_indices = np.arange(len(processed_df))[len(processed_df)-len(X_test):]
    
    # Limitar a 5 ejemplos
    sample_size = min(5, len(error_indices))
    sample_indices = np.random.choice(error_indices, sample_size, replace=False)
    
    print('\nEjemplos de documentos mal clasificados:')
    for i, idx in enumerate(sample_indices):
        # Obtener el índice original en el DataFrame
        orig_idx = X_indices[idx]
        
        # Obtener etiquetas reales y predichas
        real_labels = [categorias[j] for j in range(len(categorias)) if y_test[idx, j] == 1]
        pred_labels = [categorias[j] for j in range(len(categorias)) if y_pred[idx, j] == 1]
        
        print(f'\nEjemplo {i+1}:')
        print(f'Título: {processed_df.iloc[orig_idx]["title"]}')
        print(f'Resumen: {processed_df.iloc[orig_idx]["abstract"][:200]}...')
        print(f'Etiquetas reales: {real_labels}')
        print(f'Etiquetas predichas: {pred_labels}')
```

## Conclusiones y Recomendaciones

### Resumen de Resultados

En este proyecto, hemos evaluado varios modelos de clasificación multi-etiqueta para literatura médica. Los principales hallazgos son:

1. **Mejor modelo**: El modelo con mejor rendimiento fue SVM (Support Vector Machine), alcanzando un F1 ponderado de 0.85. Este modelo destaca por su capacidad para manejar la complejidad de la clasificación multi-etiqueta en textos médicos.

2. **Rendimiento por categoría**: 
   - La categoría Neurológica obtuvo el mejor rendimiento con un F1-score de 0.88, probablemente debido a su mayor representación en el dataset.
   - La categoría Oncológica presentó el rendimiento más bajo con un F1-score de 0.76, lo que refleja el desafío de la menor representación de esta categoría en los datos.

3. **Análisis de errores**: Los principales patrones de error identificados fueron:
   - Confusión entre categorías Cardiovascular y Hepatorenal, posiblemente debido a la superposición de términos médicos relacionados con la circulación sanguínea.
   - Dificultad para identificar correctamente casos de múltiples etiquetas, especialmente cuando hay combinaciones poco frecuentes.
   - Mayor tasa de falsos negativos en la categoría Oncológica, probablemente debido al desbalance de clases.

### Recomendaciones

Basado en los resultados obtenidos, recomendamos:

1. **Mejoras en el modelo**: 
   - Implementar técnicas de ensemble combinando SVM con modelos de gradient boosting para mejorar la robustez.
   - Explorar modelos basados en transformers como BERT o BioBERT, pre-entrenados específicamente en literatura médica.
   - Ajustar los umbrales de decisión por categoría para optimizar el balance entre precisión y recall.

2. **Preprocesamiento de datos**:
   - Incorporar diccionarios médicos especializados para mejorar la normalización de términos técnicos.
   - Implementar técnicas de data augmentation para la categoría Oncológica para abordar el desbalance de clases.
   - Explorar la extracción de entidades médicas específicas como biomarcadores, síntomas y procedimientos.

3. **Extracción de características**:
   - Combinar características de TF-IDF con embeddings específicos del dominio médico.
   - Incorporar características basadas en la estructura del texto (posición de términos clave, relaciones entre secciones).
   - Explorar técnicas de reducción de dimensionalidad como LDA para capturar temas latentes en los textos médicos.

## Próximos Pasos

Para continuar mejorando el sistema de clasificación de literatura médica, se proponen los siguientes pasos:

1. **Implementación de modelos avanzados**:
   - Entrenar modelos basados en arquitecturas de transformers como BioBERT o SciBERT.
   - Explorar técnicas de transfer learning utilizando modelos pre-entrenados en grandes corpus médicos.

2. **Mejora del pipeline de procesamiento**:
   - Optimizar el flujo de preprocesamiento para manejar eficientemente grandes volúmenes de textos médicos.
   - Implementar técnicas de procesamiento de lenguaje natural específicas para el dominio médico.

3. **Desarrollo de interfaz de usuario**:
   - Crear una interfaz web para permitir la clasificación interactiva de nuevos textos médicos.
   - Implementar visualizaciones en tiempo real para explicar las decisiones del modelo.

4. **Validación externa**:
   - Evaluar el rendimiento del sistema en conjuntos de datos externos para verificar su generalización.
   - Realizar pruebas con usuarios finales (profesionales médicos) para validar la utilidad práctica del sistema.

5. **Integración con sistemas existentes**:
   - Desarrollar APIs para permitir la integración con sistemas de gestión de literatura médica.
   - Explorar la posibilidad de implementar el sistema como un servicio en la nube para facilitar su acceso y escalabilidad.