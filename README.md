# Clasificación de Literatura Médica

## Descripción del Proyecto
Este proyecto implementa un sistema de clasificación multi-etiqueta para literatura médica, capaz de categorizar artículos científicos en diferentes dominios médicos basándose en su título y resumen. El sistema está diseñado para el AI + Data Challenge, con el objetivo de clasificar textos médicos en una o más categorías de forma precisa y eficiente.

## Estructura del Proyecto
```
Tech_Sphere/
├── data/
│   ├── raw/            # Datos originales sin procesar
│   └── processed/      # Datos procesados listos para modelado
├── notebooks/          # Jupyter notebooks para análisis exploratorio
│   ├── 01_exploratory_data_analysis.ipynb  # Análisis exploratorio inicial
│   └── 02_model_evaluation.ipynb           # Evaluación detallada de modelos
├── src/                # Código fuente del proyecto
│   ├── data/           # Scripts para procesamiento de datos
│   │   └── preprocess.py  # Funciones de preprocesamiento de texto
│   ├── features/       # Scripts para ingeniería de características
│   │   └── feature_extraction.py  # Extracción de características de texto
│   ├── models/         # Scripts para entrenamiento y evaluación de modelos
│   │   └── train_model.py  # Entrenamiento y evaluación de modelos
│   ├── visualization/  # Scripts para visualización de resultados
│   │   └── visualize.py  # Funciones de visualización
│   └── main.py         # Script principal para ejecutar el pipeline completo
├── models/             # Modelos entrenados
├── reports/            # Reportes generados
│   ├── figures/        # Figuras y visualizaciones
│   └── informe_final.md # Informe final del proyecto
└── requirements.txt    # Dependencias del proyecto
```

## Objetivos
- Desarrollar un sistema de clasificación multi-etiqueta para literatura médica
- Implementar y comparar diferentes algoritmos de aprendizaje automático
- Evaluar el rendimiento del sistema utilizando métricas adecuadas para clasificación multi-etiqueta
- Optimizar el rendimiento del modelo para maximizar el F1-score ponderado

## Categorías de Clasificación
- Cardiovascular: Relacionado con el corazón y el sistema circulatorio
- Hepatorenal: Relacionado con el hígado y los riñones
- Neurológico: Relacionado con el sistema nervioso
- Oncológico: Relacionado con el cáncer y tumores

## Dataset
El dataset contiene información de literatura médica con las siguientes columnas:
- Título: Título del artículo científico
- Resumen (Abstract): Resumen del contenido del artículo
- Grupo: Categoría(s) a la(s) que pertenece el artículo (separadas por '|')

El conjunto de datos contiene 3,565 registros y presenta un desafío de clasificación multi-etiqueta, donde cada documento puede pertenecer a una o más categorías médicas.

### Análisis Exploratorio
Del análisis exploratorio realizado, se destacan los siguientes hallazgos:

- **Distribución de categorías**:
  - Neurológica: 1,785 registros (50.1%)
  - Cardiovascular: 1,268 registros (35.6%)
  - Hepatorenal: 1,091 registros (30.6%)
  - Oncológica: 601 registros (16.9%)

- **Características de los textos**:
  - **Títulos**: Longitud media de 69.3 caracteres (mín: 20, máx: 294)
  - **Resúmenes**: Longitud media de 696.5 caracteres (mín: 180, máx: 3,814)

- No se encontraron valores nulos ni duplicados en el dataset.
- Aproximadamente el 30% de los registros pertenecen a más de una categoría.

## Metodología
1. **Preprocesamiento de texto**:
   - Limpieza de texto (eliminación de caracteres especiales, normalización)
   - Tokenización y eliminación de stopwords
   - Lematización/stemming para reducir palabras a su forma base

2. **Extracción de características**:
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Bag of Words (BoW)
   - Embeddings de documentos
   - Características estadísticas de texto

3. **Entrenamiento de modelos**:
   - Regresión Logística
   - SVM (Support Vector Machine)
   - Random Forest
   - Gradient Boosting
   - Redes Neuronales (MLP)

4. **Evaluación y optimización**:
   - Validación cruzada
   - Optimización de hiperparámetros
   - Manejo de desbalance de clases

## Métricas de Evaluación
- **Precisión, Recall y F1-Score** (micro, macro y ponderado)
- **Exactitud** (Accuracy)
- **Pérdida de Hamming** (Hamming Loss)
- **Puntuación de Jaccard** (Jaccard Score)

La métrica principal para evaluar el rendimiento del modelo es el **F1-score ponderado**, que tiene en cuenta el desbalance entre las clases.

## Instrucciones de Uso

### Requisitos
Instalar las dependencias necesarias:
```bash
pip install -r requirements.txt
```

### Ejecución del Análisis Exploratorio
Para ejecutar el análisis exploratorio simplificado:

```bash
python src/main.py
```

Esto generará visualizaciones básicas en la carpeta `reports/figures/`.

### Ejecución del Pipeline Completo (Versión Original)
Para ejecutar todo el pipeline de procesamiento, desde la carga de datos hasta la visualización de resultados:

```bash
python src/main.py --run-all --model-type logistic --feature-method tfidf
```

### Ejecución por Etapas (Versión Original)
También es posible ejecutar etapas específicas del pipeline:

```bash
# Solo preprocesamiento de datos
python src/main.py --process-data

# Extracción de características
python src/main.py --extract-features --feature-method tfidf

# Entrenamiento y evaluación de modelo
python src/main.py --train-model --model-type svm --feature-method tfidf --optimize

# Generación de visualizaciones
python src/main.py --visualize --model-type svm
```

### Parámetros Disponibles
- `--model-type`: Tipo de modelo a utilizar (logistic, svm, rf, gb, mlp)
- `--feature-method`: Método de extracción de características (tfidf, bow, embeddings, combined)
- `--balance-method`: Método para manejar desbalance de clases (none, smote, undersample)
- `--optimize`: Activar optimización de hiperparámetros

## Resultados
Los resultados detallados del análisis y la evaluación de modelos se encuentran en:
- Notebook de análisis exploratorio: `notebooks/01_exploratory_data_analysis.ipynb`
- Notebook de evaluación de modelos: `notebooks/02_model_evaluation.ipynb`
- Informe final: `reports/informe_final.md`

Las visualizaciones generadas se encuentran en la carpeta `reports/figures/`.

## Autor
Desarrollado para el Tech Sphere Challenge.