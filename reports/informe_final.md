# Informe Final: Clasificación de Literatura Médica

## Resumen Ejecutivo

Este informe presenta los resultados del proyecto de clasificación multi-etiqueta de literatura médica desarrollado para el Tech Sphere Challenge. El objetivo principal fue desarrollar un sistema capaz de clasificar textos médicos (títulos y resúmenes) en una o más categorías: Cardiovascular, Hepatorenal, Neurológica y Oncológica.

## Análisis del Dataset

El dataset proporcionado contiene 3,565 registros de literatura médica, cada uno con un título, un resumen (abstract) y una o más etiquetas de categoría. Las principales características del dataset son:

- **Distribución de categorías**:
  - Neurológica: 1,785 registros (50.1%)
  - Cardiovascular: 1,268 registros (35.6%)
  - Hepatorenal: 1,091 registros (30.6%)
  - Oncológica: 601 registros (16.9%)

- **Características de los textos**:
  - **Títulos**: Longitud media de 69.3 caracteres (mín: 20, máx: 294)
  - **Resúmenes**: Longitud media de 696.5 caracteres (mín: 180, máx: 3,814)

- **Multi-etiqueta**: Aproximadamente el 30% de los registros pertenecen a más de una categoría, lo que confirma la naturaleza multi-etiqueta del problema.

## Metodología

### Preprocesamiento de Textos

Se implementó un pipeline de preprocesamiento que incluye:

1. Limpieza de texto (eliminación de caracteres especiales, normalización)
2. Tokenización
3. Eliminación de stopwords
4. Lematización/stemming para reducir palabras a su forma base

### Extracción de Características

Se exploraron varias técnicas de extracción de características:

1. **Bag of Words (BoW)**: Representación simple basada en frecuencia de palabras
2. **TF-IDF**: Ponderación de términos según su importancia relativa
3. **Word Embeddings**: Representaciones vectoriales densas de palabras
4. **Características estadísticas**: Longitud de textos, densidad léxica, etc.

### Modelos de Clasificación

Se implementaron y evaluaron varios modelos de clasificación multi-etiqueta:

1. **Regresión Logística**: Como línea base
2. **Support Vector Machines (SVM)**: Para capturar relaciones no lineales
3. **Random Forest**: Para modelar interacciones complejas entre características
4. **Gradient Boosting**: Para mejorar el rendimiento mediante boosting
5. **Multi-layer Perceptron (MLP)**: Como aproximación a redes neuronales

Todos los modelos se implementaron con la estrategia OneVsRest para manejar la clasificación multi-etiqueta.

## Resultados y Evaluación

Los modelos se evaluaron utilizando métricas específicas para clasificación multi-etiqueta:

- **F1-score ponderado**: Métrica principal según los requisitos del challenge
- **Precisión y recall**: Para entender el balance entre falsos positivos y falsos negativos
- **Exactitud (accuracy)**: Como métrica complementaria
- **Pérdida de Hamming**: Para medir la fracción de etiquetas incorrectas

### Desafíos Encontrados

1. **Desbalance de clases**: La categoría Oncológica está subrepresentada (16.9%)
2. **Superposición de categorías**: Existen patrones complejos de co-ocurrencia entre categorías
3. **Variabilidad en la longitud de textos**: Gran variación en la longitud de los resúmenes

## Conclusiones y Recomendaciones

### Conclusiones

1. El problema de clasificación multi-etiqueta de literatura médica es complejo debido a la superposición de categorías y el desbalance de clases.
2. Las técnicas de TF-IDF combinadas con modelos como SVM o Gradient Boosting ofrecen un buen balance entre rendimiento y complejidad.
3. La incorporación de características específicas del dominio médico podría mejorar significativamente el rendimiento.

### Recomendaciones

1. **Mejora del preprocesamiento**: Incorporar diccionarios médicos específicos y técnicas de normalización adaptadas al lenguaje médico.
2. **Exploración de modelos avanzados**: Considerar modelos basados en transformers como BERT o BioBERT, específicamente pre-entrenados en textos médicos.
3. **Técnicas de data augmentation**: Para abordar el desbalance de clases, especialmente en la categoría Oncológica.
4. **Validación cruzada estratificada**: Para asegurar una evaluación robusta considerando la naturaleza multi-etiqueta.

## Próximos Pasos

1. Implementar modelos basados en transformers para comparar su rendimiento con los modelos tradicionales.
2. Explorar técnicas de explicabilidad para entender mejor las decisiones de los modelos.
3. Desarrollar una interfaz de usuario para facilitar la clasificación de nuevos textos médicos.

---

*Este proyecto fue desarrollado como parte del Tech Sphere Challenge para la clasificación multi-etiqueta de literatura médica.*