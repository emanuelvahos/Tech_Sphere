# Scripts de Demostración Refactorizados

Este documento describe los scripts de demostración refactorizados (`demo_basico.py` y `demo_avanzado.py`) y el nuevo módulo común de utilidades (`src/demo/demo_utils.py`) que comparten.

## Descripción

Los scripts de demostración han sido refactorizados para eliminar la duplicación de código y mejorar la mantenibilidad. Ahora ambos scripts utilizan un módulo común de utilidades que contiene las funciones compartidas.

### Módulo Común de Utilidades

El módulo `src/demo/demo_utils.py` contiene funciones compartidas entre los scripts de demostración, incluyendo:

- Carga y guardado de modelos
- Carga y preprocesamiento de datos
- Preparación de conjuntos de entrenamiento y prueba
- Vectorización de textos y transformación de etiquetas
- Entrenamiento de modelos simples
- Evaluación básica de modelos
- Visualización de métricas y resultados

### Demo Básica

El script `demo_basico.py` demuestra las capacidades básicas de visualización y evaluación de modelos, incluyendo:

- Carga o creación de un modelo simple
- Evaluación con métricas básicas (accuracy, precision, recall, F1-score, hamming loss)
- Visualización de matriz de confusión y gráfico de radar de métricas

### Demo Avanzada

El script `demo_avanzado.py` demuestra funcionalidades más avanzadas de visualización y evaluación, incluyendo:

- Carga de modelos existentes o entrenamiento de uno nuevo
- Evaluación con métricas avanzadas
- Visualizaciones más detalladas utilizando las funciones del módulo `src.models.evaluate_model`

## Uso

### Demo Básica

```bash
python demo_basico.py
```

Este script:
1. Carga un modelo existente o crea uno nuevo si no existe
2. Evalúa el modelo con métricas básicas
3. Genera visualizaciones simples (matriz de confusión y gráfico de radar)

### Demo Avanzada

```bash
python demo_avanzado.py
```

Este script:
1. Carga un modelo existente o crea uno nuevo si no existe
2. Evalúa el modelo con métricas avanzadas
3. Genera visualizaciones más detalladas utilizando las funciones del módulo de evaluación

## Ventajas de la Refactorización

- **Eliminación de duplicación**: El código común se ha extraído a un módulo compartido
- **Mayor mantenibilidad**: Los cambios en las funciones compartidas solo necesitan hacerse en un lugar
- **Mejor organización**: Separación clara entre funcionalidad básica y avanzada
- **Facilidad de extensión**: Nuevos scripts de demostración pueden reutilizar las funciones existentes

## Uso Programático

El módulo de utilidades también puede importarse y utilizarse en otros scripts:

```python
from src.demo.demo_utils import load_model, evaluate_model_basic

# Cargar un modelo
model, vectorizer, mlb = load_model('models/mi_modelo.pkl')

# Evaluar el modelo
y_pred, metrics = evaluate_model_basic(model, X_test, y_test)
```

## Estructura de Archivos

```
├── demo_basico.py           # Script de demostración básica
├── demo_avanzado.py         # Script de demostración avanzada
├── src/
│   ├── demo/
│   │   └── demo_utils.py    # Módulo común de utilidades
│   ├── visualization/
│   │   └── visualize.py     # Funciones de visualización
│   └── models/
│       └── evaluate_model.py # Funciones de evaluación avanzada
└── models/                  # Directorio para modelos guardados
```