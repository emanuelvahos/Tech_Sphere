# Documentación de Scripts de Demostración

Este documento describe los diferentes scripts de demostración disponibles en el proyecto Tech_Sphere para evaluar y visualizar los modelos de clasificación de literatura médica.

## Scripts Disponibles

### 1. demo_modelo.py

Script básico para demostrar la funcionalidad principal del modelo de clasificación.

```bash
python demo_modelo.py
```

**Características:**
- Carga un modelo pre-entrenado
- Permite clasificar textos de ejemplo
- Muestra las categorías predichas y sus probabilidades

### 2. demo_basico.py

Script de demostración que incluye evaluación y visualizaciones básicas sin requerir dependencias externas complejas.

```bash
python demo_basico.py
```

**Características:**
- Carga un modelo existente o entrena uno simple si no hay ninguno disponible
- Evalúa el modelo con métricas estándar (accuracy, precision, recall, F1-score, hamming loss)
- Genera visualizaciones básicas como matrices de confusión y gráficos de radar de métricas
- Guarda las visualizaciones en el directorio `reports/figures/`

### 3. demo_avanzado.py

Script de demostración avanzado que incluye todas las funcionalidades de visualización y evaluación implementadas.

```bash
python demo_avanzado.py
```

**Características:**
- Carga un modelo existente o entrena uno simple si no hay ninguno disponible
- Demuestra todas las visualizaciones avanzadas implementadas
- Realiza un análisis completo del modelo utilizando técnicas avanzadas de evaluación
- Genera visualizaciones interactivas en formato HTML en el directorio `reports/figures/`

### 4. probar_modelos.py

Script para probar y comparar diferentes modelos entrenados.

```bash
python probar_modelos.py
```

**Características:**
- Carga y prueba múltiples modelos entrenados
- Compara el rendimiento de diferentes algoritmos
- Muestra métricas comparativas

## Aplicaciones Web de Demostración

### 1. app_demo.py

Aplicación web en español para demostrar el uso de los modelos de clasificación.

```bash
python app_demo.py
```

**Características:**
- Interfaz web intuitiva
- Permite seleccionar diferentes modelos
- Clasificación en tiempo real de textos médicos
- Visualización de resultados y probabilidades

### 2. app_demo_en.py

Aplicación web en inglés para demostrar el uso de los modelos de clasificación.

```bash
python app_demo_en.py
```

**Características:**
- Mismas funcionalidades que app_demo.py pero en inglés
- Interfaz adaptada para usuarios de habla inglesa

## Recomendaciones de Uso

- Para una demostración rápida y sin dependencias externas, utilice `demo_basico.py`
- Para explorar todas las capacidades avanzadas de visualización y evaluación, utilice `demo_avanzado.py`
- Para una experiencia interactiva con interfaz gráfica, utilice `app_demo.py` o `app_demo_en.py`
- Para comparar diferentes modelos, utilice `probar_modelos.py`

## Requisitos

Los scripts básicos (`demo_modelo.py` y `demo_basico.py`) están diseñados para funcionar con dependencias mínimas:

- Python 3.6+
- scikit-learn
- pandas
- numpy
- matplotlib

Los scripts avanzados (`demo_avanzado.py`) pueden requerir dependencias adicionales:

- plotly
- gensim
- nltk

Las aplicaciones web (`app_demo.py` y `app_demo_en.py`) requieren:

- Flask
- Todas las dependencias de los scripts básicos