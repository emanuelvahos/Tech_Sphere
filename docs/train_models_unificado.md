# Script Unificado de Entrenamiento de Modelos

Este documento describe el uso del script unificado `train_models_unificado.py` que combina las funcionalidades de los scripts anteriores (`train_models.py`, `train_model_demo.py` y `crear_modelos.py`) en una única solución más eficiente y completa.

## Descripción

El script unificado permite entrenar varios tipos de modelos de clasificación de literatura médica con diferentes configuraciones y guardarlos para su uso posterior. Incluye funcionalidades para:

- Preprocesamiento de datos (limpieza, eliminación de stopwords, lematización)
- Extracción de características mediante TF-IDF
- Entrenamiento de múltiples tipos de modelos (SVM, Random Forest, Logistic Regression)
- Evaluación de rendimiento con métricas estándar
- Guardado de modelos con sus componentes asociados

## Uso

El script puede ejecutarse desde la línea de comandos con diferentes opciones:

```bash
python train_models_unificado.py [opciones]
```

### Opciones disponibles

- `--model`: Tipo de modelo a entrenar
  - `all`: Entrena todos los modelos disponibles (por defecto)
  - `svm`: Entrena solo el modelo SVM
  - `rf`: Entrena solo el modelo Random Forest
  - `lr`: Entrena solo el modelo Logistic Regression
  - `demo`: Entrena un modelo básico de demostración (SVM simplificado)

- `--data`: Ruta al archivo de datos CSV (por defecto: `data/raw/challenge_data-18-ago.csv`)

- `--features`: Número máximo de características para TF-IDF (por defecto: 10000)

- `--test-size`: Proporción del conjunto de prueba (por defecto: 0.2)

- `--seed`: Semilla aleatoria para reproducibilidad (por defecto: 42)

- `--quiet`: Modo silencioso (sin mensajes de progreso)

### Ejemplos de uso

1. Entrenar todos los modelos con configuración por defecto:

```bash
python train_models_unificado.py
```

2. Entrenar solo el modelo SVM:

```bash
python train_models_unificado.py --model svm
```

3. Entrenar un modelo de demostración simplificado:

```bash
python train_models_unificado.py --model demo
```

4. Entrenar Random Forest con más características:

```bash
python train_models_unificado.py --model rf --features 15000
```

5. Entrenar Logistic Regression con conjunto de prueba más grande y modo silencioso:

```bash
python train_models_unificado.py --model lr --test-size 0.3 --quiet
```

## Estructura de los modelos guardados

Los modelos se guardan en formato pickle en el directorio `models/` con la siguiente estructura interna:

```python
model_data = {
    'model': model,              # El modelo entrenado
    'vectorizer': vectorizer,    # El vectorizador TF-IDF
    'mlb': mlb,                  # El codificador MultiLabelBinarizer
    'metrics': metrics           # Métricas de evaluación (opcional)
}
```

Esto permite cargar fácilmente todos los componentes necesarios para realizar predicciones con los modelos entrenados.

## Uso programático

El script también puede importarse como módulo y utilizarse programáticamente:

```python
from train_models_unificado import train_single_model, train_all_models, train_demo_model

# Entrenar un modelo específico
model, vectorizer, mlb, metrics = train_single_model(model_type='svm')

# Entrenar todos los modelos
results = train_all_models()

# Entrenar modelo de demostración
model, vectorizer, mlb, metrics = train_demo_model()
```

## Ventajas sobre los scripts anteriores

- **Código más limpio y mantenible**: Elimina duplicación de código entre los scripts originales
- **Mayor flexibilidad**: Permite configurar más parámetros desde la línea de comandos
- **Mejor organización**: Funciones modulares que pueden reutilizarse
- **Documentación integrada**: Incluye docstrings y mensajes informativos
- **Evaluación consistente**: Aplica las mismas métricas a todos los modelos