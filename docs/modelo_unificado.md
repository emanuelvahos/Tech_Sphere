# Script Unificado para Probar Modelos

Este documento describe el uso del script unificado `modelo_unificado.py` que combina las funcionalidades de los scripts anteriores (`demo_modelo.py` y `probar_modelos.py`) en una única solución más eficiente y completa.

## Descripción

El script unificado permite probar modelos de clasificación de literatura médica con diferentes opciones de uso:

- Cargar un modelo específico o seleccionar entre los disponibles
- Probar el modelo con ejemplos predefinidos
- Realizar predicciones interactivas con textos ingresados por el usuario
- Visualizar las categorías predichas y sus probabilidades asociadas

## Funcionalidades

- **Carga flexible de modelos**: Permite cargar un modelo específico mediante parámetros o seleccionar interactivamente entre los disponibles
- **Predicción de categorías**: Clasifica textos médicos en categorías predefinidas
- **Cálculo de probabilidades**: Muestra la probabilidad de pertenencia a cada categoría
- **Ejemplos predefinidos**: Incluye textos de ejemplo para demostrar el funcionamiento
- **Modo interactivo**: Permite al usuario ingresar sus propios textos para clasificar

## Uso

El script puede ejecutarse desde la línea de comandos con diferentes opciones:

```bash
python modelo_unificado.py [opciones]
```

### Opciones disponibles

- `--model RUTA`: Especifica la ruta a un modelo específico a cargar
- `--interactive-select`: Permite seleccionar un modelo interactivamente entre los disponibles
- `--skip-examples`: Omite la ejecución de ejemplos predefinidos
- `--skip-interactive`: Omite el modo interactivo de predicción

### Ejemplos de uso

1. Ejecutar con el modelo por defecto (SVM):

```bash
python modelo_unificado.py
```

2. Cargar un modelo específico:

```bash
python modelo_unificado.py --model models/random_forest_model.pkl
```

3. Seleccionar un modelo interactivamente:

```bash
python modelo_unificado.py --interactive-select
```

4. Ejecutar solo el modo interactivo (sin ejemplos):

```bash
python modelo_unificado.py --skip-examples
```

5. Ejecutar solo los ejemplos (sin modo interactivo):

```bash
python modelo_unificado.py --skip-interactive
```

## Ventajas sobre los scripts anteriores

- **Código más limpio y mantenible**: Elimina duplicación de código entre los scripts originales
- **Mayor flexibilidad**: Permite configurar el comportamiento mediante parámetros
- **Mejor organización**: Funciones modulares que pueden reutilizarse
- **Documentación integrada**: Incluye docstrings y mensajes informativos
- **Interfaz mejorada**: Combina lo mejor de ambos scripts originales

## Uso programático

El script también puede importarse como módulo y utilizarse programáticamente:

```python
from modelo_unificado import load_model, predict_category, get_prediction_probabilities

# Cargar un modelo
model, vectorizer, mlb = load_model('models/svm_model.pkl')

# Predecir categorías
title = "Efectos cardiovasculares de los inhibidores de la ECA"
abstract = "Este estudio evalúa los efectos de los inhibidores..."
categories = predict_category(title, abstract, model, vectorizer, mlb)

# Obtener probabilidades
probs = get_prediction_probabilities(title, abstract, model, vectorizer, mlb)
```