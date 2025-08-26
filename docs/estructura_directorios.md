# Plan de Optimización de Estructura de Directorios

## Estructura Actual

Actualmente, el proyecto tiene varios scripts en el directorio raíz que deberían estar organizados en carpetas específicas según su funcionalidad.

## Plan de Reorganización

### 1. Scripts de Aplicación Web

Mover a `src/app/`:
- `app_demo.py`
- `app_demo_en.py`
- `app_multilingual.py`

### 2. Scripts de Entrenamiento de Modelos

Mover a `src/models/training/`:
- `train_models.py`
- `train_model_demo.py`
- `crear_modelos.py`
- `train_models_unificado.py`

### 3. Scripts de Evaluación y Prueba de Modelos

Mover a `src/models/evaluation/`:
- `evaluar_modelos.py`
- `probar_modelos.py`
- `demo_modelo.py`
- `modelo_unificado.py`

### 4. Scripts de Demostración

Mover a `src/demo/`:
- `demo_basico.py`
- `demo_avanzado.py`
- `entrenar_modelo_demo.py`

### 5. Plantillas HTML

Mantener en `templates/`:
- `app_demo.html`
- `app_demo_en.html`
- `app_multilingual.html`
- `index.html`

### 6. Datos

Mover a `data/raw/`:
- `challenge_data-18-ago.csv`

## Estructura Final Propuesta

```
Tech_Sphere/
├── data/
│   ├── processed/
│   └── raw/
│       └── challenge_data-18-ago.csv
├── docs/
│   └── ...
├── models/
│   └── ...
├── notebooks/
│   └── ...
├── reports/
│   └── ...
├── src/
│   ├── app/
│   │   ├── app_demo.py
│   │   ├── app_demo_en.py
│   │   └── app_multilingual.py
│   ├── data/
│   │   └── preprocess.py
│   ├── demo/
│   │   ├── demo_basico.py
│   │   ├── demo_avanzado.py
│   │   ├── entrenar_modelo_demo.py
│   │   └── demo_utils.py
│   ├── features/
│   │   └── feature_extraction.py
│   ├── models/
│   │   ├── evaluate_model.py
│   │   ├── test_model.py
│   │   ├── train_model.py
│   │   ├── training/
│   │   │   ├── train_models.py
│   │   │   ├── train_model_demo.py
│   │   │   ├── crear_modelos.py
│   │   │   └── train_models_unificado.py
│   │   └── evaluation/
│   │       ├── evaluar_modelos.py
│   │       ├── probar_modelos.py
│   │       ├── demo_modelo.py
│   │       └── modelo_unificado.py
│   └── visualization/
│       └── visualize.py
├── templates/
│   ├── app_demo.html
│   ├── app_demo_en.html
│   ├── app_multilingual.html
│   └── index.html
└── tests/
    └── ...
```

## Ventajas de la Nueva Estructura

1. **Mejor organización**: Los archivos están agrupados por funcionalidad
2. **Mayor claridad**: Es más fácil encontrar los archivos relacionados
3. **Mantenibilidad mejorada**: Facilita el mantenimiento y la extensión del proyecto
4. **Separación de responsabilidades**: Clara distinción entre diferentes componentes del sistema
5. **Escalabilidad**: Estructura preparada para crecer con nuevas funcionalidades