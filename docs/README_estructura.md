# Nueva Estructura de Directorios

## Descripción

Este documento describe la nueva estructura de directorios optimizada para el proyecto Tech_Sphere. La reorganización se ha realizado para mejorar la organización, mantenibilidad y escalabilidad del proyecto.

## Estructura de Directorios

```
Tech_Sphere/
├── data/
│   ├── processed/
│   └── raw/
│       └── challenge_data-18-ago.csv
├── docs/
│   ├── como_usar_modelos.md
│   ├── demo_refactorizado.md
│   ├── documentacion_proyecto.md
│   ├── estructura_directorios.md
│   ├── modelo_unificado.md
│   ├── README_estructura.md
│   ├── scripts_demo.md
│   └── train_models_unificado.md
├── models/
│   └── ...
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_model_evaluation.ipynb
├── reports/
│   ├── figures/
│   └── informe_final.md
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
    ├── __init__.py
    └── test_preprocess.py
```

## Descripción de los Directorios

### `data/`

Contiene los datos utilizados por el proyecto.

- `processed/`: Datos procesados listos para ser utilizados por los modelos.
- `raw/`: Datos sin procesar.

### `docs/`

Contiene la documentación del proyecto.

### `models/`

Contiene los modelos entrenados.

### `notebooks/`

Contiene los notebooks de Jupyter utilizados para el análisis exploratorio de datos y la evaluación de modelos.

### `reports/`

Contiene informes y visualizaciones generados por el proyecto.

- `figures/`: Visualizaciones generadas por los scripts de evaluación.

### `src/`

Contiene el código fuente del proyecto.

- `app/`: Scripts de aplicación web.
- `data/`: Scripts para el procesamiento de datos.
- `demo/`: Scripts de demostración.
- `features/`: Scripts para la extracción de características.
- `models/`: Scripts para el entrenamiento y evaluación de modelos.
  - `training/`: Scripts para el entrenamiento de modelos.
  - `evaluation/`: Scripts para la evaluación y prueba de modelos.
- `visualization/`: Scripts para la visualización de resultados.

### `templates/`

Contiene las plantillas HTML utilizadas por las aplicaciones web.

### `tests/`

Contiene las pruebas unitarias del proyecto.

## Ventajas de la Nueva Estructura

1. **Mejor organización**: Los archivos están agrupados por funcionalidad.
2. **Mayor claridad**: Es más fácil encontrar los archivos relacionados.
3. **Mantenibilidad mejorada**: Facilita el mantenimiento y la extensión del proyecto.
4. **Separación de responsabilidades**: Clara distinción entre diferentes componentes del sistema.
5. **Escalabilidad**: Estructura preparada para crecer con nuevas funcionalidades.

## Actualización de Importaciones

Se ha creado un script `src/update_imports.py` para actualizar las importaciones en los archivos movidos. Este script actualiza las rutas de importación en los archivos que han sido movidos a nuevas ubicaciones en la estructura de directorios optimizada.

Para ejecutar el script:

```bash
python src/update_imports.py
```

Esto actualizará automáticamente las importaciones en todos los archivos movidos.