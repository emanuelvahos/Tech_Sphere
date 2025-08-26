# Guía para Usar los Modelos de Clasificación de Literatura Médica

Esta guía explica cómo utilizar los modelos de clasificación de literatura médica implementados en este proyecto. Se proporcionan tres formas diferentes de interactuar con los modelos:

1. **Línea de comandos**: Para pruebas rápidas con textos individuales
2. **Evaluación de modelos**: Para evaluar el rendimiento con conjuntos de datos
3. **Aplicación web**: Para una interfaz gráfica amigable

## 1. Usando el Script de Línea de Comandos

El script `test_model.py` permite probar los modelos entrenados con textos individuales desde la línea de comandos.

### Pasos para usar el script:

1. Asegúrate de tener al menos un modelo entrenado en la carpeta `models/`
2. Ejecuta el script:

```bash
python src/models/test_model.py
```

3. Selecciona el modelo que deseas utilizar de la lista mostrada
4. Ingresa el título y resumen del artículo médico que deseas clasificar
5. El script mostrará las categorías predichas y, si están disponibles, las probabilidades para cada categoría

### Ejemplo de uso:

```
=== SISTEMA DE CLASIFICACIÓN DE LITERATURA MÉDICA ===

Modelos disponibles:
  1. svm_model_20230715.pkl
  2. rf_model_20230715.pkl

Seleccione un modelo por número: 1
Modelo cargado correctamente desde models/svm_model_20230715.pkl

El modelo puede clasificar textos en las siguientes categorías:
  - Cardiovascular
  - Hepatorenal
  - Neurológico
  - Oncológico

--------------------------------------------------
Ingrese un texto médico para clasificar (o 'salir' para terminar)

Título del artículo: Estudio sobre la eficacia de nuevos tratamientos para el glioblastoma
Resumen del artículo: Este estudio evalúa la eficacia de nuevas terapias combinadas para el tratamiento del glioblastoma multiforme, un tipo agresivo de tumor cerebral. Los resultados muestran una mejora significativa en la supervivencia de los pacientes tratados con la combinación de inmunoterapia y terapia dirigida en comparación con los tratamientos estándar.

Categorías predichas:
  - Neurológico
  - Oncológico

Probabilidades por categoría:
  - Oncológico: 0.8765
  - Neurológico: 0.7432
  - Cardiovascular: 0.1234
  - Hepatorenal: 0.0543
```

## 2. Evaluación de Modelos con Conjuntos de Datos

El script `evaluate_model.py` permite evaluar el rendimiento de los modelos con conjuntos de datos completos y generar métricas detalladas.

### Pasos para evaluar un modelo:

1. Asegúrate de tener al menos un modelo entrenado en la carpeta `models/`
2. Prepara un conjunto de datos de prueba en formato CSV con las columnas 'title', 'abstract' y 'group'
3. Ejecuta el script:

```bash
python src/models/evaluate_model.py --model models/svm_model.pkl --data data/processed/test_data.csv
```

Si no especificas los parámetros `--model` y `--data`, el script te permitirá seleccionar un modelo de la lista disponible y utilizará el conjunto de datos de prueba por defecto.

### Resultados de la evaluación:

El script generará:

- Métricas generales: accuracy, precision, recall, F1-score, etc.
- Métricas por categoría
- Visualizaciones: gráfico de radar con métricas y matrices de confusión

Las visualizaciones se guardarán en la carpeta `reports/figures/`.

## 3. Aplicación Web

La aplicación web proporciona una interfaz gráfica amigable para clasificar textos médicos.

### Pasos para ejecutar la aplicación web:

1. Asegúrate de tener Flask instalado:

```bash
pip install flask
```

2. Ejecuta la aplicación:

```bash
python src/app.py
```

3. Abre tu navegador y accede a: http://127.0.0.1:5000/

4. Utiliza el formulario para ingresar el título y resumen del artículo médico que deseas clasificar

5. Haz clic en "Clasificar" para obtener los resultados

### Características de la aplicación web:

- Interfaz intuitiva y fácil de usar
- Visualización de categorías predichas
- Gráficos de barras con las probabilidades para cada categoría
- Diseño responsive que se adapta a diferentes dispositivos

## Recomendaciones para Obtener Mejores Resultados

1. **Preprocesamiento adecuado**: Los textos deben estar en español y seguir un formato similar al de los datos de entrenamiento.

2. **Longitud del texto**: Proporciona textos con suficiente información. Los resúmenes muy cortos pueden no contener suficientes características para una clasificación precisa.

3. **Modelo recomendado**: El modelo SVM ha mostrado el mejor rendimiento general con un F1-score ponderado de 0.85.

4. **Interpretación de resultados**: Ten en cuenta que algunas categorías pueden tener un rendimiento mejor que otras. Por ejemplo, la categoría "Neurológico" tiene el mejor rendimiento, mientras que "Oncológico" puede presentar más falsos negativos debido al desbalance en los datos de entrenamiento.

## Solución de Problemas

1. **Error al cargar el modelo**: Asegúrate de que exista al menos un modelo entrenado en la carpeta `models/`. Si no hay modelos disponibles, ejecuta primero el script de entrenamiento:

```bash
python src/models/train_model.py
```

2. **Problemas con las dependencias**: Verifica que todas las dependencias estén instaladas correctamente:

```bash
pip install -r requirements.txt
```

3. **Resultados inesperados**: Si obtienes resultados inesperados, verifica que el texto proporcionado sea relevante para las categorías médicas soportadas y que contenga suficiente información técnica.