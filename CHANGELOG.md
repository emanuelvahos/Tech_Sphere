# Changelog

Todas las modificaciones notables a este proyecto serán documentadas en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Añadido
- Configuración de GitHub Actions para pruebas automatizadas
- Badges en README.md para mostrar el estado del proyecto
- Documentación para contribuidores (CONTRIBUTING.md)
- Proceso de releases documentado (RELEASE.md)

## [1.0.0] - 2023-07-15

### Añadido
- Implementación inicial del sistema de clasificación multi-etiqueta
- Modelos de machine learning: SVM, Random Forest, Gradient Boosting, MLP
- Pipeline de preprocesamiento para textos médicos
- Análisis exploratorio de datos
- Evaluación detallada de modelos con métricas específicas para clasificación multi-etiqueta
- Visualizaciones de resultados y matrices de confusión

### Características
- Clasificación en cuatro categorías médicas: Cardiovascular, Hepatorenal, Neurológico, Oncológico
- F1-score ponderado de 0.85 con el modelo SVM
- Preprocesamiento específico para textos médicos
- Extracción de características mediante TF-IDF

[Unreleased]: https://github.com/tu-usuario/Tech_Sphere/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tu-usuario/Tech_Sphere/releases/tag/v1.0.0