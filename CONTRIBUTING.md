# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto de Clasificación de Literatura Médica! Esta guía te ayudará a entender el proceso de contribución.

## Cómo Contribuir

### 1. Configuración del Entorno

1. Clona el repositorio:
   ```bash
   git clone https://github.com/emanuelvahos/Tech_Sphere.git
   cd Tech_Sphere
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

### 2. Desarrollo

1. Crea una rama para tu contribución:
   ```bash
   git checkout -b feature/nombre-de-tu-caracteristica
   ```

2. Realiza tus cambios siguiendo las convenciones de código del proyecto.

3. Asegúrate de que los tests pasen:
   ```bash
   pytest
   ```

4. Actualiza la documentación si es necesario.

### 3. Envío de Cambios

1. Haz commit de tus cambios:
   ```bash
   git add .
   git commit -m "Descripción clara de los cambios"
   ```

2. Sube tus cambios a tu fork:
   ```bash
   git push origin feature/nombre-de-tu-caracteristica
   ```

3. Crea un Pull Request desde tu rama a la rama principal del repositorio.

## Convenciones de Código

- Sigue la guía de estilo PEP 8 para Python.
- Usa nombres descriptivos para variables y funciones.
- Documenta todas las funciones y clases con docstrings.
- Escribe tests para toda nueva funcionalidad.

## Estructura del Proyecto

```
Tech_Sphere/
├── data/               # Datos del proyecto
├── notebooks/          # Jupyter notebooks para análisis
├── src/                # Código fuente del proyecto
├── tests/              # Tests unitarios
├── docs/               # Documentación
└── reports/            # Reportes y visualizaciones
```

## Flujo de Trabajo para Issues

1. Revisa los issues existentes antes de crear uno nuevo.
2. Usa etiquetas apropiadas para clasificar tu issue (bug, enhancement, etc.).
3. Proporciona toda la información necesaria para reproducir el problema o entender la propuesta.

## Proceso de Revisión

1. Cada Pull Request será revisado por al menos un mantenedor del proyecto.
2. Los comentarios de revisión deben ser abordados antes de la fusión.
3. Los tests automatizados deben pasar para que un PR sea considerado.

## Contacto

Si tienes preguntas o necesitas ayuda, puedes contactar al equipo de mantenimiento a través de:
- Email: emmanuelvahos@gmail.com

¡Gracias por contribuir a mejorar la clasificación de literatura médica!