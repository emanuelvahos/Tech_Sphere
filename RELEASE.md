# Proceso de Releases

Este documento describe el proceso para crear y gestionar releases del proyecto de Clasificación de Literatura Médica.

## Versionado Semántico

Este proyecto sigue el [Versionado Semántico 2.0.0](https://semver.org/lang/es/):

- **MAJOR.MINOR.PATCH** (ejemplo: 1.2.3)
  - **MAJOR**: Cambios incompatibles con versiones anteriores
  - **MINOR**: Nuevas funcionalidades compatibles con versiones anteriores
  - **PATCH**: Correcciones de errores compatibles con versiones anteriores

## Proceso de Release

### 1. Preparación

1. Asegúrate de que todas las pruebas pasen en la rama principal:
   ```bash
   pytest
   ```

2. Actualiza la documentación si es necesario.

3. Actualiza el archivo CHANGELOG.md con los cambios de la nueva versión.

### 2. Creación de la Release

1. Crea y sube un tag con la nueva versión:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

2. Crea una nueva release en GitHub:
   - Ve a la sección "Releases" del repositorio
   - Haz clic en "Draft a new release"
   - Selecciona el tag que acabas de crear
   - Añade un título descriptivo
   - Incluye las notas de la release basadas en el CHANGELOG
   - Adjunta cualquier archivo binario o distribución si es aplicable

### 3. Post-Release

1. Actualiza la versión en desarrollo en los archivos relevantes para reflejar la próxima versión planificada.

2. Anuncia la nueva release en los canales de comunicación apropiados.

## Tipos de Releases

### Release Estable (vX.Y.Z)

Las releases estables son versiones completamente probadas y listas para uso en producción.

### Pre-Release (vX.Y.Z-alpha.N, vX.Y.Z-beta.N, vX.Y.Z-rc.N)

- **Alpha**: Versiones tempranas con funcionalidades incompletas y posibles errores significativos.
- **Beta**: Versiones con funcionalidades completas pero que pueden contener errores.
- **Release Candidate (RC)**: Versiones candidatas a ser estables, con correcciones de errores finales.

## Mantenimiento de Versiones Anteriores

- Las versiones MAJOR anteriores recibirán correcciones de seguridad durante 6 meses después de la liberación de una nueva versión MAJOR.
- Las correcciones críticas se aplicarán a la versión actual y a la versión MAJOR anterior.

## Política de Compatibilidad

- Las actualizaciones PATCH y MINOR no deben romper la compatibilidad con versiones anteriores.
- Las actualizaciones MAJOR pueden incluir cambios que rompan la compatibilidad, que serán documentados claramente en las notas de la release.

## Changelog

El archivo CHANGELOG.md mantiene un registro de todos los cambios significativos en cada versión, organizados por versión y tipo de cambio (Added, Changed, Deprecated, Removed, Fixed, Security).