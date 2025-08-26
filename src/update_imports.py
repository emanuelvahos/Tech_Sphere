#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para actualizar las importaciones en los archivos movidos.

Este script actualiza las rutas de importación en los archivos que han sido
movidos a nuevas ubicaciones en la estructura de directorios optimizada.
"""

import os
import re
import sys

# Mapeo de archivos movidos a sus nuevas ubicaciones
MOVED_FILES = {
    'app_demo.py': 'src/app/app_demo.py',
    'app_demo_en.py': 'src/app/app_demo_en.py',
    'app_multilingual.py': 'src/app/app_multilingual.py',
    'train_models.py': 'src/models/training/train_models.py',
    'train_model_demo.py': 'src/models/training/train_model_demo.py',
    'crear_modelos.py': 'src/models/training/crear_modelos.py',
    'train_models_unificado.py': 'src/models/training/train_models_unificado.py',
    'evaluar_modelos.py': 'src/models/evaluation/evaluar_modelos.py',
    'probar_modelos.py': 'src/models/evaluation/probar_modelos.py',
    'demo_modelo.py': 'src/models/evaluation/demo_modelo.py',
    'modelo_unificado.py': 'src/models/evaluation/modelo_unificado.py',
    'demo_basico.py': 'src/demo/demo_basico.py',
    'demo_avanzado.py': 'src/demo/demo_avanzado.py',
    'entrenar_modelo_demo.py': 'src/demo/entrenar_modelo_demo.py',
}

# Patrones de importación a actualizar
IMPORT_PATTERNS = [
    # from X import Y
    re.compile(r'from\s+([\w\.]+)\s+import\s+([\w\.,\s]+)'),
    # import X
    re.compile(r'import\s+([\w\.]+)'),
    # import X as Y
    re.compile(r'import\s+([\w\.]+)\s+as\s+([\w]+)'),
    # from X import Y as Z
    re.compile(r'from\s+([\w\.]+)\s+import\s+([\w]+)\s+as\s+([\w]+)'),
]

# Patrones de rutas de archivo a actualizar
FILE_PATH_PATTERNS = [
    # os.path.join(...)
    re.compile(r'os\.path\.join\(([^)]+)\)'),
    # open(...)
    re.compile(r'open\(([^)]+)\)'),
]


def update_imports(file_path):
    """Actualiza las importaciones en un archivo."""
    print(f"Actualizando importaciones en {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Actualizar importaciones
    for pattern in IMPORT_PATTERNS:
        matches = pattern.findall(content)
        for match in matches:
            if isinstance(match, tuple):
                module = match[0]
            else:
                module = match
            
            # Verificar si el módulo corresponde a un archivo movido
            for old_path, new_path in MOVED_FILES.items():
                old_module = old_path.replace('.py', '').replace('/', '.')
                if module == old_module:
                    new_module = new_path.replace('.py', '').replace('/', '.')
                    content = content.replace(f"from {module} import", f"from {new_module} import")
                    content = content.replace(f"import {module}", f"import {new_module}")
    
    # Actualizar rutas de archivo
    for pattern in FILE_PATH_PATTERNS:
        matches = pattern.findall(content)
        for match in matches:
            for old_path, new_path in MOVED_FILES.items():
                if old_path in match:
                    content = content.replace(old_path, new_path)
    
    # Guardar el archivo actualizado
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Importaciones actualizadas en {file_path}")


def main():
    """Función principal."""
    print("=== ACTUALIZANDO IMPORTACIONES ===\n")
    
    # Actualizar importaciones en todos los archivos movidos
    for old_path, new_path in MOVED_FILES.items():
        if os.path.exists(new_path):
            update_imports(new_path)
    
    print("\n=== ACTUALIZACIÓN COMPLETADA ===\n")


if __name__ == "__main__":
    main()