# -*- coding: utf-8 -*-
"""
Tests para el módulo de preprocesamiento de texto.
"""

import unittest
import sys
import os

# Añadir el directorio raíz al path para importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import clean_text, remove_stopwords, lemmatize_text


class TestPreprocess(unittest.TestCase):
    """Pruebas para las funciones de preprocesamiento."""

    def test_clean_text(self):
        """Prueba la función de limpieza de texto."""
        text = "Este es un texto de prueba con MAYÚSCULAS y símbolos!!"
        cleaned = clean_text(text)
        self.assertEqual(cleaned.lower(), cleaned)
        self.assertNotIn("!!", cleaned)

    def test_remove_stopwords(self):
        """Prueba la función de eliminación de stopwords."""
        text = "este es un texto con palabras comunes"
        processed = remove_stopwords(text)
        self.assertNotIn("es", processed.split())
        self.assertNotIn("un", processed.split())

    def test_lemmatize_text(self):
        """Prueba la función de lematización."""
        text = "corriendo caminando estudiando"
        lemmatized = lemmatize_text(text)
        self.assertIn("correr", lemmatized)
        self.assertIn("caminar", lemmatized)
        self.assertIn("estudiar", lemmatized)


if __name__ == '__main__':
    unittest.main()