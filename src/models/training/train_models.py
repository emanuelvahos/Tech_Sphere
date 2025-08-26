# -*- coding: utf-8 -*-
"""
Script to create multiple medical literature classification models.

This script trains several types of models (SVM, Random Forest, Logistic Regression)
and saves them in the models folder for later use.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Add the root directory to the path to import project modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data.preprocess import tokenize_text

# Configure directories
RAW_DATA_PATH = os.path.join('data', 'raw', 'challenge_data-18-ago.csv')
PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')

# Create directories if they don't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(RAW_DATA_PATH, sep=';')
print(f"Data loaded: {df.shape[0]} records")

# Preprocess texts
print("Preprocessing texts...")

def preprocess_text(text):
    """Preprocess a text by applying cleaning, stopword removal, and lemmatization."""
    if pd.isna(text):
        return ""
    # Use the tokenize_text function that already includes cleaning, stopword removal, and lemmatization
    tokens = tokenize_text(text, remove_stop=True, lemmatize=True)
    return ' '.join(tokens)

# Apply preprocessing
df['title_processed'] = df['title'].apply(preprocess_text)
df['abstract_processed'] = df['abstract'].apply(preprocess_text)
df['text_combined'] = df['title_processed'] + " " + df['abstract_processed']

# Prepare labels
print("Preparing labels...")
df['group_list'] = df['group'].apply(lambda x: x.split('|') if pd.notna(x) else [])

# Binarize labels
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['group_list'])
print(f"Categories: {mlb.classes_}")

# Extract TF-IDF features
print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text_combined'])
print(f"Feature matrix: {X.shape}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Function to train and save a model
def train_and_save_model(model_type, model_name, params=None):
    print(f"\nTraining model {model_name}...")
    
    if params is None:
        params = {}
    
    # Create the model according to the type
    if model_type == 'svm':
        base_model = LinearSVC(random_state=42, **params)
    elif model_type == 'rf':
        base_model = RandomForestClassifier(random_state=42, **params)
    elif model_type == 'logistic':
        base_model = LogisticRegression(random_state=42, **params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Wrap in OneVsRestClassifier for multi-label classification
    model = OneVsRestClassifier(base_model)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate model
    print(f"Evaluating model {model_name}...")
    y_pred = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_))
    
    # Save model, vectorizer, and binarizer
    print(f"Saving model {model_name}...")
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'mlb': mlb
    }
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved at {model_path}")
    return model_path

# Train and save SVM model
svm_path = train_and_save_model('svm', 'svm_model', {'C': 1.0, 'max_iter': 1000})

# Train and save Random Forest model
rf_path = train_and_save_model('rf', 'random_forest_model', {'n_estimators': 100, 'max_depth': 10})

# Train and save Logistic Regression model
lr_path = train_and_save_model('logistic', 'logistic_regression_model', {'C': 1.0, 'max_iter': 1000})

print("\nAll models have been trained and saved successfully.")
print("\nAvailable models:")
print(f"- SVM: {svm_path}")
print(f"- Random Forest: {rf_path}")
print(f"- Logistic Regression: {lr_path}")
print("\nYou can use these models with the scripts:")
print("- src/models/test_model.py: To test the models with individual texts")
print("- src/models/evaluate_model.py: To evaluate the models with a dataset")
print("- src/app.py: To use the models through a web interface")