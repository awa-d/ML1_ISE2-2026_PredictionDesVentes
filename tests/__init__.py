"""
==============================================================================
tests/__init__.py - Initialisation du Package de Tests
==============================================================================

Ce fichier transforme le dossier 'tests/' en un package Python, permettant
à Pytest et Unittest de découvrir et exécuter automatiquement les tests.

Structure :
    tests/
    ├── __init__.py             <- Ce fichier (config + fixtures)
    ├── test_pipeline.py        <- Tests du pipeline preprocessing
    └── fixtures/               <- Données de test réduites (optionnel)
"""

import os
import sys
import polars as pl
import numpy as np
from datetime import date, timedelta

# ==============================================================================
# CONFIGURATION DES CHEMINS
# ==============================================================================

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(TESTS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Ajout de src/ au PYTHONPATH
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==============================================================================
# PARAMÈTRES DE TEST
# ==============================================================================

SAMPLE_SIZE = 1000
REQUIRED_TRAIN_COLUMNS = ["id", "date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]

# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def is_data_available() -> bool:
    """Vérifie si les données complètes sont disponibles."""
    paths_to_check = [
        os.path.join(DATA_DIR, "favorita-grocery-sales-forecasting", "train.csv"),
        os.path.join(DATA_DIR, "train.csv"),
        os.path.join(DATA_DIR, "corporacin-favorita-grocery-sales-forecasting", "train.csv"),
    ]
    return any(os.path.exists(p) for p in paths_to_check)


def get_sample_data_path(filename: str) -> str:
    """Retourne le chemin vers un fichier de données de test."""
    path = os.path.join(FIXTURES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de test introuvable : {path}")
    return path


def create_fixtures_dir():
    """Crée le dossier fixtures/ s'il n'existe pas."""
    if not os.path.exists(FIXTURES_DIR):
        os.makedirs(FIXTURES_DIR)


# ==============================================================================
# GÉNÉRATEURS DE DONNÉES DE TEST (Fixtures)
# ==============================================================================

def create_sample_dataframe(n_rows: int = 100) -> pl.DataFrame:
    """
    Crée un DataFrame de test mimiquant la structure de train.csv.
    Utilisé par les tests unitaires.
    """
    np.random.seed(42)
    
    # Génération des dates
    base_date = date(2017, 1, 1)
    dates = [base_date + timedelta(days=i % 60) for i in range(n_rows)]
    
    return pl.DataFrame({
        "id": list(range(1, n_rows + 1)),
        "date": dates,
        "store_nbr": np.random.randint(1, 55, size=n_rows).tolist(),
        "item_nbr": np.random.randint(100000, 200000, size=n_rows).tolist(),
        "unit_sales": np.random.exponential(scale=10, size=n_rows).round(2).tolist(),
        "onpromotion": np.random.choice([0, 1], size=n_rows).tolist(),
        # Colonnes supplémentaires pour les jointures
        "family": np.random.choice(["GROCERY I", "BEVERAGES", "CLEANING"], size=n_rows).tolist(),
        "perishable": np.random.choice([0, 1], size=n_rows).tolist(),
        "city": np.random.choice(["Quito", "Guayaquil", "Cuenca"], size=n_rows).tolist(),
        "state": np.random.choice(["Pichincha", "Guayas", "Azuay"], size=n_rows).tolist(),
        "type": np.random.choice(["A", "B", "C", "D"], size=n_rows).tolist(),
        "cluster": np.random.randint(1, 18, size=n_rows).tolist(),
    })


def create_sample_oil_dataframe(n_days: int = 30) -> pl.DataFrame:
    """
    Crée un DataFrame de test pour les données pétrole (oil.csv).
    """
    np.random.seed(42)
    
    base_date = date(2017, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_days)]
    
    # Prix du pétrole avec tendance + bruit
    base_price = 50.0
    prices = [base_price + np.sin(i * 0.1) * 5 + np.random.normal(0, 2) for i in range(n_days)]
    
    return pl.DataFrame({
        "date": dates,
        "dcoilwtico": prices,
    })


def create_sample_holidays_dataframe(n_events: int = 10) -> pl.DataFrame:
    """
    Crée un DataFrame de test pour les jours fériés.
    """
    base_date = date(2017, 1, 1)
    
    return pl.DataFrame({
        "date": [base_date + timedelta(days=i * 15) for i in range(n_events)],
        "type": ["Holiday", "Event", "Additional", "Transfer", "Bridge"] * 2,
        "locale": ["National", "Regional", "Local", "National", "National"] * 2,
        "locale_name": ["Ecuador", "Pichincha", "Quito", "Ecuador", "Ecuador"] * 2,
        "description": [f"Event_{i}" for i in range(n_events)],
        "transferred": [False] * n_events,
    })


# ==============================================================================
# CONFIGURATION ENVIRONNEMENT DE TEST
# ==============================================================================

os.environ["FAVORITA_TEST_MODE"] = "1"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# EXPORTS
# ==============================================================================

__all__ = [
    # Chemins
    "TESTS_DIR",
    "PROJECT_ROOT", 
    "SRC_DIR",
    "FIXTURES_DIR",
    "DATA_DIR",
    # Paramètres
    "SAMPLE_SIZE",
    "REQUIRED_TRAIN_COLUMNS",
    # Fonctions utilitaires
    "is_data_available",
    "get_sample_data_path",
    "create_fixtures_dir",
    # Générateurs de fixtures
    "create_sample_dataframe",
    "create_sample_oil_dataframe",
    "create_sample_holidays_dataframe",
]

print(f"[TESTS] Package initialisé. SRC_DIR={SRC_DIR}")
