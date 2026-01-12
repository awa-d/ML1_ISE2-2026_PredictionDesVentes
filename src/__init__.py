"""
==============================================================================
src/__init__.py - Initialisation du Package Favorita
==============================================================================

Ce fichier transforme le dossier 'src/' en un package Python importable.

IMPORTANT : 
- Quand vous exécutez un script DIRECTEMENT (ex: python train.py), 
  ce fichier ne sera PAS utilisé car Python ne voit pas 'src' comme un package.
- Quand vous importez DEPUIS l'extérieur (ex: from src import preprocessing),
  ce fichier est chargé et les imports ci-dessous sont disponibles.

Structure du Package :
    src/
    ├── __init__.py         <- Ce fichier
    ├── preprocessing.py    <- Pipeline de nettoyage et Feature Engineering
    ├── train.py            <- Entraînement du modèle LightGBM
    └── predict.py          <- API de prédiction (FastAPI)
"""

# ==============================================================================
# MÉTADONNÉES DU PACKAGE
# ==============================================================================

__version__ = "1.0.0"
__author__ = "Alioune Kane"
__project__ = "Favorita Grocery Sales Forecasting - ENSAE ISE2 2026"

# ==============================================================================
# IMPORTS CONDITIONNELS (Évite les erreurs si exécuté directement)
# ==============================================================================

try:
    # Ces imports fonctionnent quand 'src' est utilisé comme package
    # (c'est-à-dire quand on importe depuis le dossier parent)
    from . import preprocessing
    from . import train
    from . import predict
    
    # Exposition des fonctions clés
    from .preprocessing import (
        load_and_preprocess_data,
        get_dynamic_datasets,
        download_data,
    )
    
    __all__ = [
        "preprocessing",
        "train", 
        "predict",
        "load_and_preprocess_data",
        "get_dynamic_datasets",
        "download_data",
    ]
    
except ImportError:
    # Si on exécute un script directement depuis src/, 
    # les imports relatifs échouent - c'est normal.
    # Dans ce cas, on laisse __init__.py vide fonctionnellement.
    __all__ = []
