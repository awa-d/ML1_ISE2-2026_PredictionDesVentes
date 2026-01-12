
"""
==============================================================================
tests/test_pipeline.py - Tests du Pipeline de Prétraitement
==============================================================================

Ce fichier contient les tests unitaires et d'intégration pour valider
le bon fonctionnement du pipeline de prétraitement (preprocessing.py).

Objectifs :
1. Valider les transformations (nettoyage, FE)
2. Détecter les régressions après modification du code
3. Vérifier les formats de sortie
4. S'exécuter automatiquement via GitHub Actions

Usage :
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -k "test_winsorization" -v
"""

import pytest
import polars as pl
import numpy as np
import os
import sys
from datetime import date, timedelta

# ==============================================================================
# CONFIGURATION DES CHEMINS (Compatible exécution directe ET pytest)
# ==============================================================================

# Chemin absolu vers ce fichier
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers la racine du projet (parent de tests/)
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

# Chemin vers le dossier src/
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

# Chemin vers les données
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# Ajout des chemins au PYTHONPATH
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import des modules à tester (depuis src/)
from preprocessing import (
    clean_date_column,
    engineer_date_features,
    engineer_oil_features,
    engineer_store_item_features,
    winsorize_sales_by_group,
    perform_target_encoding,
    select_features,
    scale_numerical_features_eager,
)


# ==============================================================================
# GÉNÉRATEURS DE DONNÉES DE TEST (Inline - pas de dépendance externe)
# ==============================================================================

def is_data_available() -> bool:
    """Vérifie si les données complètes sont disponibles."""
    paths_to_check = [
        os.path.join(DATA_DIR, "favorita-grocery-sales-forecasting", "train.csv"),
        os.path.join(DATA_DIR, "train.csv"),
        os.path.join(DATA_DIR, "corporacin-favorita-grocery-sales-forecasting", "train.csv"),
    ]
    return any(os.path.exists(p) for p in paths_to_check)


def create_sample_dataframe(n_rows: int = 100) -> pl.DataFrame:
    """Crée un DataFrame de test mimiquant train.csv."""
    np.random.seed(42)
    base_date = date(2017, 1, 1)
    dates = [base_date + timedelta(days=i % 60) for i in range(n_rows)]
    
    return pl.DataFrame({
        "id": list(range(1, n_rows + 1)),
        "date": dates,
        "store_nbr": np.random.randint(1, 55, size=n_rows).tolist(),
        "item_nbr": np.random.randint(100000, 200000, size=n_rows).tolist(),
        "unit_sales": np.random.exponential(scale=10, size=n_rows).round(2).tolist(),
        "onpromotion": np.random.choice([0, 1], size=n_rows).tolist(),
        "family": np.random.choice(["GROCERY I", "BEVERAGES", "CLEANING"], size=n_rows).tolist(),
        "perishable": np.random.choice([0, 1], size=n_rows).tolist(),
        "city": np.random.choice(["Quito", "Guayaquil", "Cuenca"], size=n_rows).tolist(),
        "state": np.random.choice(["Pichincha", "Guayas", "Azuay"], size=n_rows).tolist(),
        "type": np.random.choice(["A", "B", "C", "D"], size=n_rows).tolist(),
        "cluster": np.random.randint(1, 18, size=n_rows).tolist(),
    })


def create_sample_oil_dataframe(n_days: int = 30) -> pl.DataFrame:
    """Crée un DataFrame de test pour oil.csv."""
    np.random.seed(42)
    base_date = date(2017, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_days)]
    prices = [50.0 + np.sin(i * 0.1) * 5 + np.random.normal(0, 2) for i in range(n_days)]
    
    return pl.DataFrame({"date": dates, "dcoilwtico": prices})



# ==============================================================================
# FIXTURES PYTEST (Données de test réutilisables)
# ==============================================================================

@pytest.fixture
def sample_train_df():
    """
    Crée un DataFrame de test mimiquant la structure de train.csv.
    """
    return create_sample_dataframe(n_rows=100)


@pytest.fixture
def sample_oil_df():
    """
    Crée un DataFrame de test pour les données pétrole.
    """
    return create_sample_oil_dataframe(n_days=30)


@pytest.fixture
def sample_train_valid_pair():
    """
    Crée une paire Train/Valid pour les tests de Target Encoding et Scaling.
    """
    train = create_sample_dataframe(n_rows=200)
    valid = create_sample_dataframe(n_rows=50)
    return train, valid


# ==============================================================================
# TESTS DE VALIDATION DES TRANSFORMATIONS
# ==============================================================================

class TestDateCleaning:
    """Tests pour la fonction clean_date_column."""
    
    def test_string_date_conversion(self):
        """Vérifie que les dates string sont converties en pl.Date."""
        df = pl.DataFrame({
            "date": ["2017-01-01", "2017-01-02", "2017-01-03"],
            "value": [1, 2, 3]
        })
        
        result = clean_date_column(df.lazy()).collect()
        
        # Correction : type check correct
        assert result["date"].dtype == pl.Date
        assert result["date"][0] == date(2017, 1, 1)
    
    def test_no_null_dates_after_cleaning(self):
        """Vérifie qu'il n'y a pas de dates nulles après nettoyage."""
        df = pl.DataFrame({
            "date": ["2017-01-01", "2017-01-02"],
            "value": [1, 2]
        })
        
        result = clean_date_column(df.lazy()).collect()
        
        assert result["date"].null_count() == 0


class TestDateFeatures:
    """Tests pour la fonction engineer_date_features."""
    
    def test_creates_expected_columns(self, sample_train_df):
        """Vérifie que les colonnes temporelles sont créées."""
        # On s'assure que la date est typée Date
        df = sample_train_df.with_columns(pl.col("date").cast(pl.Date))
        result = engineer_date_features(df.lazy()).collect()
        
        expected_cols = ["day", "month", "year", "day_of_week", "is_weekend", "is_payday", "is_year_end"]
        for col in expected_cols:
            assert col in result.columns, f"Colonne manquante : {col}"
    
    def test_is_weekend_correct(self):
        """Vérifie la logique du flag weekend."""
        df = pl.DataFrame({
            "date": [
                date(2017, 1, 7),  # Samedi
                date(2017, 1, 8),  # Dimanche
                date(2017, 1, 9),  # Lundi
            ]
        }).with_columns(pl.col("date").cast(pl.Date))
        
        result = engineer_date_features(df.lazy()).collect()
        
        # Samedi (6) et Dimanche (7) -> 1, Lundi (1) -> 0
        assert result["is_weekend"][0] == 1
        assert result["is_weekend"][1] == 1
        assert result["is_weekend"][2] == 0
    
    def test_is_payday_on_15th(self):
        """Vérifie le flag payday le 15 du mois."""
        df = pl.DataFrame({
            "date": [date(2017, 3, 15)]
        }).with_columns(pl.col("date").cast(pl.Date))
        
        result = engineer_date_features(df.lazy()).collect()
        
        assert result["is_payday"][0] == 1


class TestOilFeatures:
    """Tests pour la fonction engineer_oil_features."""
    
    def test_creates_rolling_and_lag(self, sample_oil_df):
        """Vérifie la création des features pétrole."""
        result = engineer_oil_features(sample_oil_df.lazy()).collect()
        
        assert "oil_smooth_7d" in result.columns
        assert "oil_lag_10" in result.columns
    
    def test_no_nan_in_rolling_mean(self, sample_oil_df):
        """Vérifie que le rolling mean ne produit pas que des NaN."""
        result = engineer_oil_features(sample_oil_df.lazy()).collect()
        
        # On ignore les premières lignes qui ont forcément des nulls pour le lag
        non_null_count = result["oil_smooth_7d"].is_not_null().sum()
        assert non_null_count > 0, "Aucune valeur dans oil_smooth_7d"


class TestWinsorization:
    """Tests pour la fonction winsorize_sales_by_group."""
    
    def test_creates_winsorized_column(self, sample_train_df):
        """Vérifie la création de la colonne winsorisée."""
        result = winsorize_sales_by_group(sample_train_df.lazy()).collect()
        
        assert "unit_sales_win" in result.columns
    
    def test_no_extreme_values(self):
        """Vérifie que les valeurs extrêmes sont bornées."""
        # On crée un groupe avec 1000 points. 
        # Le 99.5ème percentile sera la 995ème valeur.
        # Si la 1000ème est un outlier, elle sera ramenée à la 995ème.
        n = 1000
        dfv = pl.DataFrame({
            "store_nbr": [1] * n,
            "item_nbr": [1] * n,
            "unit_sales": [10.0] * (n - 1) + [9999.0] # Un seul outlier à la fin
        })
        
        result = winsorize_sales_by_group(dfv.lazy()).collect()
        
        # Le quantile 99.5% de 1000 valeurs de 10.0 (sauf une) est 10.0
        # Donc 9999.0 doit être devenu 10.0 (ou très proche)
        max_win = result["unit_sales_win"].max()
        assert max_win < 100.0, f"L'outlier n'a pas été borné : max est {max_win}"
    
    def test_no_negative_values(self, sample_train_df):
        """Vérifie qu'il n'y a pas de valeurs négatives après winsorisation."""
        result = winsorize_sales_by_group(sample_train_df.lazy()).collect()
        
        # Filtre les non-nulls
        non_null = result.filter(pl.col("unit_sales_win").is_not_null())
        if len(non_null) > 0:
            assert non_null["unit_sales_win"].min() >= 0, "Valeurs négatives détectées"


class TestLagsAndRolling:
    """Tests pour la fonction engineer_store_item_features."""
    
    def test_creates_lag_columns(self, sample_train_df):
        """Vérifie la création des colonnes Lag."""
        result = engineer_store_item_features(sample_train_df.lazy()).collect()
        
        expected_lags = ["sales_lag_16", "sales_lag_21", "sales_lag_28"]
        for col in expected_lags:
            assert col in result.columns, f"Colonne Lag manquante : {col}"
    
    def test_creates_rolling_columns(self, sample_train_df):
        """Vérifie la création des colonnes Rolling."""
        result = engineer_store_item_features(sample_train_df.lazy()).collect()
        
        expected_rolling = ["sales_roll_mean_7", "sales_roll_mean_28", "sales_roll_std_7"]
        for col in expected_rolling:
            assert col in result.columns, f"Colonne Rolling manquante : {col}"


# ==============================================================================
# TESTS DE FORMAT DE SORTIE
# ==============================================================================

class TestOutputFormats:
    """Tests pour vérifier les formats de sortie du pipeline."""
    
    def test_no_negative_sales_after_pipeline(self, sample_train_df):
        """Vérifie qu'aucune vente négative n'existe après traitement."""
        # Simulation d'un pipeline minimal
        result = winsorize_sales_by_group(sample_train_df.lazy()).collect()
        result = result.filter(pl.col("unit_sales_win").is_not_null())
        
        if len(result) > 0:
            assert result["unit_sales_win"].min() >= 0
    
    def test_required_columns_present(self, sample_train_df):
        """Vérifie que les colonnes obligatoires sont présentes."""
        result = engineer_date_features(sample_train_df.lazy()).collect()
        result = winsorize_sales_by_group(result.lazy()).collect()
        
        required = ["date", "store_nbr", "item_nbr", "unit_sales_win"]
        for col in required:
            assert col in result.columns, f"Colonne requise manquante : {col}"
    
    def test_dtypes_are_correct(self, sample_train_df):
        """Vérifie que les types de données sont corrects."""
        result = sample_train_df
        
        assert result["store_nbr"].dtype in [pl.Int64, pl.Int32]
        assert result["item_nbr"].dtype in [pl.Int64, pl.Int32]
        assert result["unit_sales"].dtype in [pl.Float64, pl.Float32, pl.Int64]


class TestTargetEncoding:
    """Tests pour la fonction perform_target_encoding."""
    
    def test_creates_encoded_columns(self, sample_train_valid_pair):
        """Vérifie la création des colonnes encodées."""
        train, valid = sample_train_valid_pair
        train_enc, valid_enc = perform_target_encoding(train, valid)
        
        # Au moins une colonne *_target_enc doit exister
        enc_cols = [c for c in train_enc.columns if c.endswith("_target_enc")]
        assert len(enc_cols) > 0, "Aucune colonne Target Encoded créée"
    
    def test_no_null_in_encoding(self, sample_train_valid_pair):
        """Vérifie qu'il n'y a pas de nulls dans les encodages."""
        train, valid = sample_train_valid_pair
        train_enc, valid_enc = perform_target_encoding(train, valid)
        
        enc_cols = [c for c in valid_enc.columns if c.endswith("_target_enc")]
        for col in enc_cols:
            null_count = valid_enc[col].null_count()
            assert null_count == 0, f"Nulls trouvés dans {col}"


class TestFeatureSelection:
    """Tests pour la fonction select_features."""
    
    def test_removes_constant_columns(self):
        """Vérifie la suppression des colonnes constantes."""
        df = pl.DataFrame({
            "var1": [1, 1, 1, 1],
            "var2": [1.0, 2.0, 3.0, 4.0],
            "constant": [0, 0, 0, 0],
        })
        
        result, dropped = select_features(df)
        
        assert "constant" not in result.columns
        assert "constant" in dropped
    
    def test_keeps_variable_columns(self):
        """Vérifie que les colonnes non-constantes sont conservées."""
        df = pl.DataFrame({
            "var1": [1.0, 2.0, 3.0, 4.0],
            "var2": [10, 20, 30, 40],
        })
        
        result, dropped = select_features(df)
        
        assert "var1" in result.columns
        assert "var2" in result.columns


# ==============================================================================
# TESTS D'INTÉGRATION (Nécessitent les vraies données)
# ==============================================================================

@pytest.mark.skipif(not is_data_available(), reason="Données complètes non disponibles")
class TestIntegration:
    """Tests d'intégration avec les vraies données."""
    
    def test_full_pipeline_runs_without_error(self):
        """Vérifie que le pipeline complet s'exécute sans erreur."""
        from preprocessing import get_dynamic_datasets
        
        # Exécution sur une période réduite
        try:
            train_df, valid_df = get_dynamic_datasets(DATA_DIR)
            assert train_df is not None
            assert valid_df is not None
            assert len(train_df) > 0
            assert len(valid_df) > 0
        except Exception as e:
            pytest.fail(f"Le pipeline a échoué : {e}")


# ==============================================================================
# POINT D'ENTRÉE POUR EXÉCUTION DIRECTE
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
