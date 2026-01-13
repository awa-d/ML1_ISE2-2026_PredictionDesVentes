import os
import glob
import sys
import types
import polars as pl
import numpy as np
import datetime as dt
from datetime import timedelta

# ==============================================================================
# PARTIE 1 : CONFIGURATION ET UTILITAIRES DE BASE
# ==============================================================================

def find_csv(base_path: str, filename: str) -> str:
    """
    Localise un fichier CSV récursivement dans un chemin de base.
    """
    search_pattern = os.path.join(base_path, "**", filename)
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"{filename} introuvable dans {base_path}")
    return files[0]

def get_data_paths(base_path: str) -> dict:
    """
    Retourne un dictionnaire des chemins de fichiers pour les jeux de données.
    """
    return {
        "train": find_csv(base_path, "train.csv"),
        "test": find_csv(base_path, "test.csv"),
        "items": find_csv(base_path, "items.csv"),
        "stores": find_csv(base_path, "stores.csv"),
        "oil": find_csv(base_path, "oil.csv"),
        "holidays": find_csv(base_path, "holidays_events.csv"),
        "transactions": find_csv(base_path, "transactions.csv"),
    }

# ==============================================================================
# PARTIE 2 : FONCTIONS DE NETTOYAGE ET STANDARDISATION
# ==============================================================================

def clean_date_column(lf: pl.LazyFrame, date_col: str = "date") -> pl.LazyFrame:
    """
    S'assure que la colonne date est bien typée (pl.Date) et propre.
    """
    schema = lf.collect_schema()
    dtype = schema.get(date_col)

    if dtype == pl.Date:
        date_expr = pl.col(date_col)
    elif dtype is not None and "Datetime" in str(dtype):
        date_expr = pl.col(date_col).dt.date()
    else:
        date_expr = (
            pl.col(date_col)
            .cast(pl.Utf8)
            .str.slice(0, 10)
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
    
    return lf.with_columns(date_expr.alias(date_col))

# ==============================================================================
# PARTIE 4-BIS : FEATURE ENGINEERING AVANCE
# ==============================================================================

def engineer_date_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ajoute les variables temporelles : Jour, Mois, Année, JourSem, Payday, Fin d'année.
    """
    return (
        lf.with_columns([
            pl.col("date").dt.day().alias("day"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.weekday().alias("day_of_week"),
            pl.col("date").dt.week().alias("week_of_year"),  # AMÉLIORATION: Semaine de l'année
        ])
        .with_columns([
            (pl.col("day_of_week") >= 6).cast(pl.Int32).alias("is_weekend"),
            # Payday : 15 ou fin de mois (approx)
            ((pl.col("day") == 15) | (pl.col("day") == pl.col("day").max().over("month"))) 
            .cast(pl.Int32).alias("is_payday"),
            # Fin d'année (Noël/Nouvel An)
            ((pl.col("month") == 12) & (pl.col("day") >= 20)).cast(pl.Int32).alias("is_year_end"),
            # Début de mois (1-5) - AMÉLIORATION
            (pl.col("day") <= 5).cast(pl.Int32).alias("is_month_start"),
            # Fin de mois (26-31) - AMÉLIORATION  
            (pl.col("day") >= 26).cast(pl.Int32).alias("is_month_end"),
        ])
    )

def engineer_interaction_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    AMÉLIORATION: Ajoute des features d'interaction entre promotions et contexte temporel.
    Ces interactions capturent des effets non-linéaires importants.
    """
    return lf.with_columns([
        # Interaction Promo × Weekend (les promos ont plus d'impact le weekend)
        (pl.col("onpromotion") * pl.col("is_weekend")).alias("promo_weekend"),
        # Interaction Promo × Payday (les promos ont plus d'impact les jours de paie)
        (pl.col("onpromotion") * pl.col("is_payday")).alias("promo_payday"),
        # Interaction Promo × Jour férié
        (pl.col("onpromotion") * pl.col("is_holiday_event")).alias("promo_holiday"),
        # Interaction Perishable × Weekend (produits périssables vendus différemment le WE)
        (pl.col("perishable") * pl.col("is_weekend")).alias("perishable_weekend"),
    ])

def engineer_oil_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ajoute le lissage (Rolling Mean 7j) et le Lag (10j) sur le prix du pétrole.
    Suppose que 'dcoilwtico' existe déjà.
    """
    # Note: Rolling et Lag nécessitent un tri par date,
    # Mais ici on le fait sur une série temporelle simple (Oil) avant jointure
    return lf.sort("date").with_columns([
        pl.col("dcoilwtico").rolling_mean(window_size=7, min_periods=1).alias("oil_smooth_7d"),
        pl.col("dcoilwtico").shift(10).alias("oil_lag_10")
    ])

def engineer_holiday_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Raffine les features de vacances : 
    - Type encodé
    - Flag 'is_closed' (Basé sur règles manuelles ou données manquantes)
    """
    # Ici, on enrichit simplement avec des indicateurs binaires forts
    # La logique 'locale' est déjà gérée dans process_holidays
    return lf.with_columns([
        (pl.col("n_events_total") > 0).cast(pl.Int32).alias("is_holiday_event"),
        # Si Nationale, impact fort
        (pl.col("n_nat") > 0).cast(pl.Int32).alias("is_national_holiday")
    ])

def engineer_store_item_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ajoute Lags et Moyennes Mobiles sur les ventes.
    Attention : Nécessite des données triées par Date, et groupées par Store/Item.
    """
    # Définition des fenêtres
    # On calcule ça sur unit_sales (ou log1p si on avait transformé)
    
    # Lag 16 jours (le minimum pour Kaggle test set)
    # Rolling 7, 28 jours
    
    # Note : shift/rolling en Lazy sur partitions (over)
    groups = ["store_nbr", "item_nbr"]
    return lf.sort("date").with_columns([ # Sort est crucial
        # Lags courts (7, 14 jours) - AMÉLIORATION
        pl.col("unit_sales").shift(7).over(groups).alias("sales_lag_7"),
        pl.col("unit_sales").shift(14).over(groups).alias("sales_lag_14"),
        # Lags originaux
        pl.col("unit_sales").shift(16).over(groups).alias("sales_lag_16"),
        pl.col("unit_sales").shift(21).over(groups).alias("sales_lag_21"),
        pl.col("unit_sales").shift(28).over(groups).alias("sales_lag_28"),
        
        # Rolling means
        pl.col("unit_sales").rolling_mean(7).over(groups).alias("sales_roll_mean_7"),
        pl.col("unit_sales").rolling_mean(14).over(groups).alias("sales_roll_mean_14"),
        pl.col("unit_sales").rolling_mean(28).over(groups).alias("sales_roll_mean_28"),
        
        # Rolling std
        pl.col("unit_sales").rolling_std(7).over(groups).alias("sales_roll_std_7"),
        
        # Rolling min/max pour capturer les extremes - AMÉLIORATION
        pl.col("unit_sales").rolling_min(7).over(groups).alias("sales_roll_min_7"),
        pl.col("unit_sales").rolling_max(7).over(groups).alias("sales_roll_max_7"),
    ])

    return lf.with_columns(exprs)

def scale_numerical_features_eager(train_df: pl.DataFrame, valid_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    StandardScaler (Fit sur Train, Transform sur Train & Valid).
    Opère sur des DataFrames collectés.
    """
    exclude_cols = ["id", "date", "store_nbr", "item_nbr", "unit_sales", "unit_sales_win", 
                    "is_payday", "is_weekend", "is_year_end", "is_holiday_event", "is_national_holiday",
                    "sales_lag_16", "sales_lag_21", "sales_lag_28"] # On ne scale pas forcément les lags bruts, mais bon.
    # Excluons aussi les colonnes cibles ou ID
    
    numeric_cols = [
        col for col, dtype in train_df.schema.items() 
        if col not in exclude_cols 
        and (dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32])
    ]
    
    print(f"Scaling de {len(numeric_cols)} features...")
    
    # Calcul des stats sur TRAIN
    means = train_df.select([pl.col(c).mean().alias(f"{c}_mean") for c in numeric_cols])
    stds = train_df.select([pl.col(c).std().alias(f"{c}_std") for c in numeric_cols])
    
    # Construction des expressions de scaling
    # (col - mean) / std
    # Pour faire ça efficacement en Polars eager sans itérer trop lentement:
    # On peut utiliser with_columns avec des lit.
    
    # Astuce : On convertit les stats en dict pour accès rapide
    stats_dict = {}
    for c in numeric_cols:
        m = means[f"{c}_mean"][0]
        s = stds[f"{c}_std"][0]
        stats_dict[c] = (m, s)
        
    def get_scale_exprs(numeric_columns):
        return [
            ((pl.col(c) - stats_dict[c][0]) / (stats_dict[c][1] + 1e-8)).alias(c)
            for c in numeric_columns
        ]

    train_scaled = train_df.with_columns(get_scale_exprs(numeric_cols))
    valid_scaled = valid_df.with_columns(get_scale_exprs(numeric_cols))
    
    return train_scaled, valid_scaled

def select_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """
    Sélection des features : Suppression des colonnes constantes.
    Retourne le DataFrame nettoyé ET la liste des colonnes supprimées.
    """
    # Variance Threshold simple (si min == max)
    drops = []
    for col in df.columns:
        if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            if df[col].min() == df[col].max():
                drops.append(col)
                
    if drops:
        print(f"Suppression de {len(drops)} features constantes : {drops}")
        return df.drop(drops), drops
    return df, []

def perform_target_encoding(train_df: pl.DataFrame, valid_df: pl.DataFrame) -> tuple:
    """
    Applique le Target Encoding sur les colonnes catégorielles.
    Apprend sur TRAIN, applique sur TRAIN et VALID.
    """
    # Colonnes à encoder - AMÉLIORATION: Ajout de 'state' (région/province)
    cat_cols = ["store_nbr", "item_nbr", "family", "city", "state", "cluster", "type"]
    
    # Si les colonnes n'existent pas (car dans items/stores), on suppose qu'elles sont jointes.
    available_cols = [c for c in cat_cols if c in train_df.columns]
    
    train_encoded = train_df.lazy()
    valid_encoded = valid_df.lazy()
    
    for col in available_cols:
        # Calcul de la moyenne sur TRAIN
        # Utilisation de log1p pour lisser
        mean_enc = train_df.group_by(col).agg(
            pl.col("unit_sales").log1p().mean().alias(f"{col}_target_enc")
        )
        
        # Join sur Train et Valid
        train_encoded = train_encoded.join(mean_enc.lazy(), on=col, how="left")
        valid_encoded = valid_encoded.join(mean_enc.lazy(), on=col, how="left")
        
        # Fill nulls avec la moyenne globale (pour les catégories inconnues dans valid)
        global_mean = train_df["unit_sales"].log1p().mean()
        train_encoded = train_encoded.with_columns(pl.col(f"{col}_target_enc").fill_null(global_mean))
        valid_encoded = valid_encoded.with_columns(pl.col(f"{col}_target_enc").fill_null(global_mean))
        
    return train_encoded.collect(), valid_encoded.collect()

def winsorize_sales_by_group(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Applique une winsorisation (clipping) sur unit_sales par groupe (Magasin, Produit).
    Seuils : q05 (5%) et q995 (99.5%).
    Crée une nouvelle colonne 'unit_sales_win'.
    """
    # Définition des clés de groupe
    keys = ["store_nbr", "item_nbr"]
    
    # Calcul des quantiles par groupe (Window Function)
    q05 = pl.col("unit_sales").quantile(0.05).over(keys)
    q995 = pl.col("unit_sales").quantile(0.995).over(keys)
    
    # Application du clipping
    # On utilise min/max horizontal pour borner la valeur
    return lf.with_columns(
        pl.min_horizontal(
            pl.max_horizontal(pl.col("unit_sales"), q05), # >= q05
            q995                                          # <= q995
        ).alias("unit_sales_win")
    )

# ==============================================================================
# PARTIE 3 : TRAITEMENT DES DONNÉES AUXILIAIRES (JOURS FÉRIÉS, PÉTROLE)
# ==============================================================================

def process_holidays(holidays_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Prétraitement des jours fériés : agrégation par date (National, Régional, Local).
    """
    hol = holidays_lf.collect()
    hol = clean_date_column(pl.LazyFrame(hol)).collect()

    # Suppression des jours fériés qui ont été transférés (non chômés ce jour-là)
    if "transferred" in hol.columns:
         hol = hol.filter(pl.col("transferred").cast(pl.Boolean, strict=False) == False)

    # Création des compteurs d'événements par date
    summary = (
        hol.group_by("date")
           .agg([
               pl.len().alias("n_events_total"),
               (pl.col("locale") == "National").cast(pl.Int32).sum().alias("n_nat"),
               (pl.col("locale") == "Regional").cast(pl.Int32).sum().alias("n_reg"),
               (pl.col("locale") == "Local").cast(pl.Int32).sum().alias("n_loc"),
           ])
    )

    # Calcul du nombre d'états et de villes impactés par des événements régionaux/locaux
    n_states = (
        hol.filter(pl.col("locale") == "Regional")
           .group_by("date")
           .agg(pl.col("locale_name").n_unique().alias("n_states_affected"))
    )

    n_cities = (
        hol.filter(pl.col("locale") == "Local")
           .group_by("date")
           .agg(pl.col("locale_name").n_unique().alias("n_cities_affected"))
    )

    summary = (
        summary.join(n_states, on="date", how="left")
               .join(n_cities, on="date", how="left")
               .with_columns([
                   pl.col("n_states_affected").fill_null(0).cast(pl.Int32),
                   pl.col("n_cities_affected").fill_null(0).cast(pl.Int32),
               ])
               .sort("date")
    )
    
    return summary

def process_oil_price(oil_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Traitement des prix du pétrole : remplissage des jours manquants par interpolation.
    """
    oil = oil_lf.collect()
    oil = clean_date_column(pl.LazyFrame(oil)).collect()
    
    # Création d'un axe temporel complet pour ne pas avoir de trous
    min_date = oil["date"].min()
    max_date = oil["date"].max()
    
    full_index = pl.DataFrame({
        "date": pl.date_range(min_date, max_date, interval="1d", eager=True)
    })
    
    # Jointure et interpolation linéaire
    oil_interpolated = (
        full_index.join(oil, on="date", how="left")
        .with_columns(
            pl.col("dcoilwtico")
              .cast(pl.Float64)
              .interpolate()
              .fill_null(strategy="forward")
              .fill_null(strategy="backward")
              .alias("dcoilwtico")
        )
    )
    return oil_interpolated

def process_oil_price_advanced(oil_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Version avancée avec lissage et lag.
    """
    # On réutilise la base
    df = process_oil_price(oil_lf) 
    # Et on applique l'engineer
    return engineer_oil_features(df.lazy()).collect()

# ==============================================================================
# PARTIE 4 : CHARGEMENT ET FUSION DU PIPELINE
# ==============================================================================

def load_and_preprocess_data(base_path: str, date_range: tuple = None, filter_year: int = None):
    """
    Charge toutes les données, applique les filtres temporels et réalise les jointures.
    """
    paths = get_data_paths(base_path)
    
    # 1. Chargement paresseux (Lazy) des CSV pour optimiser la RAM
    train_lf = pl.scan_csv(paths["train"], ignore_errors=True)
    items_lf = pl.scan_csv(paths["items"], ignore_errors=True)
    stores_lf = pl.scan_csv(paths["stores"], ignore_errors=True)
    oil_lf = pl.scan_csv(paths["oil"], ignore_errors=True)
    holidays_lf = pl.scan_csv(paths["holidays"], ignore_errors=True)
    transactions_lf = pl.scan_csv(paths["transactions"], ignore_errors=True)
    
    # 2. Nettoyage de la date dans la table principale
    train_lf = clean_date_column(train_lf)
    
    # 3. Filtrage temporel (Réduction drastique du volume de données)
    if date_range:
        start_date, end_date = date_range
        # Conversion string -> date
        s_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        e_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        
        train_lf = train_lf.filter(
            (pl.col("date") >= s_date) & 
            (pl.col("date") <= e_date)
        )
    elif filter_year:
        # Filtrage par année si pas de plage précise
        train_lf = train_lf.filter(pl.col("date").dt.year() == filter_year)
        
    # Filtrer les ventes négatives (retours)
    train_lf = train_lf.filter(pl.col("unit_sales") >= 0)
    
    # Conversion de onpromotion (boolean/string -> int)
    # La colonne peut être 'true'/'false' (string) ou True/False (boolean)
    train_lf = train_lf.with_columns(
        pl.when(pl.col("onpromotion").cast(pl.Utf8).str.to_lowercase() == "true")
        .then(1)
        .when(pl.col("onpromotion").cast(pl.Utf8).str.to_lowercase() == "false")
        .then(0)
        .otherwise(pl.col("onpromotion").cast(pl.Int32))
        .fill_null(0)
        .alias("onpromotion")
    )

    # 4. Préparation des tables auxiliaires
    items = items_lf.collect()
    stores = stores_lf.collect()
    transactions = clean_date_column(transactions_lf).collect()
    
    holiday_features = process_holidays(holidays_lf)
    # Utilisation de la version avancée pour l'huile
    oil_features = process_oil_price_advanced(oil_lf)
    
    # 5. Jointures finales
    # On reste en mode Lazy le plus longtemps possible
    train_joined = (
        train_lf
        .join(items.lazy(), on="item_nbr", how="left")
        .join(stores.lazy(), on="store_nbr", how="left")
        .join(oil_features.lazy(), on="date", how="left")
        .join(transactions.lazy(), on=["date", "store_nbr"], how="left")
        .join(holiday_features.lazy(), on="date", how="left")
    )
    
    # Remplacement des valeurs nulles par 0 pour les colonnes "événements"
    holiday_cols = ["n_events_total", "n_nat", "n_reg", "n_loc", "n_states_affected", "n_cities_affected"]
    train_joined = train_joined.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Int32) for c in holiday_cols]
    )
    
    # 6. Winsorisation (Traitement des outliers)
    train_joined = winsorize_sales_by_group(train_joined)

    # 7. Basic Date Features (Part 1 du FE)
    train_joined = engineer_date_features(train_joined)

    # 8. Holiday Features Avancés
    train_joined = engineer_holiday_features(train_joined)
    
    # 9. Lags & Rolling (Part 2 du FE - Stateful)
    # Attention, cela nécessite tout l'historique dispo dans le LazyFrame
    train_joined = engineer_store_item_features(train_joined)
    
    # 10. Features d'interaction (AMÉLIORATION)
    # Doit être appelé après onpromotion, is_weekend, is_payday, is_holiday_event
    train_joined = engineer_interaction_features(train_joined)
    
    return train_joined

# ==============================================================================
# PARTIE 5 : GESTION DYNAMIQUE (FENÊTRES GLISSANTES)
# ==============================================================================

def get_latest_available_date(base_path: str) -> dt.date:
    """
    Récupère la toute dernière date disponible dans les données d'entraînement.
    """
    paths = get_data_paths(base_path)
    try:
        # Scan rapide (Lazy) pour trouver le max
        lf = pl.scan_csv(paths["train"], n_rows=1000000000) 
        return clean_date_column(lf).select(pl.col("date").max()).collect().item()
    except:
        # Date par défaut (fin 2017) si échec
        return dt.date(2017, 8, 15)

def get_dynamic_datasets(base_path: str, reference_date: str = None, 
                         train_weeks: int = 7, valid_weeks: int = 2, gap_days: int = 7):
    """
    Génère automatiquement les jeux Train et Validation en fonction de la date actuelle des données.
    
    Mécanique :
    1. Identifie la dernière date (ref_dt).
    2. Calcule :
       - VALIDATION = [ref_dt - 2 semaines, ref_dt]
       - TRAIN      = [ref_dt - 2 sem - 7 jours - 7 sem, ref_dt - 2 sem - 7 jours]
    """
    
    if reference_date:
        ref_dt = dt.datetime.strptime(reference_date, "%Y-%m-%d").date()
        print(f"Mode : Date de référence fixée manuellement à {ref_dt}")
    else:
        ref_dt = get_latest_available_date(base_path)
        print(f"Mode Dynamique : Dernière date détectée = {ref_dt}")

    # Calcul des bornes temporelles
    valid_end = ref_dt
    valid_start = valid_end - timedelta(weeks=valid_weeks)
    
    train_end = valid_start - timedelta(days=gap_days)
    train_start = train_end - timedelta(weeks=train_weeks)
    
    # Formatage string pour les filtres
    str_train_start = train_start.strftime("%Y-%m-%d")
    str_train_end = train_end.strftime("%Y-%m-%d")
    str_valid_start = valid_start.strftime("%Y-%m-%d")
    str_valid_end = valid_end.strftime("%Y-%m-%d")
    
    print(f"--- Configuration des Fenêtres Glissantes ---")
    print(f"Train     : {str_train_start} -> {str_train_end} ({train_weeks} semaines)")
    print(f"Gap       : {gap_days} jours (Simulation délai prod)")
    print(f"Valid     : {str_valid_start} -> {str_valid_end} ({valid_weeks} semaines)")
    print(f"---------------------------------------------")
    
    # 1. TRAIN
    print("> Génération du jeu d'ENTRAÎNEMENT...")
    train_lf = load_and_preprocess_data(base_path, date_range=(str_train_start, str_train_end))
    train_df = train_lf.collect() # Collecte pour le Feature Engineering Eager
    print(f"  -> Train Raw Dimensions : {train_df.shape}")
    
    # 2. VALIDATION (Avec Lookback pour les Lags)
    print("> Génération du jeu de VALIDATION...")
    # On recule de 35 jours le début pour avoir l'historique nécessaire aux Rolling/Lags
    lookback_days = 35 
    valid_start_extended = valid_start - timedelta(days=lookback_days)
    str_valid_start_ext = valid_start_extended.strftime("%Y-%m-%d")
    
    valid_lf_ext = load_and_preprocess_data(base_path, date_range=(str_valid_start_ext, str_valid_end))
    valid_df_ext = valid_lf_ext.collect()
    
    # Maintenant on recolle à la vraie période de validation
    # (On supprime l'historique de chauffe)
    valid_s_date = dt.datetime.strptime(str_valid_start, "%Y-%m-%d").date()
    valid_df = valid_df_ext.filter(pl.col("date") >= valid_s_date)
    
    print(f"  -> Valid Raw Dimensions : {valid_df.shape} (Après coupe du lookback)")

    # ==============================================================================
    # PHASE STATEFUL : Target Encoding, Scaling, Selection
    # ==============================================================================
    
    print("--- Phase Feature Engineering Stateful ---")
    
    # A. Target Encoding
    print("(A) Target Encoding...")
    train_df, valid_df = perform_target_encoding(train_df, valid_df)
    
    # B. Scaling
    print("(B) Feature Scaling (StandardScaler)...")
    train_df, valid_df = scale_numerical_features_eager(train_df, valid_df)
    
    # C. Feature Selection
    # C. Feature Selection
    print("(C) Feature Selection (Variance)...")
    # On calcule les colonnes à supprimer UNIQUEMENT sur le Train
    train_df, dropped_cols = select_features(train_df)
    
    # Et on applique exactement la même suppression sur le Valid
    # (Pour garantir que les schémas restent identiques)
    if dropped_cols:
        valid_df = valid_df.drop(dropped_cols)
    
    print("--- Fin du Pipeline ---")
    return train_df, valid_df

# ==============================================================================
# PARTIE TÉLÉCHARGEMENT : INTERFACE KAGGLE / OPENDATASETS
# ==============================================================================

def download_data(dataset_url: str = "https://www.kaggle.com/datasets/ruiyuanfan/corporacin-favorita-grocery-sales-forecasting", path: str = "data"):
    """
    Télécharge le dataset via la librairie opendatasets.
    Gère le patch de compatibilité pour Python 3.13 (module cgi manquant).
    """
    
    # Patch de compatibilité pour 'cgi' manquant dans Python 3.13
    if sys.version_info >= (3, 13):
        if "cgi" not in sys.modules:
            dummy_cgi = types.ModuleType("cgi")
            def parse_header(line):
                parts = line.split(";")
                key = parts[0].strip()
                pdict = {}
                for part in parts[1:]:
                   if "=" in part:
                       k, v = part.split("=", 1)
                       pdict[k.strip()] = v.strip().strip('"')
                return key, pdict
            dummy_cgi.parse_header = parse_header
            sys.modules["cgi"] = dummy_cgi

    import opendatasets as od
    import json
    
    # Configuration des identifiants Kaggle (Projet Scolaire)
    # Pour éviter la demande interactive
    kaggle_creds = {
        "username": "aliounekane",
        "key": "KGAT_afd9210cc83889bdff43ff3ad6ec2f37"
    }
    
    # Création du fichier kaggle.json dans le dossier courant pour opendatasets
    if not os.path.exists("kaggle.json"):
        with open("kaggle.json", "w") as f:
            json.dump(kaggle_creds, f)
        print("Fichier 'kaggle.json' créé automatiquement.")
    
    # Création du dossier cible
    if not os.path.exists(path):
        os.makedirs(path)
        
    print(f"Démarrage du téléchargement depuis : {dataset_url}")
    print(f"Destination : {path}")
    try:
        # opendatasets gère l'authentification interactive ou via kaggle.json
        od.download(dataset_url, data_dir=path)
        print("Téléchargement réussi avec succès.")
    except Exception as e:
        print(f"Erreur critique lors du téléchargement : {e}")
        raise e

# ==============================================================================
# MAIN : POINT D'ENTRÉE DU SCRIPT
# ==============================================================================

if __name__ == "__main__":
    
    # Définition du chemin des données RELATIF au script
    # On utilise __file__ pour être robuste quel que soit le dossier d'exécution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # La structure attendue est :
    # root/
    #   src/
    #     preprocessing.py
    #   data/
    BASE_PATH = os.path.join(script_dir, "..", "data")
    
    # Normalisation du chemin (pour éviter les ..\data)
    BASE_PATH = os.path.normpath(BASE_PATH)

    print(f"=== DÉMARRAGE DU PIPELINE DE PRÉTRAITEMENT ===")
    print(f"Script exécuté depuis : {script_dir}")
    print(f"Dossier de données ciblé (Relatif) : {BASE_PATH}")

    try:
        # Essai direct : on suppose que les données sont là
        train_data, valid_data = get_dynamic_datasets(BASE_PATH)
        print("=== SUCCÈS : Données traitées et prêtes ===")
        
    except FileNotFoundError as e:
        print(f"ALERTE : Données manquantes ({e})")
        print(">>> Lancement du module de téléchargement...")
        
        try:
            download_data(path=BASE_PATH)
            
            # Re-tentative après téléchargement
            print(">>> Reprise du traitement après téléchargement...")
            train_data, valid_data = get_dynamic_datasets(BASE_PATH)
            print("=== SUCCÈS : Données traitées et prêtes apres téléchargement ===")
            
        except Exception as inner_e:
            print(f"ÉCHEC FATAL : Impossible de récupérer ou traiter les données. Erreur : {inner_e}")
            
    except Exception as e:
        print(f"ERREUR INATTENDUE : {e}")
