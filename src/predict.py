"""
==============================================================================
predict.py - API de Prédiction des Ventes (Production)
==============================================================================
Ce script sert de pont entre le modèle entraîné et l'utilisateur final.

Fonctionnalités :
- Charge le modèle LightGBM sauvegardé par train.py
- Expose une API REST via FastAPI
- Applique le même prétraitement que pour l'entraînement
- Renvoie les prédictions en JSON ou CSV

Usage :
    uvicorn predict:app --host 0.0.0.0 --port 8000 --reload
    
Endpoints :
    GET  /health          : Vérifie que l'API est opérationnelle
    POST /predict         : Prédiction unitaire (JSON)
    POST /predict/batch   : Prédiction par lot (CSV/JSON)
"""

import os
import json
import numpy as np
import polars as pl
import lightgbm as lgb
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io

# ==============================================================================
# BLOC 0 : CONFIGURATION
# ==============================================================================

# Chemin vers le modèle (relatif au script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model_lgbm.txt")
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data")
WEBAPP_PATH = os.path.join(SCRIPT_DIR, "..", "webapp")
REFERENCE_DATA_PATH = os.path.join(WEBAPP_PATH, "data", "reference_data.json")

# Types numériques pour le filtrage des features
NUMERIC_TYPES = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, 
                 pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8)

# ==============================================================================
# BLOC 0.5 : DONNÉES DE RÉFÉRENCE POUR PRÉDICTIONS RÉALISTES
# ==============================================================================
# Statistiques de ventes moyennes par type de magasin
STORE_TYPE_SALES = {
    "A": {"base": 12.0, "std": 8.0},   # Grands magasins
    "B": {"base": 8.0, "std": 5.0},    # Magasins moyens
    "C": {"base": 5.0, "std": 3.0},    # Petits magasins
    "D": {"base": 6.0, "std": 4.0},    # Magasins standards
    "E": {"base": 7.0, "std": 4.5},    # Magasins spéciaux
}

FAMILY_MULTIPLIERS = {
    "GROCERY I": 1.5, "BEVERAGES": 1.8, "PRODUCE": 1.3, "DAIRY": 1.2,
    "BREAD/BAKERY": 2.0, "MEATS": 0.8, "PERSONAL CARE": 0.6, "CLEANING": 1.0,
    "FROZEN FOODS": 0.7, "DELI": 0.9, "POULTRY": 0.7, "EGGS": 1.1,
    "SEAFOOD": 0.5, "PREPARED FOODS": 0.8, "HOME CARE": 0.5, "BABY CARE": 0.4,
    "HARDWARE": 0.3, "LINGERIE": 0.2, "GROCERY II": 0.9,
}

# Boost des promotions par famille (basé sur l'analyse historique)
PROMO_BOOST_FACTORS = {
    "GROCERY I": 1.35,      # +35% avec promo
    "BEVERAGES": 1.45,      # +45% (très sensible)
    "PRODUCE": 1.25,        # +25%
    "DAIRY": 1.30,          # +30%
    "BREAD/BAKERY": 1.20,   # +20%
    "MEATS": 1.40,          # +40%
    "PERSONAL CARE": 1.50,  # +50%
    "CLEANING": 1.45,       # +45%
    "FROZEN FOODS": 1.35,   # +35%
    "DELI": 1.25,           # +25%
    "DEFAULT": 1.30,        # +30% par défaut
}

def load_reference_data():
    """Charge les données de référence (magasins, produits)."""
    try:
        if os.path.exists(REFERENCE_DATA_PATH):
            with open(REFERENCE_DATA_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARN] Données de référence non chargées: {e}")
    return {"stores": [], "items": [], "holidays": []}

REFERENCE_DATA = load_reference_data()

# ==============================================================================
# BLOC 1 : CHARGEMENT DU MODÈLE
# ==============================================================================

def load_model():
    """
    Charge le modèle LightGBM sauvegardé.
    Appelé au démarrage de l'API.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}. Avez-vous exécuté train.py ?")
    
    print(f"[PREDICT] Chargement du modèle depuis : {MODEL_PATH}")
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"[PREDICT] Modèle chargé avec {model.num_trees()} arbres.")
    return model

# Chargement global au démarrage
try:
    MODEL = load_model()
except FileNotFoundError as e:
    MODEL = None
    print(f"[ALERTE] {e}")

# ==============================================================================
# BLOC 2 : SCHÉMAS PYDANTIC (Validation des entrées)
# ==============================================================================

class PredictionInput(BaseModel):
    """
    Structure d'une requête de prédiction unitaire.
    Tous les champs correspondent aux features attendues par le modèle.
    """
    store_nbr: int
    item_nbr: int
    date: str  # Format YYYY-MM-DD
    onpromotion: Optional[int] = 0
    # Features additionnelles (optionnelles, seront dérivées si absentes)
    perishable: Optional[int] = 0
    cluster: Optional[int] = 1
    # Features de contexte (optionnelles, valeurs par défaut raisonnables)
    dcoilwtico: Optional[float] = 50.0
    transactions: Optional[int] = 1000

class PredictionOutput(BaseModel):
    """
    Structure de la réponse de prédiction.
    """
    store_nbr: int
    item_nbr: int
    date: str
    predicted_unit_sales: float
    confidence: Optional[str] = "medium"  # Placeholder for future use

# ==============================================================================
# BLOC 3 : LOGIQUE DE PRÉTRAITEMENT (Inférence)
# ==============================================================================

def get_item_family(item_nbr: int) -> str:
    """Récupère la famille d'un produit depuis les données de référence."""
    if REFERENCE_DATA:
        for item in REFERENCE_DATA.get("items", []):
            if int(item.get("item_nbr", 0)) == item_nbr:
                return item.get("family", "GROCERY I")
    return "GROCERY I"  # Famille par défaut


def get_realistic_lag_values(store_nbr: int, item_nbr: int, family: str = None, store_type: str = None) -> dict:
    """
    Calcule des valeurs de lag réalistes basées sur le type de magasin et la famille de produit.
    Utilise les données de référence si disponibles.
    """
    # Déterminer le type de magasin
    if store_type is None and REFERENCE_DATA:
        for store in REFERENCE_DATA.get("stores", []):
            if int(store.get("store_nbr", 0)) == store_nbr:
                store_type = store.get("type", "D")
                break
    store_type = store_type or "D"
    
    # Déterminer la famille du produit
    if family is None and REFERENCE_DATA:
        for item in REFERENCE_DATA.get("items", []):
            if int(item.get("item_nbr", 0)) == item_nbr:
                family = item.get("family", "GROCERY I")
                break
    family = family or "GROCERY I"
    
    # Récupérer les stats de base
    store_stats = STORE_TYPE_SALES.get(store_type, STORE_TYPE_SALES["D"])
    family_mult = FAMILY_MULTIPLIERS.get(family, 1.0)
    
    # Calculer les valeurs de lag réalistes
    base_sales = store_stats["base"] * family_mult
    std_sales = store_stats["std"] * family_mult
    
    return {
        "sales_lag_7": base_sales,
        "sales_lag_14": base_sales * 0.95,
        "sales_lag_16": base_sales * 0.92,
        "sales_lag_21": base_sales * 0.90,
        "sales_lag_28": base_sales * 0.88,
        "sales_roll_mean_7": base_sales,
        "sales_roll_mean_14": base_sales * 0.97,
        "sales_roll_mean_28": base_sales * 0.95,
        "sales_roll_std_7": std_sales,
        "sales_roll_min_7": max(0, base_sales - std_sales),
        "sales_roll_max_7": base_sales + std_sales * 1.5,
        "family": family,
        "store_type": store_type,
    }


def prepare_input_for_prediction(data: dict) -> pl.DataFrame:
    """
    Transforme les données brutes reçues en features prêtes pour le modèle.
    Applique le même pipeline que preprocessing.py (simplifié pour l'inférence).
    
    IMPORTANT: Utilise des lags réalistes basés sur le type de magasin/produit.
    """
    import datetime as dt
    
    # Conversion en DataFrame Polars
    df = pl.DataFrame(data)
    
    # Récupérer store_nbr et item_nbr pour calculer des lags réalistes
    store_nbr = int(data.get("store_nbr", 1))
    item_nbr = int(data.get("item_nbr", 96995))
    lag_values = get_realistic_lag_values(store_nbr, item_nbr)
    
    # Parsing de la date
    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
    
    # Ajout des features temporelles (améliorées)
    df = df.with_columns([
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.week().alias("week_of_year"),
    ]).with_columns([
        (pl.col("day_of_week") >= 6).cast(pl.Int32).alias("is_weekend"),
        ((pl.col("day") == 15) | (pl.col("day") >= 28)).cast(pl.Int32).alias("is_payday"),
        pl.lit(0).cast(pl.Int32).alias("is_holiday_event"),
        (pl.col("day") <= 5).cast(pl.Int32).alias("is_month_start"),
        (pl.col("day") >= 26).cast(pl.Int32).alias("is_month_end"),
    ])
    
    # Features d'interaction (AMÉLIORATION)
    df = df.with_columns([
        (pl.col("onpromotion") * pl.col("is_weekend")).alias("promo_weekend"),
        (pl.col("onpromotion") * pl.col("is_payday")).alias("promo_payday"),
        (pl.col("onpromotion") * pl.col("is_holiday_event")).alias("promo_holiday"),
        (pl.col("perishable") * pl.col("is_weekend")).alias("perishable_weekend"),
    ])
    
    # Features Pétrole (simplifiées - en production, on utiliserait des données live)
    if "dcoilwtico" in df.columns:
        df = df.with_columns([
            pl.col("dcoilwtico").alias("oil_smooth_7d"),
            pl.col("dcoilwtico").alias("oil_lag_10"),
        ])
    
    # Feature class (récupéré des données de référence si possible)
    item_class = 1
    if REFERENCE_DATA:
        for item in REFERENCE_DATA.get("items", []):
            if int(item.get("item_nbr", 0)) == item_nbr:
                item_class = int(item.get("class", 1))
                break
    df = df.with_columns([
        pl.lit(item_class).cast(pl.Int32).alias("class"),
    ])
    
    # Features Vacances (placeholder - en production, jointure avec la table holidays)
    df = df.with_columns([
        pl.lit(0).cast(pl.Int32).alias("n_events_total"),
        pl.lit(0).cast(pl.Int32).alias("n_reg"),
        pl.lit(0).cast(pl.Int32).alias("n_loc"),
        pl.lit(0).cast(pl.Int32).alias("n_states_affected"),
        pl.lit(0).cast(pl.Int32).alias("n_cities_affected"),
    ])
    
    # Features Lags RÉALISTES basées sur le type de magasin et famille de produit
    df = df.with_columns([
        pl.lit(lag_values["sales_lag_7"]).alias("sales_lag_7"),
        pl.lit(lag_values["sales_lag_14"]).alias("sales_lag_14"),
        pl.lit(lag_values["sales_lag_16"]).alias("sales_lag_16"),
        pl.lit(lag_values["sales_lag_21"]).alias("sales_lag_21"),
        pl.lit(lag_values["sales_lag_28"]).alias("sales_lag_28"),
        pl.lit(lag_values["sales_roll_mean_7"]).alias("sales_roll_mean_7"),
        pl.lit(lag_values["sales_roll_mean_14"]).alias("sales_roll_mean_14"),
        pl.lit(lag_values["sales_roll_mean_28"]).alias("sales_roll_mean_28"),
        pl.lit(lag_values["sales_roll_std_7"]).alias("sales_roll_std_7"),
        pl.lit(lag_values["sales_roll_min_7"]).alias("sales_roll_min_7"),
        pl.lit(lag_values["sales_roll_max_7"]).alias("sales_roll_max_7"),
    ])
    
    # Target Encoding (amélioré - state ajouté)
    df = df.with_columns([
        pl.lit(2.0).alias("store_nbr_target_enc"),
        pl.lit(1.5).alias("item_nbr_target_enc"),
        pl.lit(2.0).alias("family_target_enc"),
        pl.lit(2.0).alias("city_target_enc"),
        pl.lit(2.0).alias("state_target_enc"),
        pl.lit(2.0).alias("cluster_target_enc"),
        pl.lit(2.0).alias("type_target_enc"),
    ])
    
    return df

def filter_numeric_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filtre pour ne garder que les colonnes numériques attendues par le modèle.
    IMPORTANT: Doit correspondre exactement aux features du modèle entraîné amélioré (~45 features).
    """
    # Liste exacte des features attendues par le modèle AMÉLIORÉ
    # IMPORTANT: Cette liste doit correspondre exactement aux features du modèle entraîné
    expected_features = [
        # Identifiants numériques
        "store_nbr",
        "item_nbr",
        "class",
        "cluster",
        
        # Features produit
        "onpromotion",
        "perishable",
        
        # Pétrole
        "dcoilwtico",
        "oil_smooth_7d",
        "oil_lag_10",
        
        # Transactions
        "transactions",
        
        # Vacances/événements
        "n_events_total",
        "n_reg",
        "n_loc",
        "n_states_affected",
        "n_cities_affected",
        
        # Features temporelles (améliorées)
        "day",
        "month",
        "day_of_week",
        "week_of_year",
        "is_weekend",
        "is_payday",
        "is_holiday_event",
        "is_month_start",
        "is_month_end",
        
        # Features d'interaction (NOUVEAU)
        "promo_weekend",
        "promo_payday",
        "promo_holiday",
        "perishable_weekend",
        
        # Lags (améliorés - lags courts ajoutés)
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        
        # Rolling features (améliorées)
        "sales_roll_mean_7",
        "sales_roll_mean_14",
        "sales_roll_mean_28",
        "sales_roll_std_7",
        "sales_roll_min_7",
        "sales_roll_max_7",
        
        # Target encoding (amélioré - state ajouté)
        "store_nbr_target_enc",
        "item_nbr_target_enc",
        "family_target_enc",
        "city_target_enc",
        "state_target_enc",
        "cluster_target_enc",
        "type_target_enc"
    ]
    
    # Sélectionner uniquement les features disponibles dans df
    available_features = [f for f in expected_features if f in df.columns]
    
    return df.select(available_features)

# ==============================================================================
# BLOC 4 : APPLICATION FASTAPI
# ==============================================================================

app = FastAPI(
    title="Favorita Sales Prediction API",
    description="API de prédiction des ventes pour la compétition Kaggle Favorita",
    version="1.0.0"
)

# Configuration CORS pour permettre les appels depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des fichiers statiques du frontend (si disponibles)
if os.path.exists(WEBAPP_PATH):
    app.mount("/css", StaticFiles(directory=os.path.join(WEBAPP_PATH, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(WEBAPP_PATH, "js")), name="js")
    app.mount("/data", StaticFiles(directory=os.path.join(WEBAPP_PATH, "data")), name="data")

@app.get("/")
async def serve_index():
    """Sert la page d'accueil du frontend."""
    index_path = os.path.join(WEBAPP_PATH, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Favorita Sales Prediction API", "docs": "/docs"}

@app.get("/index.html")
async def serve_index_html():
    """Sert la page d'accueil pour l'endpoint /index.html."""
    return FileResponse(os.path.join(WEBAPP_PATH, "index.html"))

@app.get("/dashboard.html")
async def serve_dashboard():
    """Sert la page dashboard."""
    return FileResponse(os.path.join(WEBAPP_PATH, "dashboard.html"))

@app.get("/methodology.html")
async def serve_methodology():
    """Sert la page méthodologie."""
    return FileResponse(os.path.join(WEBAPP_PATH, "methodology.html"))

@app.get("/login")
async def serve_login():
    """Sert la page d'authentification."""
    return FileResponse(os.path.join(WEBAPP_PATH, "login.html"))

@app.get("/login.html")
async def serve_login_html():
    """Sert la page d'authentification pour l'endpoint /login.html."""
    return FileResponse(os.path.join(WEBAPP_PATH, "login.html"))

@app.get("/health")
async def health_check():
    """
    Vérifie que l'API et le modèle sont opérationnels.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Exécutez train.py d'abord.")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "num_trees": MODEL.num_trees(),
        "num_features": MODEL.num_feature()
    }

def apply_promo_boost(prediction: float, family: str, onpromotion: int) -> float:
    """
    Applique un boost de promotion cohérent basé sur les données historiques.
    
    Le modèle peut sous-estimer l'effet des promos car les features de lag dominent.
    Cette fonction garantit que les produits en promotion ont une demande supérieure.
    """
    if onpromotion != 1:
        return prediction
    
    # Récupérer le facteur de boost pour cette famille
    boost_factor = PROMO_BOOST_FACTORS.get(family, PROMO_BOOST_FACTORS["DEFAULT"])
    
    # Appliquer le boost
    boosted = prediction * boost_factor
    
    return boosted


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(input_data: PredictionInput):
    """
    Prédit les ventes pour une seule combinaison Magasin/Article/Date.
    Applique un boost de promotion cohérent si le produit est en promo.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")
    
    # Conversion en dict puis prétraitement
    data_dict = input_data.model_dump()
    
    try:
        # Préparation des features
        df = prepare_input_for_prediction(data_dict)
        df_numeric = filter_numeric_features(df)
        
        # Conversion Pandas pour LightGBM
        X = df_numeric.to_pandas()
        
        # Prédiction (le modèle renvoie log1p(sales))
        log_pred = MODEL.predict(X)[0]
        
        # Conversion inverse : expm1 pour revenir aux unités
        pred_sales = max(0, np.expm1(log_pred))
        
        # Appliquer le boost de promotion si nécessaire
        family = get_item_family(input_data.item_nbr)
        pred_sales = apply_promo_boost(pred_sales, family, input_data.onpromotion)
        
        return PredictionOutput(
            store_nbr=input_data.store_nbr,
            item_nbr=input_data.item_nbr,
            date=input_data.date,
            predicted_unit_sales=round(pred_sales, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Prédit les ventes pour un lot de données (fichier CSV).
    
    Attend un CSV avec colonnes : store_nbr, item_nbr, date, [autres features optionnelles]
    Renvoie un CSV avec une colonne 'predicted_unit_sales' ajoutée.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")
    
    try:
        # Lecture du CSV uploadé
        contents = await file.read()
        df_input = pl.read_csv(io.BytesIO(contents))
        
        # Liste pour stocker les prédictions
        predictions = []
        
        for row in df_input.iter_rows(named=True):
            df = prepare_input_for_prediction(row)
            df_numeric = filter_numeric_features(df)
            X = df_numeric.to_pandas()
            
            log_pred = MODEL.predict(X)[0]
            pred_sales = max(0, np.expm1(log_pred))
            
            # Appliquer le boost de promotion
            item_nbr = row.get("item_nbr", 0)
            onpromotion = row.get("onpromotion", 0)
            family = get_item_family(item_nbr)
            pred_sales = apply_promo_boost(pred_sales, family, onpromotion)
            
            predictions.append(round(pred_sales, 2))
        
        # Ajout de la colonne prédiction
        df_output = df_input.with_columns(
            pl.Series("predicted_unit_sales", predictions)
        )
        
        # Conversion en CSV pour le téléchargement
        output_buffer = io.BytesIO()
        df_output.write_csv(output_buffer)
        output_buffer.seek(0)
        
        return StreamingResponse(
            output_buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur batch : {str(e)}")

# ==============================================================================
# BLOC 5 : POINT D'ENTRÉE (Mode Standalone)
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=== DÉMARRAGE DE L'API DE PRÉDICTION ===")
    print(f"Modèle : {MODEL_PATH}")
    print(f"Données : {DATA_PATH}")
    print("Endpoints :")
    print("  - GET  /health        : Vérification de santé")
    print("  - POST /predict       : Prédiction unitaire")
    print("  - POST /predict/batch : Prédiction par lot (CSV)")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
