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

# Types numériques pour le filtrage des features
NUMERIC_TYPES = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, 
                 pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8)

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

def prepare_input_for_prediction(data: dict) -> pl.DataFrame:
    """
    Transforme les données brutes reçues en features prêtes pour le modèle.
    Applique le même pipeline que preprocessing.py (simplifié pour l'inférence).
    
    IMPORTANT: Doit générer exactement les 32 features attendues par le modèle.
    """
    import datetime as dt
    
    # Conversion en DataFrame Polars
    df = pl.DataFrame(data)
    
    # Parsing de la date
    if "date" in df.columns:
        df = df.with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
    
    # Ajout des features temporelles (SANS year et is_year_end qui ne sont pas dans le modèle)
    df = df.with_columns([
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.weekday().alias("day_of_week"),
    ]).with_columns([
        (pl.col("day_of_week") >= 6).cast(pl.Int32).alias("is_weekend"),
        ((pl.col("day") == 15) | (pl.col("day") >= 28)).cast(pl.Int32).alias("is_payday"),
        pl.lit(0).cast(pl.Int32).alias("is_holiday_event"),
    ])
    
    # Features Pétrole (simplifiées - en production, on utiliserait des données live)
    if "dcoilwtico" in df.columns:
        df = df.with_columns([
            pl.col("dcoilwtico").alias("oil_smooth_7d"),
            pl.col("dcoilwtico").alias("oil_lag_10"),
        ])
    
    # Feature class (placeholder - devrait être récupéré d'une table de référence)
    df = df.with_columns([
        pl.lit(1).cast(pl.Int32).alias("class"),
    ])
    
    # Features Vacances (placeholder - en production, jointure avec la table holidays)
    # ATTENTION: Utiliser n_reg et n_states_affected, PAS n_nat ni is_national_holiday
    df = df.with_columns([
        pl.lit(0).cast(pl.Int32).alias("n_events_total"),
        pl.lit(0).cast(pl.Int32).alias("n_reg"),
        pl.lit(0).cast(pl.Int32).alias("n_loc"),
        pl.lit(0).cast(pl.Int32).alias("n_states_affected"),
        pl.lit(0).cast(pl.Int32).alias("n_cities_affected"),
    ])
    
    # Features Lags (placeholder - valeurs nulles remplacées par la médiane typique)
    # En production, on récupérerait les vraies ventes passées depuis une BDD
    df = df.with_columns([
        pl.lit(5.0).alias("sales_lag_16"),
        pl.lit(5.0).alias("sales_lag_21"),
        pl.lit(5.0).alias("sales_lag_28"),
        pl.lit(5.0).alias("sales_roll_mean_7"),
        pl.lit(5.0).alias("sales_roll_mean_28"),
        pl.lit(2.0).alias("sales_roll_std_7"),
    ])
    
    # Target Encoding (placeholder - en production, on chargerait les moyennes pré-calculées)
    df = df.with_columns([
        pl.lit(2.0).alias("store_nbr_target_enc"),
        pl.lit(1.5).alias("item_nbr_target_enc"),
        pl.lit(2.0).alias("family_target_enc"),
        pl.lit(2.0).alias("city_target_enc"),
        pl.lit(2.0).alias("cluster_target_enc"),
        pl.lit(2.0).alias("type_target_enc"),
    ])
    
    return df

def filter_numeric_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filtre pour ne garder que les colonnes numériques attendues par le modèle.
    IMPORTANT: Doit correspondre exactement aux 32 features du modèle entraîné.
    """
    # Liste exacte des features attendues par le modèle (dans l'ordre)
    # IMPORTANT: Cette liste doit correspondre exactement aux features du modèle entraîné
    expected_features = [
        "store_nbr",
        "item_nbr",
        "onpromotion",
        "class",
        "perishable",
        "cluster",
        "dcoilwtico",
        "oil_smooth_7d",
        "oil_lag_10",
        "transactions",
        "n_events_total",
        "n_reg",
        "n_loc",
        "n_states_affected",
        "n_cities_affected",
        "day",
        "month",
        "day_of_week",
        "is_weekend",
        "is_payday",
        "is_holiday_event",
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_roll_mean_7",
        "sales_roll_mean_28",
        "sales_roll_std_7",
        "store_nbr_target_enc",
        "item_nbr_target_enc",
        "family_target_enc",
        "city_target_enc",
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

@app.post("/predict", response_model=PredictionOutput)
async def predict_single(input_data: PredictionInput):
    """
    Prédit les ventes pour une seule combinaison Magasin/Article/Date.
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
