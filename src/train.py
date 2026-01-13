import os
import gc
import json
from datetime import datetime
import numpy as np
import polars as pl
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import preprocessing  # <-- On importe directement les données traitées via ce module

# ==============================================================================
# BLOC 0 : CONFIGURATION & HYPERPARAMÈTRES
# ==============================================================================
# Rôle : Définir tous les paramètres du modèle et de l'entraînement au même endroit.
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 100,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,                  # Assure la reproductibilité de l'initialisation
    "feature_fraction_seed": 42, # Reproductibilité de la sélection des features
    "bagging_seed": 42,          # Reproductibilité du bagging
    "data_random_seed": 42       # Reproductibilité des splits
}

# Training Configuration
NUM_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50
WEIGHT_PERISHABLE = 1.25
WEIGHT_NORMAL = 1.0

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME = "favorita-sales-forecasting"
MLFLOW_TRACKING_URI = "./mlruns"  # Local tracking directory


# ==============================================================================
# BLOC 1 : CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ==============================================================================
def load_preprocessed_data():
    """
    Rôle : Récupérer les DataFrames d'Entraînement et de Validation générés par preprocessing.py.
    """
    print("\n[BLOC 1] Chargement des données via le Pipeline de Prétraitement...")
    
    # 1. Calcul du chemin relatif vers /data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "..", "data")
    base_path = os.path.normpath(base_path)
    
    # 2. Appel au module preprocessing (données 2017)
    train_lf, valid_lf = preprocessing.get_dynamic_datasets(base_path)
    
    # 3. Matérialisation en RAM
    print("   > Données déjà matérialisées (DataFrame).")
    train_df = train_lf
    valid_df = valid_lf
    
    print(f"   > Train set : {train_df.shape}")
    print(f"   > Valid set : {valid_df.shape}")
    
    # Log dataset info to MLflow
    dataset_info = {
        "train_samples": train_df.shape[0],
        "valid_samples": valid_df.shape[0],
        "train_features": train_df.shape[1],
        "valid_features": valid_df.shape[1]
    }
    
    return train_df, valid_df, script_dir, dataset_info


# ==============================================================================
# BLOC 2 : PRÉPARATION DES FEATURES & TARGETS
# ==============================================================================
def prepare_features_and_target(train_df, valid_df):
    """
    Rôle : Séparer les variables explicatives (X), la cible (y) et les poids (weights).
    Target : log1p(unit_sales_win)
    """
    print("\n[BLOC 2] Préparation des Inputs (X), Targets (y) et Poids (w)...")
    
    # Colonnes à exclure de l'apprentissage (Cibles, IDs)
    cols_to_drop = ["date", "id", "unit_sales", "unit_sales_win"]
    
    # Types numériques acceptés par LightGBM
    import polars as pl
    numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8)
    
    # Filtrage : On ne garde que les colonnes numériques
    features = [
        c for c in train_df.columns 
        if c not in cols_to_drop 
        and train_df[c].dtype in numeric_types
    ]
    print(f"   > {len(features)} Features numériques sélectionnées.")
    
    excluded_cols = [c for c in train_df.columns if c not in features and c not in cols_to_drop]
    if excluded_cols:
        print(f"   > Colonnes non-numériques exclues : {excluded_cols}")
    
    # 1. Création des matrices X
    X_train = train_df.select(features).to_pandas()
    X_valid = valid_df.select(features).to_pandas()
    
    # 2. Création de la Cible y (Log transform)
    y_train = np.log1p(train_df["unit_sales_win"].to_numpy())
    y_valid = np.log1p(valid_df["unit_sales_win"].to_numpy())
    
    # 3. Création des Poids (Weighted Loss)
    if "perishable" in X_train.columns:
        print("   > Application des poids : Perishable = 1.25")
        w_train = X_train["perishable"].values * (WEIGHT_PERISHABLE - WEIGHT_NORMAL) + WEIGHT_NORMAL
        w_valid = X_valid["perishable"].values * (WEIGHT_PERISHABLE - WEIGHT_NORMAL) + WEIGHT_NORMAL
    else:
        w_train = np.ones(len(y_train))
        w_valid = np.ones(len(y_valid))
    
    feature_info = {
        "num_features": len(features),
        "features_list": features,
        "excluded_columns": excluded_cols,
        "target_transformation": "log1p"
    }
        
    return X_train, y_train, w_train, X_valid, y_valid, w_valid, feature_info


# ==============================================================================
# BLOC 3 : ENTRAÎNEMENT DU MODÈLE
# ==============================================================================
def train_lightgbm(X_train, y_train, w_train, X_valid, y_valid, w_valid):
    """
    Rôle : Entraînement avec Early Stopping.
    """
    print("\n[BLOC 3] Entraînement du modèle LightGBM...")
    
    lgb_train = lgb.Dataset(X_train, y_train, weight=w_train, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_valid, y_valid, weight=w_valid, reference=lgb_train, free_raw_data=False)
    
    evals_result = {}
    
    model = lgb.train(
        LGB_PARAMS,
        lgb_train,
        num_boost_round=NUM_BOOST_ROUNDS,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals_result)
        ]
    )
    
    # Log metrics to MLflow
    for iteration in range(len(evals_result['train']['rmse'])):
        mlflow.log_metric("train_rmse", evals_result['train']['rmse'][iteration], step=iteration)
        mlflow.log_metric("valid_rmse", evals_result['valid']['rmse'][iteration], step=iteration)
    
    mlflow.log_metric("best_iteration", model.best_iteration)
    
    return model, evals_result


# ==============================================================================
# BLOC 4 : ÉVALUATION & SAUVEGARDE
# ==============================================================================
def evaluate_and_save(model, X_valid, y_valid, w_valid, script_dir, X_train, y_train, w_train):
    """
    Rôle : Calculer NWRMSLE, MAE, RMSE et sauvegarder le modèle.
    """
    print("\n[BLOC 4] Évaluation et Sauvegarde...")
    
    # 1. Prédictions
    preds_valid = np.maximum(model.predict(X_valid, num_iteration=model.best_iteration), 0)
    preds_train = np.maximum(model.predict(X_train, num_iteration=model.best_iteration), 0)
    
    # 2. Calcul NWRMSLE
    nwrmsle_valid = np.sqrt(np.sum(w_valid * ((preds_valid - y_valid) ** 2)) / np.sum(w_valid))
    nwrmsle_train = np.sqrt(np.sum(w_train * ((preds_train - y_train) ** 2)) / np.sum(w_train))
    
    print(f"   >>> SCORE FINAL (NWRMSLE - Train) : {nwrmsle_train:.5f}")
    print(f"   >>> SCORE FINAL (NWRMSLE - Valid) : {nwrmsle_valid:.5f}")
    
    mlflow.log_metric("nwrmsle_train", nwrmsle_train)
    mlflow.log_metric("nwrmsle_valid", nwrmsle_valid)
    
    # 3. Métriques interprétables (Échelle originale)
    mae_valid = np.mean(np.abs(np.expm1(preds_valid) - np.expm1(y_valid)))
    rmse_valid = np.sqrt(np.mean((np.expm1(preds_valid) - np.expm1(y_valid)) ** 2))
    
    mlflow.log_metric("mae_valid_original_scale", mae_valid)
    mlflow.log_metric("rmse_valid_original_scale", rmse_valid)
    
    print(f"   >>> MAE (Valid - Original Scale) : {mae_valid:.2f}")
    print(f"   >>> RMSE (Valid - Original Scale) : {rmse_valid:.2f}")
    
    # 4. Sauvegarde
    save_path = os.path.join(script_dir, "model_lgbm.txt")
    model.save_model(save_path)
    print(f"   > Modèle sauvegardé dans : {save_path}")
    
    # 5. Feature Importance
    print("\n[Top 10 Variables Importantes]")
    gain_imp = model.feature_importance(importance_type='gain')
    names = model.feature_name()
    feat_imp = sorted(zip(names, gain_imp), key=lambda x: x[1], reverse=True)[:10]
    feature_importance_dict = {name: float(gain) for name, gain in zip(names, gain_imp)}
    
    for name, imp in feat_imp:
        print(f"   - {name}: {imp:.2f}")
        
    mlflow.log_dict(feature_importance_dict, "feature_importance_gain.json")
    mlflow.lightgbm.log_model(model, "model")
    
    return nwrmsle_valid


# ==============================================================================
# ORCHESTRATION PRINCIPALE
# ==============================================================================
if __name__ == "__main__": 
    
    # Setup MLflow
    if not os.path.exists(MLFLOW_TRACKING_URI):
        os.makedirs(MLFLOW_TRACKING_URI)
        
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"lgb_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        print("\n" + "="*80)
        print("🚀 DÉBUT DE L'EXPÉRIENCE (AVEC MLFLOW)")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        print("="*80)
        
        mlflow.log_params(LGB_PARAMS)
        
        # 1. Chargement
        train_df, valid_df, script_dir, dataset_info = load_preprocessed_data()
        mlflow.log_params(dataset_info)
        
        # 2. Préparation
        X_train, y_train, w_train, X_valid, y_valid, w_valid, feature_info = prepare_features_and_target(train_df, valid_df)
        mlflow.log_param("num_features", feature_info["num_features"])
        
        del train_df, valid_df
        gc.collect()
        
        # 3. Entraînement
        model, evals_result = train_lightgbm(X_train, y_train, w_train, X_valid, y_valid, w_valid)
        
        # 4. Évaluation
        evaluate_and_save(model, X_valid, y_valid, w_valid, script_dir, X_train, y_train, w_train)
        
        print("\n" + "="*80)
        print("✅ FIN DE L'EXPÉRIENCE")
        print("   Pour voir les résultats : 'mlflow ui'")
        print("="*80)