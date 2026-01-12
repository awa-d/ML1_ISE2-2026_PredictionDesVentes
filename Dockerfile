# ==============================================================================
# Dockerfile - Favorita Sales Forecasting API
# ==============================================================================
# Ce Dockerfile crée une image légère pour déployer l'API de prédiction.
#
# Build:   docker build -t favorita-api .
# Run:     docker run -p 8000:8000 favorita-api
# ==============================================================================

# --- Stage 1: Base Image ---
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Alioune Kane"
LABEL project="Favorita Grocery Sales Forecasting - ENSAE ISE2 2026"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FAVORITA_TEST_MODE=0

# Répertoire de travail dans le conteneur
WORKDIR /app

# --- Stage 2: Installation des dépendances ---
# On copie d'abord requirements.txt pour profiter du cache Docker
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 3: Copie du code source ---
# On copie uniquement les fichiers nécessaires à l'API
COPY src/predict.py ./src/
COPY src/preprocessing.py ./src/
COPY src/__init__.py ./src/
COPY src/model_lgbm.txt ./src/

# Création du dossier data (vide, sera monté en volume si besoin)
RUN mkdir -p /app/data

# --- Stage 4: Configuration du serveur ---
# Port exposé par l'API
EXPOSE 8000

# Commande de démarrage
# --host 0.0.0.0 permet d'accepter les connexions externes au conteneur
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Notes de Déploiement:
# ==============================================================================
# 
# 1. Pour construire l'image:
#    docker build -t favorita-api .
#
# 2. Pour lancer le conteneur:
#    docker run -d -p 8000:8000 --name favorita favorita-api
#
# 3. Pour monter les données (optionnel):
#    docker run -d -p 8000:8000 -v /chemin/local/data:/app/data favorita-api
#
# 4. Pour voir les logs:
#    docker logs favorita
#
# 5. Accès à l'API:
#    - Santé:   http://localhost:8000/health
#    - Swagger: http://localhost:8000/docs
#    - Predict: POST http://localhost:8000/predict
# ==============================================================================
