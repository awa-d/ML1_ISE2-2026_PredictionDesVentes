# 🛒 Favorita Grocery Sales Forecasting

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Deploy](https://img.shields.io/badge/Deploy-Render-purple.svg)](https://render.com/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-orange.svg)](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

## 📝 Présentation du Projet
Ce projet est réalisé dans le cadre de la formation **ISE à l'ENSAE**. L'objectif est de prédire les ventes unitaires de milliers d'articles vendus dans les magasins **Favorita** (une grande enseigne équatorienne).

La gestion optimale des stocks est un enjeu crucial : une prévision précise permet d'éviter les ruptures de stock tout en minimisant le gaspillage alimentaire et les coûts de stockage.

## 🎯 Objectifs
- Analyser les tendances temporelles et l'impact des facteurs externes (promotions, jours fériés, prix du pétrole).
- Développer un modèle de Machine Learning robuste pour la prévision de séries temporelles.
- Mettre en place un pipeline complet : de l'exploration des données à l'exposition du modèle via une API.

---

## 🏗️ Méthodologie & Architecture
Le projet suit une approche structurée, comparable à la construction d'un édifice :

1.  **Analyse Exploratoire (EDA) :** Étude des saisonnalités, des tendances et nettoyage des données.
2.  **Feature Engineering :** Création de variables explicatives (lags, moyennes mobiles, indicateurs de fêtes).
3.  **Modélisation :** Comparaison de modèles baselines (moyennes naïves) avec des modèles avancés (Random Forest, XGBoost, LightGBM).
4.  **Évaluation :** Utilisation de métriques spécifiques aux séries temporelles (RMSE, MAE).
5.  **Déploiement :** Tracking des expériences et création d'un endpoint de prédiction.



![Workflow du Machine Learning](https://vectormine.b-cdn.net/wp-content/uploads/workflow_of_machine_learning_outline_diagram-1.jpg)

---

## 🛠️ Stack Technique

### 💻 Langages
*   **Python 3.8+** : Cœur du backend et du Machine Learning.
*   **JavaScript (ES6+)** : Logique du frontend (appels API, interactivité).
*   **HTML5 / CSS3** : Structure et design responsive de l'interface utilisateur.

### 🧠 Data & Machine Learning
*   **Polars** : Manipulation de données haute performance (plus rapide que Pandas pour les grands volumes).
*   **LightGBM** : Modèle principal de Gradient Boosting (choisi pour son efficacité sur les données tabulaires).
*   **Scikit-Learn** : Preprocessing, pipelines et métriques d'évaluation.
*   **MLflow** : Tracking des expériences, versioning des modèles et registre.
*   **Pandas / Numpy** : Exploration de données (EDA) et prototypage.

### ⚙️ Backend & API
*   **FastAPI** : Framework moderne et rapide pour créer l'API d'inférence.
*   **Uvicorn** : Serveur ASGI pour la production.
*   **Pydantic** : Validation robuste des données d'entrée/sortie.

### 🚀 DevOps & Déploiement
*   **Docker** : Conteneurisation de l'application pour garantir la reproductibilité.
*   **Render** : Plateforme Cloud (PaaS) utilisée pour le déploiement de l'API et du site web.
*   **GitHub Actions** : Pipeline CI/CD pour l'automatisation des tests et du déploiement.

### 🧪 Qualité & Outils
*   **Pytest** : Tests unitaires et d'intégration.
*   **Jupyter Notebooks** : Espace de recherche et d'analyse exploratoire.
*   **Git / GitHub** : Gestion de version collaborative.

---

## 📂 Structure du Repository
Le projet est organisé selon les pratiques de structuration de projets Data Science :

```text
.
├── data/                                         # Données du projet
│   └── corporacin-favorita-grocery-sales-forecasting/  # Données brutes Kaggle
│       ├── train.csv                             # Historique des ventes
│       ├── test.csv                              # Données de test
│       ├── transactions.csv                      # Transactions par magasin
│       ├── oil.csv                               # Prix du pétrole
│       ├── holidays_events.csv                   # Jours fériés
│       ├── stores.csv                            # Infos magasins
│       └── items.csv                             # Infos articles
│
├── docs/                                         # Documentation et livrables
│   ├── DEPLOYMENT.md                             # Guide de déploiement (Render/Docker)
│   ├── MLFLOW_GUIDE.md                           # Guide MLflow
│   ├── iml-project-description_REG09.pdf         # Sujet du projet
│   └── PrédictionsVentesFavorita_IML2026.pdf     # Support de présentation
│
├── notebooks/                                    # Notebooks d'exploration
│   ├── 01_eda.ipynb                              # Analyse exploratoire (EDA)
│   ├── 02_feature_engineering.ipynb              # Feature Engineering
│   ├── 03_modeling.ipynb                         # Modélisation avancée
│   └── favorita-full-project.ipynb               # Notebook consolidé
│
├── src/                                          # Code source backend/ML
│   ├── __init__.py
│   ├── predict.py                                # API FastAPI et inférence
│   ├── preprocessing.py                          # Fonctions de transformation
│   └── train.py                                  # Script d'entraînement
│
├── tests/                                        # Tests automatisés
│   ├── test_api.py                               # Tests des endpoints API
│   └── test_pipeline.py                          # Tests du pipeline de données
│
├── webapp/                                       # Application Frontend
│   ├── css/                                      # Feuilles de style
│   ├── js/                                       # Scripts JavaScript
│   ├── data/                                     # Données de référence (JSON)
│   ├── dashboard.html                            # Dashboard de visualisation
│   ├── index.html                                # Page d'accueil
│   └── methodology.html                          # Page méthodologie
│
├── mlruns/                                       # Tracking MLflow
├── .gitattributes                                # Configuration Git (fins de ligne)
├── Dockerfile                                    # Configuration Docker pour l'API + Frontend
├── Procfile                                      # Fichier de démarrage (déploiement)
├── README.md                                     # Documentation générale (ce fichier)
├── render.yaml                                   # Configuration IaC pour Render
├── requirements.txt                              # Dépendances Python
├── runtime.txt                                   # Version Python (déploiement)
└── serve_webapp.py                               # Serveur local de développement
```

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/awa-d/favorita-sales-forecasting_ENSAE-ISE2-2026.git
cd favorita-sales-forecasting_ENSAE-ISE2-2026
```

### 2. Environnement Virtuel (Recommandé)
Il est fortement conseillé d'utiliser un environnement virtuel pour isoler les dépendances du projet.

**Sur Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

**Sur macOS / Linux :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Données et Préparation
Les données sont déjà disponibles dans le dossier `data/`.

Lancez le script de prétraitement pour générer les features (seules les 10 dernières semaines sont conservées pour l'entraînement) :
```bash
python src/preprocessing.py
```

---

## 🎮 Utilisation

### 1. Entraînement du Modèle
Pour lancer l'entraînement du modèle LightGBM (avec tracking MLflow) :
```bash
python src/train.py
```
*   Cela générera le fichier du modèle : `src/model_lgbm.txt`.
*   Les métriques et logs seront disponibles dans `mlruns/` (visualisable avec `mlflow ui`).

### 2. Démarrer l'API + Frontend (Production)
L'API FastAPI sert également le frontend directement :
```bash
uvicorn src.predict:app --host 0.0.0.0 --port 8000
```
*   **Page d'accueil** : `http://localhost:8000`
*   **Dashboard** : `http://localhost:8000/dashboard.html`
*   **Méthodologie** : `http://localhost:8000/methodology.html`
*   **Documentation API (Swagger)** : `http://localhost:8000/docs`
*   **Health Check** : `http://localhost:8000/health`

### 3. Serveur de Développement Local
Pour le développement avec proxy vers l'API Render :
```bash
python serve_webapp.py
```
*   Le script ouvrira automatiquement votre navigateur à l'adresse : `http://localhost:8080`
*   Utilise un proxy pour rediriger les appels `/api/*` vers l'API Render.

### 4. Docker (Production)
Pour conteneuriser et lancer l'application complète (API + Frontend) :
```bash
docker build -t favorita-api .
docker run -p 8000:8000 favorita-api
```

---

## 🌐 Déploiement sur Render

### Architecture Unifiée
L'application est déployée en tant que service unique sur Render :

```
https://favorita-sales-api.onrender.com
├── /                    → Page d'accueil (Frontend)
├── /dashboard.html      → Dashboard interactif
├── /methodology.html    → Page méthodologie
├── /css, /js, /data     → Fichiers statiques
├── /predict             → API Prédiction (POST)
├── /predict/batch       → API Prédiction Batch (POST)
├── /health              → Health Check
└── /docs                → Documentation Swagger
```

### Déployer
1. Pusher le code sur GitHub
2. Créer un service sur [render.com](https://render.com) → "New +" → "Blueprint"
3. Sélectionner le repository (Render détecte `render.yaml`)
4. Cliquer "Apply"

Pour plus de détails, voir [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---
## Lien vers la présentation sur canva [**ICI**](https://www.canva.com/design/DAG86BK4mTc/RT29hLb2_2HkrrFvX65mfg/edit?utm_content=DAG86BK4mTc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 📡 Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Page d'accueil (Frontend) |
| `GET` | `/health` | Vérification de santé de l'API |
| `POST` | `/predict` | Prédiction unitaire (JSON) |
| `POST` | `/predict/batch` | Prédiction par lot (CSV) |
| `GET` | `/docs` | Documentation Swagger |

### Exemple de Prédiction
```bash
curl -X POST "https://favorita-sales-api.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "store_nbr": 1,
    "item_nbr": 103520,
    "date": "2017-08-16",
    "onpromotion": 0,
    "perishable": 1,
    "dcoilwtico": 47.5,
    "transactions": 1500
  }'
```

---

## 👥 Membres du Groupe

Ce projet est réalisé par :

* [**Awa Diaw**](https://github.com/Awa-d)
* [**Alioune Abdou Salam Kane**](https://github.com/AliouneKane)
* [**Paul Balafai**](https://github.com/ruskovin)
* [**Jeanne de la Flèche Onanena Amana**](https://github.com/Lafleche06)
* [**Mame Balla Bousso**](https://github.com/MameBallaBousso)

🎓 *Étudiants en **ISE 2**, ENSAE de Dakar*

---

**Encadrement pédagogique :** **Madame Mously Diaw**, *Freelance Senior Data Scientist / ML Engineer*

