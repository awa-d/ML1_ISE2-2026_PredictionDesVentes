# 🚀 Guide de Déploiement sur Render

## 📋 Prérequis

- ✅ Un compte GitHub avec votre code pushé
- ✅ Un compte Render (gratuit) : [render.com](https://render.com)
- ✅ Le modèle entraîné `src/model_lgbm.txt`

## 🎯 Étapes de Déploiement

### 1️⃣ Préparer votre Repository Git

```bash
# Si pas encore fait, initialisez Git
cd favorita-sales-forecasting_ENSAE-ISE2-2026
git init

# Ajoutez vos fichiers
git add .
git commit -m "Ready for Render deployment"

# Poussez vers GitHub
git remote add origin https://github.com/VOTRE_USERNAME/favorita-sales-api.git
git branch -M main
git push -u origin main
```

### 2️⃣ Créer le Service sur Render

#### Option A : Via Blueprint (AUTOMATIQUE - Recommandé)

1. Connectez-vous sur [render.com](https://render.com)
2. Cliquez sur **"New +"** → **"Blueprint"**
3. Sélectionnez votre repository GitHub
4. Render détecte automatiquement `render.yaml`
5. Cliquez **"Apply"**
6. ✅ Le déploiement commence !

#### Option B : Via Dashboard (MANUEL)

1. Connectez-vous sur [render.com](https://render.com)
2. Cliquez **"New +"** → **"Web Service"**
3. Connectez votre repository GitHub
4. Configurez :
   - **Name** : `favorita-sales-api`
   - **Region** : `Frankfurt` (Europe) ou `Oregon` (USA)
   - **Branch** : `main`
   - **Runtime** : `Python 3`
   - **Build Command** : 
     ```
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command** : 
     ```
     uvicorn src.predict:app --host 0.0.0.0 --port $PORT --workers 1
     ```
   - **Instance Type** : `Free` (ou `Starter` pour 7$/mois)

5. **Variables d'environnement** (optionnel) :
   - `PYTHON_VERSION` = `3.11.0`

6. Cliquez **"Create Web Service"**

### 3️⃣ Surveillance du Déploiement

Dans le dashboard Render :
- **Logs** : Voir le build en temps réel
- Status devient **"Live"** quand prêt (2-5 min)
- URL disponible : `https://favorita-sales-api.onrender.com`

## ⚠️ IMPORTANT : Le Modèle

Le fichier `src/model_lgbm.txt` **DOIT** être présent. Deux options :

### Option 1 : Commit dans Git (Simple)
```bash
git add src/model_lgbm.txt
git commit -m "Add trained model"
git push
```

### Option 2 : Utiliser Render Disk (Si modèle > 100MB)
1. Dashboard Render → **"Disks"** → **"New Disk"**
2. Montez le disk sur `/app/src`
3. Uploadez `model_lgbm.txt` via SFTP

## 🧪 Tester votre API Déployée

### Health Check
```bash
curl https://favorita-sales-api.onrender.com/health
```

**Réponse attendue** :
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_trees": 450,
  "num_features": 35
}
```

### Prédiction Unique
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

**Réponse attendue** :
```json
{
  "store_nbr": 1,
  "item_nbr": 103520,
  "date": "2017-08-16",
  "predicted_unit_sales": 12.45,
  "confidence": "medium"
}
```

### Documentation Interactive
Accédez à : `https://favorita-sales-api.onrender.com/docs`
- Interface Swagger UI
- Testez les endpoints directement

## 📊 Surveillance en Production

### Voir les Logs
```bash
# Via CLI (installez render CLI)
render logs -s favorita-sales-api -t

# Ou via Dashboard
# Render Dashboard → Services → favorita-sales-api → Logs
```

### Métriques Disponibles
- **CPU Usage** : Dashboard → Metrics
- **Memory Usage** : Dashboard → Metrics
- **Response Time** : Via logs
- **Error Rate** : Via logs

## ⚡ Optimisations

### Plan Gratuit (Free)
- ✅ Suffisant pour démonstration
- ⚠️ **Cold Start** : 30-50 secondes si inactif 15 min
- 512 MB RAM
- CPU partagé

### Plan Starter (7$/mois)
- ✅ Pas de cold start
- 512 MB RAM garantie
- Plus rapide
- Custom domains

### Astuces Performance

1. **Éviter Cold Start** (Plan gratuit) :
   - Utilisez un service comme [UptimeRobot](https://uptimerobot.com) pour ping toutes les 10 min
   - Ou utilisez GitHub Actions :
   ```yaml
   # .github/workflows/keep-alive.yml
   name: Keep Alive
   on:
     schedule:
       - cron: '*/10 * * * *'  # Toutes les 10 minutes
   jobs:
     ping:
       runs-on: ubuntu-latest
       steps:
         - run: curl https://favorita-sales-api.onrender.com/health
   ```

2. **Réduire Taille du Build** :
   ```txt
   # Commentez dans requirements.txt les dépendances inutiles en prod
   # mlflow>=2.9.0  # Seulement pour training
   # opendatasets>=0.1.22  # Seulement pour download
   # pytest>=7.0.0  # Seulement pour tests
   ```

3. **Optimiser Workers** :
   - Restez à `--workers 1` sur plan gratuit
   - Passez à `--workers 2` sur Starter

## 🔧 Dépannage

### ❌ Erreur : "Application failed to respond"
**Solution** : Vérifiez que l'app écoute sur `$PORT` :
```python
# src/predict.py doit avoir :
PORT = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=PORT)
```

### ❌ Erreur : "Modèle non chargé"
**Solution** : Vérifiez que `src/model_lgbm.txt` existe :
```bash
# Localement
ls src/model_lgbm.txt

# Via logs Render
# Cherchez : "[PREDICT] Chargement du modèle depuis"
```

### ❌ Erreur : "Out of Memory"
**Solutions** :
1. Passez au plan Starter (512 MB garantis)
2. Optimisez le modèle (moins d'arbres)
3. Utilisez `num_threads=1` dans LightGBM

### ❌ Build échoue : "Requirements installation failed"
**Solution** : Vérifiez les versions dans `requirements.txt` :
```bash
# Testez localement
pip install -r requirements.txt
```

## 🌐 Utilisation depuis une App Frontend

### JavaScript / React
```javascript
async function predictSales(data) {
  const response = await fetch('https://favorita-sales-api.onrender.com/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return await response.json();
}

// Utilisation
const result = await predictSales({
  store_nbr: 1,
  item_nbr: 103520,
  date: "2017-08-16",
  onpromotion: 0,
  perishable: 1,
  dcoilwtico: 47.5,
  transactions: 1500
});
console.log(result.predicted_unit_sales);
```

### Python (requests)
```python
import requests

url = "https://favorita-sales-api.onrender.com/predict"
data = {
    "store_nbr": 1,
    "item_nbr": 103520,
    "date": "2017-08-16",
    "onpromotion": 0,
    "perishable": 1,
    "dcoilwtico": 47.5,
    "transactions": 1500
}

response = requests.post(url, json=data)
print(response.json())
```

## 🔗 Ressources

- [Render Documentation](https://render.com/docs)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Render Community Forum](https://community.render.com/)

## 📞 Support

En cas de problème :
1. Vérifiez les logs : Dashboard → Logs
2. Consultez [Status Render](https://status.render.com/)
3. Forum communautaire : [community.render.com](https://community.render.com/)

---

**🎉 Votre API est maintenant en production !**

URL : `https://favorita-sales-api.onrender.com`
Docs : `https://favorita-sales-api.onrender.com/docs`
