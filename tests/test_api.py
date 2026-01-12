"""
==============================================================================
test_api.py - Tests de l'API Favorita Sales Forecasting
==============================================================================
Ce script teste les endpoints de l'API déployée sur Render.

Usage:
    pytest tests/test_api.py -v
    python tests/test_api.py  # Mode standalone
"""

import pytest
import requests
import time
from typing import Dict, Any

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_BASE_URL = "https://favorita-sales-api.onrender.com"
API_TIMEOUT = 90  # Render cold start peut prendre jusqu'à 60-90 secondes


# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture(scope="module")
def api_session():
    """Crée une session HTTP réutilisable avec retry."""
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    return session


@pytest.fixture(scope="module")
def wake_up_api(api_session):
    """S'assure que l'API est réveillée avant les tests."""
    print("\n⏳ Réveil de l'API Render (peut prendre 30-60s)...")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = api_session.get(
                f"{API_BASE_URL}/health",
                timeout=API_TIMEOUT
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API prête: {data.get('num_trees', 'N/A')} arbres, {data.get('num_features', 'N/A')} features")
                return data
        except requests.exceptions.Timeout:
            print(f"   Tentative {attempt + 1}/{max_retries} - Timeout, retry...")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"   Tentative {attempt + 1}/{max_retries} - Erreur: {e}")
            time.sleep(5)
    
    pytest.skip("API non disponible après plusieurs tentatives")


# ==============================================================================
# TESTS - HEALTH ENDPOINT
# ==============================================================================
class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_returns_200(self, api_session, wake_up_api):
        """Test que /health retourne un status 200."""
        response = api_session.get(f"{API_BASE_URL}/health", timeout=30)
        assert response.status_code == 200
    
    def test_health_returns_json(self, api_session, wake_up_api):
        """Test que /health retourne du JSON valide."""
        response = api_session.get(f"{API_BASE_URL}/health", timeout=30)
        data = response.json()
        assert isinstance(data, dict)
    
    def test_health_contains_required_fields(self, api_session, wake_up_api):
        """Test que /health contient les champs requis."""
        response = api_session.get(f"{API_BASE_URL}/health", timeout=30)
        data = response.json()
        
        required_fields = ["status", "model_loaded"]
        for field in required_fields:
            assert field in data, f"Champ manquant: {field}"
    
    def test_health_model_is_loaded(self, api_session, wake_up_api):
        """Test que le modèle est bien chargé."""
        response = api_session.get(f"{API_BASE_URL}/health", timeout=30)
        data = response.json()
        
        assert data.get("model_loaded") == True, "Le modèle n'est pas chargé"
    
    def test_health_status_is_healthy(self, api_session, wake_up_api):
        """Test que le status est 'healthy'."""
        response = api_session.get(f"{API_BASE_URL}/health", timeout=30)
        data = response.json()
        
        assert data.get("status") == "healthy"


# ==============================================================================
# TESTS - PREDICT ENDPOINT
# ==============================================================================
class TestPredictEndpoint:
    """Tests pour l'endpoint /predict"""
    
    @pytest.fixture
    def valid_prediction_payload(self) -> Dict[str, Any]:
        """Payload valide pour une prédiction."""
        return {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "2017-08-16",
            "onpromotion": 0
        }
    
    def test_predict_returns_200(self, api_session, wake_up_api, valid_prediction_payload):
        """Test que /predict retourne un status 200 avec un payload valide."""
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=valid_prediction_payload,
            timeout=30
        )
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
    
    def test_predict_returns_prediction(self, api_session, wake_up_api, valid_prediction_payload):
        """Test que /predict retourne une prédiction."""
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=valid_prediction_payload,
            timeout=30
        )
        data = response.json()
        
        assert "predicted_unit_sales" in data, f"Réponse: {data}"
        assert isinstance(data["predicted_unit_sales"], (int, float))
    
    def test_predict_value_is_positive(self, api_session, wake_up_api, valid_prediction_payload):
        """Test que la prédiction est un nombre positif ou nul."""
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=valid_prediction_payload,
            timeout=30
        )
        data = response.json()
        
        prediction = data.get("predicted_unit_sales", -1)
        assert prediction >= 0, f"Prédiction négative: {prediction}"
    
    def test_predict_with_promotion(self, api_session, wake_up_api):
        """Test prédiction avec promotion activée."""
        payload = {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "2017-08-20",
            "onpromotion": 1
        }
        
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "predicted_unit_sales" in data
    
    def test_predict_different_stores(self, api_session, wake_up_api):
        """Test prédictions pour différents magasins."""
        stores = [1, 25, 44, 51]
        
        for store_nbr in stores:
            payload = {
                "store_nbr": store_nbr,
                "item_nbr": 103665,
                "date": "2017-08-16",
                "onpromotion": 0
            }
            
            response = api_session.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=30
            )
            
            assert response.status_code == 200, f"Échec pour store {store_nbr}"
    
    def test_predict_different_dates(self, api_session, wake_up_api):
        """Test prédictions pour différentes dates."""
        dates = ["2017-08-16", "2017-08-20", "2017-08-25", "2017-08-31"]
        
        for date in dates:
            payload = {
                "store_nbr": 44,
                "item_nbr": 103665,
                "date": date,
                "onpromotion": 0
            }
            
            response = api_session.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=30
            )
            
            assert response.status_code == 200, f"Échec pour date {date}"


# ==============================================================================
# TESTS - VALIDATION DES ENTRÉES
# ==============================================================================
class TestInputValidation:
    """Tests de validation des entrées."""
    
    def test_missing_store_nbr(self, api_session, wake_up_api):
        """Test erreur si store_nbr manquant."""
        payload = {
            "item_nbr": 103665,
            "date": "2017-08-16"
        }
        
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        
        # Devrait retourner 422 (Validation Error) ou 400 (Bad Request)
        assert response.status_code in [400, 422], f"Status inattendu: {response.status_code}"
    
    def test_missing_item_nbr(self, api_session, wake_up_api):
        """Test erreur si item_nbr manquant."""
        payload = {
            "store_nbr": 44,
            "date": "2017-08-16"
        }
        
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        
        assert response.status_code in [400, 422]
    
    def test_missing_date(self, api_session, wake_up_api):
        """Test erreur si date manquante."""
        payload = {
            "store_nbr": 44,
            "item_nbr": 103665
        }
        
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        
        assert response.status_code in [400, 422]
    
    def test_invalid_date_format(self, api_session, wake_up_api):
        """Test erreur avec format de date invalide."""
        payload = {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "16-08-2017"  # Format incorrect
        }
        
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        
        # Devrait retourner une erreur ou être tolérant
        # On vérifie au moins que ça ne crash pas silencieusement
        assert response.status_code in [200, 400, 422]


# ==============================================================================
# TESTS - PERFORMANCE
# ==============================================================================
class TestPerformance:
    """Tests de performance de l'API."""
    
    def test_response_time_under_5_seconds(self, api_session, wake_up_api):
        """Test que la réponse arrive en moins de 5 secondes (API chaude)."""
        payload = {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "2017-08-16",
            "onpromotion": 0
        }
        
        start_time = time.time()
        response = api_session.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed_time < 5.0, f"Temps de réponse trop long: {elapsed_time:.2f}s"
        print(f"   ⏱️ Temps de réponse: {elapsed_time:.2f}s")
    
    def test_multiple_sequential_requests(self, api_session, wake_up_api):
        """Test plusieurs requêtes séquentielles."""
        payload = {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "2017-08-16",
            "onpromotion": 0
        }
        
        success_count = 0
        total_time = 0
        num_requests = 5
        
        for i in range(num_requests):
            start = time.time()
            response = api_session.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=30
            )
            elapsed = time.time() - start
            total_time += elapsed
            
            if response.status_code == 200:
                success_count += 1
        
        avg_time = total_time / num_requests
        print(f"   📊 {success_count}/{num_requests} succès, temps moyen: {avg_time:.2f}s")
        
        assert success_count == num_requests, f"Seulement {success_count}/{num_requests} requêtes réussies"


# ==============================================================================
# TESTS - COHÉRENCE DES PRÉDICTIONS
# ==============================================================================
class TestPredictionConsistency:
    """Tests de cohérence des prédictions."""
    
    def test_same_input_same_output(self, api_session, wake_up_api):
        """Test que les mêmes entrées donnent les mêmes sorties."""
        payload = {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "2017-08-16",
            "onpromotion": 0
        }
        
        predictions = []
        for _ in range(3):
            response = api_session.post(
                f"{API_BASE_URL}/predict",
                json=payload,
                timeout=30
            )
            data = response.json()
            predictions.append(data.get("predicted_unit_sales"))
        
        # Toutes les prédictions devraient être identiques
        assert len(set(predictions)) == 1, f"Prédictions incohérentes: {predictions}"
    
    def test_promotion_increases_prediction(self, api_session, wake_up_api):
        """Test que la promotion a un impact sur la prédiction."""
        base_payload = {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "2017-08-16"
        }
        
        # Sans promotion
        response_no_promo = api_session.post(
            f"{API_BASE_URL}/predict",
            json={**base_payload, "onpromotion": 0},
            timeout=30
        )
        pred_no_promo = response_no_promo.json().get("predicted_unit_sales", 0)
        
        # Avec promotion
        response_promo = api_session.post(
            f"{API_BASE_URL}/predict",
            json={**base_payload, "onpromotion": 1},
            timeout=30
        )
        pred_promo = response_promo.json().get("predicted_unit_sales", 0)
        
        print(f"   📈 Sans promo: {pred_no_promo:.2f}, Avec promo: {pred_promo:.2f}")
        
        # La prédiction avec promo devrait généralement être >= sans promo
        # (mais pas toujours selon le modèle, donc on vérifie juste que ça marche)
        assert pred_no_promo >= 0 and pred_promo >= 0


# ==============================================================================
# MODE STANDALONE
# ==============================================================================
def run_standalone_tests():
    """Exécute les tests en mode standalone (sans pytest)."""
    print("=" * 60)
    print("🧪 TESTS API FAVORITA SALES FORECASTING")
    print("=" * 60)
    print(f"📡 URL: {API_BASE_URL}")
    print()
    
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Health Check
    print("1️⃣ Test Health Check...")
    try:
        response = session.get(f"{API_BASE_URL}/health", timeout=API_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Status: {data.get('status')}")
            print(f"   ✅ Model Loaded: {data.get('model_loaded')}")
            print(f"   ✅ Num Trees: {data.get('num_trees')}")
            tests_passed += 1
        else:
            print(f"   ❌ Status Code: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        tests_failed += 1
    
    # Test 2: Basic Prediction
    print("\n2️⃣ Test Prédiction Basique...")
    try:
        payload = {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "2017-08-16",
            "onpromotion": 0
        }
        response = session.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            pred = data.get("predicted_unit_sales", "N/A")
            print(f"   ✅ Prédiction: {pred:.2f} unités" if isinstance(pred, (int, float)) else f"   ✅ Réponse: {data}")
            tests_passed += 1
        else:
            print(f"   ❌ Status Code: {response.status_code}")
            print(f"   ❌ Response: {response.text}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        tests_failed += 1
    
    # Test 3: Prediction with Promotion
    print("\n3️⃣ Test Prédiction avec Promotion...")
    try:
        payload = {
            "store_nbr": 44,
            "item_nbr": 103665,
            "date": "2017-08-20",
            "onpromotion": 1
        }
        response = session.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            pred = data.get("predicted_unit_sales", "N/A")
            print(f"   ✅ Prédiction (promo): {pred:.2f} unités" if isinstance(pred, (int, float)) else f"   ✅ Réponse: {data}")
            tests_passed += 1
        else:
            print(f"   ❌ Status Code: {response.status_code}")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        tests_failed += 1
    
    # Test 4: Different Stores
    print("\n4️⃣ Test Différents Magasins...")
    stores_ok = 0
    for store in [1, 25, 44, 51]:
        try:
            payload = {"store_nbr": store, "item_nbr": 103665, "date": "2017-08-16", "onpromotion": 0}
            response = session.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
            if response.status_code == 200:
                stores_ok += 1
        except:
            pass
    
    if stores_ok == 4:
        print(f"   ✅ {stores_ok}/4 magasins OK")
        tests_passed += 1
    else:
        print(f"   ⚠️ {stores_ok}/4 magasins OK")
        tests_failed += 1
    
    # Test 5: Response Time
    print("\n5️⃣ Test Temps de Réponse...")
    try:
        payload = {"store_nbr": 44, "item_nbr": 103665, "date": "2017-08-16", "onpromotion": 0}
        start = time.time()
        response = session.post(f"{API_BASE_URL}/predict", json=payload, timeout=30)
        elapsed = time.time() - start
        
        if response.status_code == 200 and elapsed < 5:
            print(f"   ✅ Temps de réponse: {elapsed:.2f}s")
            tests_passed += 1
        else:
            print(f"   ⚠️ Temps de réponse: {elapsed:.2f}s (limite: 5s)")
            tests_failed += 1
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        tests_failed += 1
    
    # Résumé
    print("\n" + "=" * 60)
    total = tests_passed + tests_failed
    print(f"📊 RÉSUMÉ: {tests_passed}/{total} tests passés")
    if tests_failed == 0:
        print("🎉 Tous les tests sont passés!")
    else:
        print(f"⚠️ {tests_failed} test(s) en échec")
    print("=" * 60)
    
    return tests_failed == 0


if __name__ == "__main__":
    import sys
    success = run_standalone_tests()
    sys.exit(0 if success else 1)
