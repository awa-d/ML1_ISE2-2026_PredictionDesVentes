"""Test de l'effet des promotions sur les prédictions."""
import requests

BASE_URL = 'http://localhost:8000'

test_cases = [
    {'item_nbr': 96995, 'family': 'GROCERY I', 'store_nbr': 1},
    {'item_nbr': 1503899, 'family': 'BEVERAGES', 'store_nbr': 44},
    {'item_nbr': 103665, 'family': 'BREAD/BAKERY', 'store_nbr': 1},
    {'item_nbr': 115267, 'family': 'CLEANING', 'store_nbr': 51},
    {'item_nbr': 114799, 'family': 'PERSONAL CARE', 'store_nbr': 9},
    {'item_nbr': 2090247, 'family': 'MEATS', 'store_nbr': 25},
]

print('=== TEST EFFET PROMOTION ===')
print()
print('Status | Famille          | Sans promo | Avec promo | Boost')
print('-' * 65)

all_ok = True
for tc in test_cases:
    payload = {
        'store_nbr': tc['store_nbr'],
        'item_nbr': tc['item_nbr'],
        'date': '2025-01-15',
        'onpromotion': 0,
        'perishable': 0,
        'cluster': 13,
        'dcoilwtico': 50.0,
        'transactions': 1000
    }
    
    try:
        r1 = requests.post(f'{BASE_URL}/predict', json=payload, timeout=5)
        payload['onpromotion'] = 1
        r2 = requests.post(f'{BASE_URL}/predict', json=payload, timeout=5)
        
        if r1.status_code == 200 and r2.status_code == 200:
            no_promo = r1.json()['predicted_unit_sales']
            with_promo = r2.json()['predicted_unit_sales']
            boost = ((with_promo / no_promo) - 1) * 100 if no_promo > 0 else 0
            
            status = 'OK' if with_promo > no_promo else 'FAIL'
            if with_promo <= no_promo:
                all_ok = False
            
            family = tc['family']
            print(f'{status:6} | {family:16} | {no_promo:10.1f} | {with_promo:10.1f} | +{boost:.0f}%')
        else:
            print(f'ERROR  | {tc["family"]:16} | HTTP {r1.status_code}')
            all_ok = False
    except Exception as e:
        print(f'ERROR  | {tc["family"]:16} | {e}')
        all_ok = False

print()
if all_ok:
    print('✅ SUCCES: Toutes les promotions augmentent la demande predite!')
else:
    print('❌ ECHEC: Certaines promotions ne fonctionnent pas')
