/* ============================================================================
   FAVORITA INTELLIGENCE HUB - MAIN JAVASCRIPT
   Core application logic for the prediction dashboard
   ============================================================================ */

// ==============================================================================
// CONFIGURATION
// ==============================================================================
const CONFIG = {
    // API URL - auto-detect based on environment
    API_URL: window.location.hostname === 'localhost' ? '/api' : '',  // Use relative URL in production
    API_URL_DIRECT: 'https://favorita-sales-api.onrender.com',  // Direct URL for fallback
    RENDER_API_KEY: 'rnd_t1Hvg8aJreD789iAZuhzLiMi2f32',
    GEMINI_API_KEY: 'AIzaSyAnYx7krhfhd5sONq3sL0Hsk_fQi0gtqP4',
    GEMINI_API_URL: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
    DATE_RANGE: {
        start: '2017-08-16',
        end: '2017-08-31'
    },
    API_TIMEOUT: 60000, // 60 seconds for Render cold start
    API_RETRY_DELAY: 3000 // 3 seconds between retries
};

// ==============================================================================
// DATA STORES (Loaded from CSV)
// ==============================================================================
let STORES_DATA = [];
let ITEMS_DATA = [];
let HOLIDAYS_DATA = [];

// ==============================================================================
// CSV PARSING UTILITIES
// ==============================================================================
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const row = {};
        headers.forEach((header, index) => {
            row[header] = values[index]?.trim() || '';
        });
        data.push(row);
    }
    return data;
}

// ==============================================================================
// DATA LOADING
// ==============================================================================
async function loadCSVData() {
    try {
        // First try to load from local JSON (more reliable for web serving)
        const jsonResponse = await fetch('data/reference_data.json');
        if (jsonResponse.ok) {
            const data = await jsonResponse.json();
            STORES_DATA = data.stores || [];
            ITEMS_DATA = data.items || [];
            HOLIDAYS_DATA = data.holidays || [];
            console.log(`✅ Loaded data from JSON: ${STORES_DATA.length} stores, ${ITEMS_DATA.length} items, ${HOLIDAYS_DATA.length} holidays`);
            return true;
        }
    } catch (jsonError) {
        console.log('JSON load failed, trying CSV files...');
    }
    
    try {
        // Fallback: Load from CSV files
        const storesResponse = await fetch('../data/corporacin-favorita-grocery-sales-forecasting/stores.csv');
        if (storesResponse.ok) {
            const storesText = await storesResponse.text();
            STORES_DATA = parseCSV(storesText);
            console.log(`✅ Loaded ${STORES_DATA.length} stores from CSV`);
        }
        
        const itemsResponse = await fetch('../data/corporacin-favorita-grocery-sales-forecasting/items.csv');
        if (itemsResponse.ok) {
            const itemsText = await itemsResponse.text();
            ITEMS_DATA = parseCSV(itemsText);
            console.log(`✅ Loaded ${ITEMS_DATA.length} items from CSV`);
        }
        
        const holidaysResponse = await fetch('../data/corporacin-favorita-grocery-sales-forecasting/holidays_events.csv');
        if (holidaysResponse.ok) {
            const holidaysText = await holidaysResponse.text();
            HOLIDAYS_DATA = parseCSV(holidaysText);
            console.log(`✅ Loaded ${HOLIDAYS_DATA.length} holidays from CSV`);
        }
        
        return true;
    } catch (error) {
        console.error('Error loading CSV data:', error);
        // Use embedded fallback data
        loadFallbackData();
        return false;
    }
}

function loadFallbackData() {
    // Fallback stores data
    STORES_DATA = [
        { store_nbr: '1', city: 'Quito', state: 'Pichincha', type: 'D', cluster: '13' },
        { store_nbr: '44', city: 'Quito', state: 'Pichincha', type: 'A', cluster: '5' },
        { store_nbr: '45', city: 'Quito', state: 'Pichincha', type: 'A', cluster: '11' },
        { store_nbr: '46', city: 'Quito', state: 'Pichincha', type: 'A', cluster: '14' },
        { store_nbr: '47', city: 'Quito', state: 'Pichincha', type: 'A', cluster: '14' },
        { store_nbr: '48', city: 'Quito', state: 'Pichincha', type: 'A', cluster: '14' },
        { store_nbr: '49', city: 'Quito', state: 'Pichincha', type: 'A', cluster: '11' },
        { store_nbr: '50', city: 'Ambato', state: 'Tungurahua', type: 'A', cluster: '14' },
        { store_nbr: '51', city: 'Guayaquil', state: 'Guayas', type: 'A', cluster: '17' },
        { store_nbr: '52', city: 'Manta', state: 'Manabi', type: 'A', cluster: '11' },
        { store_nbr: '24', city: 'Guayaquil', state: 'Guayas', type: 'D', cluster: '1' },
        { store_nbr: '25', city: 'Salinas', state: 'Santa Elena', type: 'D', cluster: '1' },
        { store_nbr: '37', city: 'Cuenca', state: 'Azuay', type: 'D', cluster: '2' },
        { store_nbr: '3', city: 'Quito', state: 'Pichincha', type: 'D', cluster: '8' }
    ];
    
    // Fallback popular items
    ITEMS_DATA = [
        { item_nbr: '103665', family: 'BREAD/BAKERY', class: '2712', perishable: '1' },
        { item_nbr: '105574', family: 'GROCERY I', class: '1045', perishable: '0' },
        { item_nbr: '108696', family: 'DELI', class: '2636', perishable: '1' },
        { item_nbr: '108831', family: 'POULTRY', class: '2416', perishable: '1' },
        { item_nbr: '108833', family: 'EGGS', class: '2502', perishable: '1' },
        { item_nbr: '96995', family: 'GROCERY I', class: '1093', perishable: '0' },
        { item_nbr: '99197', family: 'GROCERY I', class: '1067', perishable: '0' },
        { item_nbr: '103501', family: 'CLEANING', class: '3008', perishable: '0' },
        { item_nbr: '114799', family: 'PERSONAL CARE', class: '4126', perishable: '0' },
        { item_nbr: '1503899', family: 'BEVERAGES', class: '1040', perishable: '0' }
    ];
    
    // August 2017 holidays
    HOLIDAYS_DATA = [
        { date: '2017-08-10', type: 'Holiday', locale: 'National', locale_name: 'Ecuador', description: 'Primer Grito de Independencia', transferred: 'false' },
        { date: '2017-08-11', type: 'Bridge', locale: 'National', locale_name: 'Ecuador', description: 'Puente Primer Grito de Independencia', transferred: 'false' },
        { date: '2017-08-15', type: 'Holiday', locale: 'Local', locale_name: 'Riobamba', description: 'Fundacion de Riobamba', transferred: 'false' },
        { date: '2017-08-24', type: 'Holiday', locale: 'Local', locale_name: 'Ambato', description: 'Fundacion de Ambato', transferred: 'false' }
    ];
    
    console.log('📦 Using fallback embedded data');
}

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================
function getStoreInfo(storeNbr) {
    const store = STORES_DATA.find(s => s.store_nbr === String(storeNbr));
    return store || { city: 'Inconnu', state: 'Inconnu', type: 'N/A', cluster: 'N/A' };
}

function getItemInfo(itemNbr) {
    const item = ITEMS_DATA.find(i => i.item_nbr === String(itemNbr));
    return item || { family: 'Inconnu', class: 'N/A', perishable: '0' };
}

function getHolidaysForDate(date, city = null) {
    return HOLIDAYS_DATA.filter(h => {
        const isDateMatch = h.date === date;
        const isNational = h.locale === 'National';
        const isLocalMatch = city && h.locale_name === city;
        return isDateMatch && (isNational || isLocalMatch);
    });
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('fr-FR', { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    });
}

function isPerishable(itemNbr) {
    const item = getItemInfo(itemNbr);
    return item.perishable === '1';
}

// ==============================================================================
// KPI CALCULATIONS
// ==============================================================================
function calculateUrgencyIndex(currentStock, prediction) {
    if (prediction <= 0) return { level: 'success', ratio: Infinity, label: 'Stock suffisant' };
    
    const ratio = currentStock / prediction;
    
    if (ratio < 0.5) {
        return { level: 'danger', ratio, label: 'Rupture imminente' };
    } else if (ratio < 1) {
        return { level: 'warning', ratio, label: 'Vigilance requise' };
    } else if (ratio < 1.5) {
        return { level: 'success', ratio, label: 'Stock optimal' };
    } else {
        return { level: 'info', ratio, label: 'Surstock potentiel' };
    }
}

function calculateEventImpact(date, city) {
    const holidays = getHolidaysForDate(date, city);
    
    if (holidays.length === 0) {
        return { impact: 'normal', multiplier: 1, events: [] };
    }
    
    const hasNational = holidays.some(h => h.locale === 'National');
    
    return {
        impact: hasNational ? 'high' : 'medium',
        multiplier: hasNational ? 1.4 : 1.2,
        events: holidays
    };
}

function getTrafficLightStatus(urgency, isPerishableProduct) {
    let status, description;
    
    if (urgency.level === 'danger') {
        status = 'red';
        description = isPerishableProduct 
            ? '⚠️ ALERTE CRITIQUE: Rupture imminente sur produit périssable!' 
            : 'Rupture de stock imminente. Réapprovisionnement urgent requis.';
    } else if (urgency.level === 'warning') {
        status = 'orange';
        description = isPerishableProduct 
            ? 'Flux tendu sur produit périssable. Surveillez les dates de péremption.' 
            : 'Stock en flux tendu. Planifiez le réapprovisionnement.';
    } else {
        status = 'green';
        description = 'Niveau de stock optimal. Opérations normales.';
    }
    
    return { status, description };
}

// ==============================================================================
// API FUNCTIONS
// ==============================================================================

// Helper function for fetch with timeout
async function fetchWithTimeout(url, options = {}, timeout = CONFIG.API_TIMEOUT) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error;
    }
}

// Get common headers for API requests
function getAPIHeaders() {
    return {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${CONFIG.RENDER_API_KEY}`,
        'X-API-Key': CONFIG.RENDER_API_KEY
    };
}

async function checkAPIHealth() {
    updateAPIStatus('checking');
    
    try {
        // Use local proxy to avoid CORS issues
        const response = await fetchWithTimeout(`${CONFIG.API_URL}/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        }, 30000); // 30 second timeout for health check
        
        if (response.ok) {
            const data = await response.json();
            updateAPIStatus(true, data);
            return true;
        }
        
        // Handle specific error codes
        if (response.status === 503) {
            updateAPIStatus('waking');
            // Render is waking up, retry after delay
            setTimeout(checkAPIHealth, CONFIG.API_RETRY_DELAY);
            return false;
        }
        
        updateAPIStatus(false);
        return false;
    } catch (error) {
        console.error('API Health Check Failed:', error);
        updateAPIStatus(false);
        return false;
    }
}

function updateAPIStatus(status, data = null) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.api-status span:last-child');
    
    if (statusDot && statusText) {
        // Remove all status classes
        statusDot.classList.remove('online', 'offline', 'checking', 'waking');
        
        if (status === true) {
            statusDot.classList.add('online');
            statusText.textContent = data?.model_loaded ? `API En ligne (${data.num_trees} arbres)` : 'API En ligne ✓';
        } else if (status === 'checking') {
            statusDot.classList.add('checking');
            statusText.textContent = 'Connexion...';
        } else if (status === 'waking') {
            statusDot.classList.add('waking');
            statusText.textContent = 'Réveil API (30s)...';
        } else {
            statusDot.classList.add('offline');
            statusText.textContent = 'API Hors ligne';
        }
    }
}

async function getPrediction(storeNbr, itemNbr, date, onpromotion = 0) {
    try {
        const response = await fetchWithTimeout(`${CONFIG.API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                store_nbr: parseInt(storeNbr),
                item_nbr: parseInt(itemNbr),
                date: date,
                onpromotion: parseInt(onpromotion)
            })
        }, CONFIG.API_TIMEOUT);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Erreur API');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Prediction Error:', error);
        throw error;
    }
}

// ==============================================================================
// GEMINI AI INTEGRATION
// ==============================================================================
async function getAIRecommendations(context) {
    const prompt = `Tu es un expert en gestion de stocks pour les supermarchés Favorita en Équateur.

CONTEXTE ACTUEL:
- Magasin: ${context.store.city} (Type ${context.store.type}, Cluster ${context.store.cluster})
- Produit: ${context.item.family} (${context.isPerishable ? 'PÉRISSABLE' : 'Non périssable'})
- Date: ${formatDate(context.date)}
- Prédiction de ventes: ${context.prediction.toFixed(0)} unités
- Stock actuel: ${context.currentStock} unités
- Ratio couverture: ${(context.currentStock / context.prediction).toFixed(2)}
${context.events.length > 0 ? `- ÉVÉNEMENT: ${context.events.map(e => e.description).join(', ')}` : '- Pas d\'événement spécial'}
- Niveau d'urgence: ${context.urgency.label}

IMPORTANT: Tu dois répondre UNIQUEMENT au format JSON suivant, sans aucun autre texte:
{
    "retenir": "Une phrase d'analyse de la situation (max 50 mots)",
    "faire": "Une action concrète à entreprendre (max 40 mots)",
    "quantite": "Recommandation chiffrée précise avec justification (max 30 mots)"
}`;

    try {
        const response = await fetch(`${CONFIG.GEMINI_API_URL}?key=${CONFIG.GEMINI_API_KEY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contents: [{
                    parts: [{ text: prompt }]
                }],
                generationConfig: {
                    temperature: 0.7,
                    maxOutputTokens: 500
                }
            })
        });
        
        if (!response.ok) {
            throw new Error('Gemini API Error');
        }
        
        const data = await response.json();
        const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
        
        if (text) {
            // Try to parse as JSON
            try {
                // Extract JSON from response (handle markdown code blocks)
                let jsonStr = text;
                if (text.includes('```')) {
                    jsonStr = text.replace(/```json?\n?/g, '').replace(/```/g, '').trim();
                }
                const parsed = JSON.parse(jsonStr);
                return {
                    retenir: parsed.retenir || 'Analyse en cours...',
                    faire: parsed.faire || 'Vérifier les niveaux de stock.',
                    quantite: parsed.quantite || `Commander ${Math.ceil(context.prediction * 1.1)} unités.`
                };
            } catch (parseError) {
                // Fallback: generate structured response from text
                return generateFallbackRecommendation(context);
            }
        }
        
        return generateFallbackRecommendation(context);
    } catch (error) {
        console.error('Gemini AI Error:', error);
        return generateFallbackRecommendation(context);
    }
}

function generateFallbackRecommendation(context) {
    const deficit = context.prediction - context.currentStock;
    const safetyStock = Math.ceil(context.prediction * 0.1);
    const orderQuantity = Math.max(0, deficit + safetyStock);
    
    let retenir, faire, quantite;
    
    if (context.urgency.level === 'danger') {
        retenir = `Situation critique: le stock actuel (${context.currentStock}) ne couvre que ${Math.round(context.urgency.ratio * 100)}% de la demande prévue.${context.events.length > 0 ? ` Attention: ${context.events[0].description} impacte les ventes.` : ''}`;
        faire = `Réapprovisionner immédiatement le rayon ${context.item.family}${context.isPerishable ? ' en priorité (produit périssable)' : ''}.`;
        quantite = `Commander ${orderQuantity} unités minimum (${Math.ceil(context.prediction)} prévus + ${safetyStock} marge sécurité).`;
    } else if (context.urgency.level === 'warning') {
        retenir = `Stock tendu: couverture à ${Math.round(context.urgency.ratio * 100)}% de la demande.${context.events.length > 0 ? ` ${context.events[0].description} peut augmenter la demande.` : ''}`;
        faire = `Planifier un réapprovisionnement pour ${context.item.family} avant ${context.isPerishable ? 'épuisement et péremption' : 'rupture'}.`;
        quantite = `Prévoir ${orderQuantity} unités supplémentaires pour sécuriser le stock.`;
    } else {
        retenir = `Stock optimal avec ${Math.round(context.urgency.ratio * 100)}% de couverture.${context.isPerishable ? ' Surveiller les dates de péremption.' : ''}`;
        faire = `Maintenir le niveau actuel. ${context.events.length > 0 ? `Anticiper une légère hausse pour ${context.events[0].description}.` : 'Pas d\'action urgente requise.'}`;
        quantite = `Stock suffisant. Prochaine commande: ${Math.ceil(context.prediction * 0.5)} unités pour réassort standard.`;
    }
    
    return { retenir, faire, quantite };
}

// ==============================================================================
// UI RENDERING FUNCTIONS
// ==============================================================================
function renderPredictionResults(prediction, storeNbr, itemNbr, date, currentStock) {
    const resultsPanel = document.querySelector('.results-panel');
    if (!resultsPanel) return;
    
    const store = getStoreInfo(storeNbr);
    const item = getItemInfo(itemNbr);
    const isPerishableProduct = isPerishable(itemNbr);
    const urgency = calculateUrgencyIndex(currentStock, prediction);
    const eventImpact = calculateEventImpact(date, store.city);
    const trafficLight = getTrafficLightStatus(urgency, isPerishableProduct);
    
    resultsPanel.innerHTML = `
        <!-- Prediction Card -->
        <div class="result-card fade-in">
            <div class="result-header">
                <h3><i class="fas fa-chart-line"></i> Résultat de Prédiction</h3>
                ${eventImpact.events.length > 0 ? `
                    <span class="event-badge ${eventImpact.events[0].locale === 'National' ? 'national' : ''}">
                        <i class="fas fa-calendar-star"></i>
                        ${eventImpact.events[0].description}
                    </span>
                ` : ''}
            </div>
            <div class="result-body">
                <div class="prediction-result">
                    <div class="prediction-main">
                        <div class="prediction-value">${Math.round(prediction)}</div>
                        <div class="prediction-label">Unités prévues pour le ${formatDate(date)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${store.city}</div>
                        <div class="metric-label">Magasin #${storeNbr} (Type ${store.type})</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${item.family}</div>
                        <div class="metric-label">Produit #${itemNbr}</div>
                    </div>
                </div>
                
                <!-- Traffic Light System -->
                <div class="traffic-light-container">
                    <div class="traffic-light">
                        <div class="light red ${trafficLight.status === 'red' ? 'active' : ''}"></div>
                        <div class="light orange ${trafficLight.status === 'orange' ? 'active' : ''}"></div>
                        <div class="light green ${trafficLight.status === 'green' ? 'active' : ''}"></div>
                    </div>
                    <div class="traffic-info">
                        <div class="traffic-status ${urgency.level}">${urgency.label}</div>
                        <div class="traffic-description">${trafficLight.description}</div>
                    </div>
                </div>
                
                ${isPerishableProduct ? `
                    <div class="perishable-warning">
                        <i class="fas fa-exclamation-triangle"></i>
                        <span><strong>Produit périssable:</strong> Attention aux dates de péremption. Privilégiez la rotation FIFO.</span>
                    </div>
                ` : ''}
            </div>
        </div>
        
        <!-- KPIs Card -->
        <div class="result-card fade-in" style="animation-delay: 0.1s;">
            <div class="result-header">
                <h3><i class="fas fa-tachometer-alt"></i> Indicateurs Clés (KPIs)</h3>
            </div>
            <div class="result-body">
                <div class="kpis-grid">
                    <div class="kpi-card">
                        <div class="kpi-icon ${urgency.level}"><i class="fas fa-warehouse"></i></div>
                        <div class="kpi-value">${currentStock}</div>
                        <div class="kpi-label">Stock Actuel</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon info"><i class="fas fa-bullseye"></i></div>
                        <div class="kpi-value">${Math.round(prediction)}</div>
                        <div class="kpi-label">Demande Prévue</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon ${urgency.level}"><i class="fas fa-percentage"></i></div>
                        <div class="kpi-value ${urgency.level}">${Math.round(urgency.ratio * 100)}%</div>
                        <div class="kpi-label">Taux Couverture</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon ${eventImpact.impact === 'high' ? 'danger' : eventImpact.impact === 'medium' ? 'warning' : 'success'}">
                            <i class="fas fa-calendar-day"></i>
                        </div>
                        <div class="kpi-value">x${eventImpact.multiplier.toFixed(1)}</div>
                        <div class="kpi-label">Impact Événement</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon ${currentStock - prediction < 0 ? 'danger' : 'success'}">
                            <i class="fas fa-boxes"></i>
                        </div>
                        <div class="kpi-value ${currentStock - prediction < 0 ? 'danger' : 'success'}">
                            ${currentStock - Math.round(prediction) > 0 ? '+' : ''}${currentStock - Math.round(prediction)}
                        </div>
                        <div class="kpi-label">Écart Stock</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon info"><i class="fas fa-truck"></i></div>
                        <div class="kpi-value">${Math.max(0, Math.round(prediction * 1.1) - currentStock)}</div>
                        <div class="kpi-label">À Commander</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- AI Recommendations Card (Loading) -->
        <div class="result-card ai-plan-card fade-in" style="animation-delay: 0.2s;" id="ai-recommendations">
            <div class="result-header">
                <h3><i class="fas fa-robot"></i> Plan d'Action IA (Gemini)</h3>
            </div>
            <div class="result-body">
                <div class="ai-loading">
                    <i class="fas fa-spinner"></i>
                    <span>Génération des recommandations en cours...</span>
                </div>
            </div>
        </div>
    `;
    
    // Fetch AI recommendations
    const context = {
        store,
        item,
        date,
        prediction,
        currentStock,
        isPerishable: isPerishableProduct,
        urgency,
        events: eventImpact.events
    };
    
    getAIRecommendations(context).then(recommendations => {
        renderAIRecommendations(recommendations);
    });
}

function renderAIRecommendations(recommendations) {
    const aiCard = document.getElementById('ai-recommendations');
    if (!aiCard) return;
    
    const body = aiCard.querySelector('.result-body');
    body.innerHTML = `
        <div class="ai-plan-grid">
            <div class="ai-section retain">
                <div class="ai-section-header">
                    <i class="fas fa-lightbulb"></i>
                    <span>Ce qu'il faut retenir</span>
                </div>
                <div class="ai-section-content">${recommendations.retenir}</div>
            </div>
            <div class="ai-section action">
                <div class="ai-section-header">
                    <i class="fas fa-tasks"></i>
                    <span>Ce qu'il faut faire</span>
                </div>
                <div class="ai-section-content">${recommendations.faire}</div>
            </div>
            <div class="ai-section quantity">
                <div class="ai-section-header">
                    <i class="fas fa-calculator"></i>
                    <span>En quelle quantité</span>
                </div>
                <div class="ai-section-content">${recommendations.quantite}</div>
            </div>
        </div>
    `;
}

function renderError(message) {
    const resultsPanel = document.querySelector('.results-panel');
    if (!resultsPanel) return;
    
    resultsPanel.innerHTML = `
        <div class="result-card fade-in" style="border-color: var(--status-danger);">
            <div class="result-header" style="background: rgba(239, 68, 68, 0.1);">
                <h3><i class="fas fa-exclamation-circle" style="color: var(--status-danger);"></i> Erreur</h3>
            </div>
            <div class="result-body">
                <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                    ${message}
                </p>
                <p style="color: var(--text-muted); text-align: center; font-size: 0.9rem;">
                    Vérifiez que l'API est en ligne et réessayez.
                </p>
            </div>
        </div>
    `;
}

// ==============================================================================
// FORM HANDLING
// ==============================================================================
function setupDatePicker() {
    const dateInput = document.getElementById('prediction-date');
    if (!dateInput) return;
    
    // Set min/max dates for August 16-31, 2017
    dateInput.min = CONFIG.DATE_RANGE.start;
    dateInput.max = CONFIG.DATE_RANGE.end;
    dateInput.value = CONFIG.DATE_RANGE.start;
}

function populateStoreSelect() {
    const storeSelect = document.getElementById('store-select');
    if (!storeSelect) return;
    
    // Group stores by city
    const storesByCity = {};
    STORES_DATA.forEach(store => {
        if (!storesByCity[store.city]) {
            storesByCity[store.city] = [];
        }
        storesByCity[store.city].push(store);
    });
    
    // Create optgroups
    storeSelect.innerHTML = '<option value="">Sélectionnez un magasin...</option>';
    
    Object.entries(storesByCity).sort().forEach(([city, stores]) => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = city;
        
        stores.forEach(store => {
            const option = document.createElement('option');
            option.value = store.store_nbr;
            option.textContent = `#${store.store_nbr} - Type ${store.type} (Cluster ${store.cluster})`;
            optgroup.appendChild(option);
        });
        
        storeSelect.appendChild(optgroup);
    });
}

function populateItemSelect() {
    const itemSelect = document.getElementById('item-select');
    if (!itemSelect) return;
    
    // Group items by family
    const itemsByFamily = {};
    ITEMS_DATA.forEach(item => {
        if (!itemsByFamily[item.family]) {
            itemsByFamily[item.family] = [];
        }
        itemsByFamily[item.family].push(item);
    });
    
    itemSelect.innerHTML = '<option value="">Sélectionnez un produit...</option>';
    
    // Sort families, prioritizing common ones
    const priorityFamilies = ['GROCERY I', 'BEVERAGES', 'CLEANING', 'DAIRY', 'BREAD/BAKERY', 'DELI', 'EGGS', 'POULTRY'];
    const sortedFamilies = Object.keys(itemsByFamily).sort((a, b) => {
        const aIndex = priorityFamilies.indexOf(a);
        const bIndex = priorityFamilies.indexOf(b);
        if (aIndex !== -1 && bIndex !== -1) return aIndex - bIndex;
        if (aIndex !== -1) return -1;
        if (bIndex !== -1) return 1;
        return a.localeCompare(b);
    });
    
    sortedFamilies.forEach(family => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = family;
        
        itemsByFamily[family].slice(0, 20).forEach(item => {
            const option = document.createElement('option');
            option.value = item.item_nbr;
            option.textContent = `#${item.item_nbr} ${item.perishable === '1' ? '🥬' : '📦'}`;
            optgroup.appendChild(option);
        });
        
        itemSelect.appendChild(optgroup);
    });
}

function updateContextWidgets() {
    const storeNbr = document.getElementById('store-select')?.value;
    const itemNbr = document.getElementById('item-select')?.value;
    const date = document.getElementById('prediction-date')?.value;
    
    // Update store widget
    const storeWidget = document.getElementById('store-widget');
    if (storeWidget && storeNbr) {
        const store = getStoreInfo(storeNbr);
        storeWidget.querySelector('.widget-content').textContent = `${store.city} - Type ${store.type}`;
    }
    
    // Update item widget
    const itemWidget = document.getElementById('item-widget');
    if (itemWidget && itemNbr) {
        const item = getItemInfo(itemNbr);
        const isPerishableProduct = item.perishable === '1';
        itemWidget.querySelector('.widget-content').innerHTML = 
            `${item.family} ${isPerishableProduct ? '<span style="color: var(--status-warning);">🥬 Périssable</span>' : ''}`;
    }
    
    // Update event widget
    const eventWidget = document.getElementById('event-widget');
    if (eventWidget && date && storeNbr) {
        const store = getStoreInfo(storeNbr);
        const events = getHolidaysForDate(date, store.city);
        if (events.length > 0) {
            eventWidget.querySelector('.widget-content').innerHTML = 
                `<span class="highlight">${events[0].description}</span>`;
        } else {
            eventWidget.querySelector('.widget-content').textContent = 'Jour normal';
        }
    }
}

async function handlePrediction(event) {
    event.preventDefault();
    
    const submitBtn = document.getElementById('predict-btn');
    const storeNbr = document.getElementById('store-select')?.value;
    const itemNbr = document.getElementById('item-select')?.value;
    const date = document.getElementById('prediction-date')?.value;
    const currentStock = parseInt(document.getElementById('current-stock')?.value) || 100;
    const onpromotion = document.getElementById('onpromotion')?.checked ? 1 : 0;
    
    // Validation
    if (!storeNbr || !itemNbr || !date) {
        alert('Veuillez remplir tous les champs requis.');
        return;
    }
    
    // Show loading state
    submitBtn.classList.add('loading');
    submitBtn.innerHTML = '<span class="spinner"></span> Analyse en cours...';
    
    try {
        const result = await getPrediction(storeNbr, itemNbr, date, onpromotion);
        
        if (result && result.predicted_unit_sales !== undefined) {
            renderPredictionResults(result.predicted_unit_sales, storeNbr, itemNbr, date, currentStock);
        } else {
            throw new Error('Réponse invalide de l\'API');
        }
    } catch (error) {
        renderError(`Erreur lors de la prédiction: ${error.message}`);
    } finally {
        // Reset button
        submitBtn.classList.remove('loading');
        submitBtn.innerHTML = '<i class="fas fa-magic"></i> Lancer la Prédiction';
    }
}

// ==============================================================================
// INITIALIZATION
// ==============================================================================
document.addEventListener('DOMContentLoaded', async function() {
    console.log('🚀 Favorita Intelligence Hub - Initializing...');
    
    // Load data
    await loadCSVData();
    
    // Check if we're on the dashboard page
    const isDashboard = document.body.classList.contains('dashboard-page');
    
    if (isDashboard) {
        // Setup form elements
        setupDatePicker();
        populateStoreSelect();
        populateItemSelect();
        
        // Check API status
        checkAPIHealth();
        setInterval(checkAPIHealth, 30000); // Check every 30 seconds
        
        // Setup event listeners
        const predictForm = document.getElementById('prediction-form');
        if (predictForm) {
            predictForm.addEventListener('submit', handlePrediction);
        }
        
        // Update context widgets on selection change
        ['store-select', 'item-select', 'prediction-date'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', updateContextWidgets);
            }
        });
    }
    
    // Add smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    console.log('✅ Favorita Intelligence Hub - Ready!');
});
