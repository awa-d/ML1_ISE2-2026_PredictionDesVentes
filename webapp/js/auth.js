/* ============================================================================
   FAVORITA INTELLIGENCE HUB - AUTHENTICATION MODULE
   Handles user authentication and session management
   ============================================================================ */

// ==============================================================================
// CONFIGURATION DES IDENTIFIANTS
// ==============================================================================
// ⚠️ IMPORTANT: Modifiez ces valeurs pour configurer vos identifiants
// Pour une sécurité renforcée en production, utilisez un backend avec hachage de mots de passe

const AUTH_CONFIG = {
    // ========================================
    // 👤 CONFIGUREZ VOS IDENTIFIANTS ICI
    // ========================================
    users: [
        {
            username: 'admin',           // Nom d'utilisateur
            password: 'favorita2026'     // Mot de passe
        },
        {
            username: 'manager',         // Utilisateur supplémentaire
            password: 'manager123'       // Mot de passe
        }
        // Ajoutez d'autres utilisateurs si nécessaire:
        // { username: 'nouvel_utilisateur', password: 'mot_de_passe' }
    ],
    
    // ========================================
    // ⚙️ PARAMÈTRES DE SESSION
    // ========================================
    sessionKey: 'favorita_auth_session',
    sessionDuration: 8 * 60 * 60 * 1000,  // 8 heures en millisecondes
    
    // Pages qui ne nécessitent pas d'authentification
    publicPages: ['login.html']
};

// ==============================================================================
// FONCTIONS D'AUTHENTIFICATION
// ==============================================================================

/**
 * Vérifie si les identifiants sont valides
 */
function validateCredentials(username, password) {
    return AUTH_CONFIG.users.some(
        user => user.username === username && user.password === password
    );
}

/**
 * Crée une session utilisateur
 */
function createSession(username) {
    const session = {
        username: username,
        loginTime: Date.now(),
        expiresAt: Date.now() + AUTH_CONFIG.sessionDuration
    };
    
    localStorage.setItem(AUTH_CONFIG.sessionKey, JSON.stringify(session));
    return session;
}

/**
 * Récupère la session actuelle
 */
function getSession() {
    try {
        const sessionData = localStorage.getItem(AUTH_CONFIG.sessionKey);
        if (!sessionData) return null;
        
        const session = JSON.parse(sessionData);
        
        // Vérifie si la session a expiré
        if (Date.now() > session.expiresAt) {
            destroySession();
            return null;
        }
        
        return session;
    } catch (error) {
        console.error('Erreur de session:', error);
        return null;
    }
}

/**
 * Détruit la session (déconnexion)
 */
function destroySession() {
    localStorage.removeItem(AUTH_CONFIG.sessionKey);
}

/**
 * Vérifie si l'utilisateur est authentifié
 */
function isAuthenticated() {
    return getSession() !== null;
}

/**
 * Obtient le nom de la page actuelle
 */
function getCurrentPage() {
    const path = window.location.pathname;
    return path.substring(path.lastIndexOf('/') + 1) || 'index.html';
}

/**
 * Vérifie si la page actuelle est publique
 */
function isPublicPage() {
    const currentPage = getCurrentPage();
    return AUTH_CONFIG.publicPages.includes(currentPage);
}

/**
 * Redirige vers la page de connexion
 */
function redirectToLogin() {
    if (!isPublicPage()) {
        window.location.href = 'login.html';
    }
}

/**
 * Redirige vers le dashboard après connexion
 */
function redirectToDashboard() {
    window.location.href = 'index.html';
}

/**
 * Déconnecte l'utilisateur
 */
function logout() {
    destroySession();
    redirectToLogin();
}

// ==============================================================================
// PROTECTION DES PAGES
// ==============================================================================

/**
 * Protège la page actuelle - redirige si non authentifié
 */
function protectPage() {
    if (!isPublicPage() && !isAuthenticated()) {
        redirectToLogin();
        return false;
    }
    return true;
}

// ==============================================================================
// GESTION DU FORMULAIRE DE CONNEXION
// ==============================================================================

function initLoginForm() {
    const form = document.getElementById('login-form');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const loginBtn = document.getElementById('login-btn');
    const togglePassword = document.getElementById('toggle-password');
    const passwordInput = document.getElementById('password');
    
    if (!form) return;
    
    // Si déjà connecté, rediriger vers le dashboard
    if (isAuthenticated()) {
        redirectToDashboard();
        return;
    }
    
    // Toggle password visibility
    if (togglePassword && passwordInput) {
        togglePassword.addEventListener('click', () => {
            const type = passwordInput.type === 'password' ? 'text' : 'password';
            passwordInput.type = type;
            togglePassword.querySelector('i').classList.toggle('fa-eye');
            togglePassword.querySelector('i').classList.toggle('fa-eye-slash');
        });
    }
    
    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value;
        
        // Disable button during validation
        loginBtn.disabled = true;
        loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connexion...';
        
        // Simulate a small delay for UX
        await new Promise(resolve => setTimeout(resolve, 500));
        
        if (validateCredentials(username, password)) {
            // Success - create session and redirect
            createSession(username);
            loginBtn.innerHTML = '<i class="fas fa-check"></i> Connecté!';
            
            setTimeout(() => {
                redirectToDashboard();
            }, 500);
        } else {
            // Error - show message
            errorText.textContent = 'Nom d\'utilisateur ou mot de passe incorrect';
            errorMessage.classList.add('show');
            
            // Shake animation
            form.classList.add('shake');
            setTimeout(() => form.classList.remove('shake'), 300);
            
            // Reset button
            loginBtn.disabled = false;
            loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Se connecter';
            
            // Focus on username
            document.getElementById('username').focus();
        }
    });
    
    // Hide error on input change
    ['username', 'password'].forEach(id => {
        document.getElementById(id)?.addEventListener('input', () => {
            errorMessage.classList.remove('show');
        });
    });
}

// ==============================================================================
// AJOUT DU BOUTON DE DÉCONNEXION À LA NAVBAR
// ==============================================================================

function addLogoutButton() {
    if (isPublicPage() || !isAuthenticated()) return;
    
    const session = getSession();
    const navContainer = document.querySelector('.nav-container');
    
    if (navContainer) {
        // Créer le bouton de déconnexion
        const logoutDiv = document.createElement('div');
        logoutDiv.className = 'nav-user';
        logoutDiv.innerHTML = `
            <span class="user-name" style="color: var(--text-secondary); font-size: 0.875rem; margin-right: 1rem;">
                <i class="fas fa-user-circle" style="color: var(--accent-primary);"></i>
                ${session.username}
            </span>
            <button id="logout-btn" class="btn-logout" style="
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: var(--status-danger);
                padding: 0.5rem 1rem;
                border-radius: var(--radius-md);
                cursor: pointer;
                font-size: 0.875rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                transition: var(--transition-fast);
            ">
                <i class="fas fa-sign-out-alt"></i>
                Déconnexion
            </button>
        `;
        
        // Insérer avant le statut API
        const apiStatus = navContainer.querySelector('.api-status');
        if (apiStatus) {
            navContainer.insertBefore(logoutDiv, apiStatus);
        } else {
            navContainer.appendChild(logoutDiv);
        }
        
        // Ajouter l'événement de déconnexion
        document.getElementById('logout-btn')?.addEventListener('click', logout);
        
        // Style hover
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('mouseenter', () => {
                logoutBtn.style.background = 'rgba(239, 68, 68, 0.2)';
            });
            logoutBtn.addEventListener('mouseleave', () => {
                logoutBtn.style.background = 'rgba(239, 68, 68, 0.1)';
            });
        }
    }
}

// ==============================================================================
// INITIALISATION
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const currentPage = getCurrentPage();
    
    if (currentPage === 'login.html') {
        // Page de connexion
        initLoginForm();
    } else {
        // Autres pages - vérifier l'authentification
        if (!protectPage()) return;
        
        // Ajouter le bouton de déconnexion
        addLogoutButton();
    }
});

// Exporter les fonctions pour usage externe
window.FavoritaAuth = {
    isAuthenticated,
    getSession,
    logout,
    protectPage
};
