"""
Serveur local pour l'application web Favorita Intelligence Hub.
Inclut un proxy pour contourner les problèmes CORS avec l'API Render.
Lance le serveur puis ouvrez http://localhost:8080 dans votre navigateur.
"""

import http.server
import socketserver
import os
import webbrowser
import threading
import json
import urllib.request
import urllib.error

PORT = 8080
WEBAPP_DIR = os.path.join(os.path.dirname(__file__), 'webapp')
API_URL = "https://favorita-sales-api.onrender.com"

class ProxyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Handler avec proxy API et support CORS pour le développement local."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEBAPP_DIR, **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        # Proxy API requests
        if self.path.startswith('/api/'):
            self.proxy_request('GET')
        else:
            super().do_GET()
    
    def do_POST(self):
        # Proxy API requests
        if self.path.startswith('/api/'):
            self.proxy_request('POST')
        else:
            self.send_error(405, "Method Not Allowed")
    
    def proxy_request(self, method):
        """Proxy requests to the Render API."""
        try:
            # Remove /api prefix and forward to Render
            api_path = self.path[4:]  # Remove '/api'
            target_url = f"{API_URL}{api_path}"
            
            print(f"🔄 Proxy {method} -> {target_url}")
            
            # Read request body for POST
            body = None
            if method == 'POST':
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
            
            # Create request
            req = urllib.request.Request(
                target_url,
                data=body,
                method=method,
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            # Send request to Render API
            with urllib.request.urlopen(req, timeout=60) as response:
                response_data = response.read()
                
                self.send_response(response.status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(response_data)
                
                print(f"✅ Proxy response: {response.status}")
                
        except urllib.error.HTTPError as e:
            print(f"❌ API Error: {e.code} - {e.reason}")
            self.send_response(e.code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_body = e.read() if hasattr(e, 'read') else b'{"error": "API Error"}'
            self.wfile.write(error_body)
            
        except urllib.error.URLError as e:
            print(f"❌ Connection Error: {e.reason}")
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e.reason)}).encode())
            
        except Exception as e:
            print(f"❌ Proxy Error: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

def open_browser():
    """Ouvre le navigateur après un délai."""
    import time
    time.sleep(1)
    webbrowser.open(f'http://localhost:{PORT}')

if __name__ == '__main__':
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           FAVORITA INTELLIGENCE HUB - SERVEUR LOCAL          ║
╠══════════════════════════════════════════════════════════════╣
║  🌐 URL: http://localhost:{PORT}                               ║
║  🔄 Proxy: /api/* -> Render API                               ║
║  ⏹️  Arrêter: Ctrl+C                                          ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Open browser in a thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    with socketserver.TCPServer(("", PORT), ProxyHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Serveur arrêté.")
