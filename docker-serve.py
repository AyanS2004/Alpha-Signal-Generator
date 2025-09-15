#!/usr/bin/env python3
"""
Docker server script to serve both frontend and backend.
"""

import os
import sys
import threading
import time
from pathlib import Path
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the main Flask app
from backend.app import app as backend_app

# Create a new Flask app for serving static files
static_app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(static_app)

@static_app.route('/', defaults={'path': ''})
@static_app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend files."""
    if path != "" and os.path.exists(static_app.static_folder + '/' + path):
        return send_from_directory(static_app.static_folder, path)
    else:
        return send_from_directory(static_app.static_folder, 'index.html')

@static_app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy_api(path):
    """Proxy API requests to backend."""
    # This will be handled by the backend app
    pass

def run_backend():
    """Run the backend server."""
    backend_app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

def run_frontend():
    """Run the frontend server."""
    static_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    print("🚀 Starting Alpha Signal Engine in Docker...")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Give backend time to start
    time.sleep(2)
    
    # Start frontend (main thread)
    print("✅ Backend started on port 5001")
    print("✅ Frontend starting on port 5000")
    run_frontend()
