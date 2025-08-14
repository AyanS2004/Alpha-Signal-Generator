#!/usr/bin/env python3
"""
Startup script for Alpha Signal Engine.
Launches both frontend and backend servers.
"""

import os
import sys
import subprocess
import time
import webbrowser
import signal
import threading
from pathlib import Path
import io

# Ensure UTF-8 capable output on Windows consoles
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

class AlphaSignalApp:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = False
        
    def start_backend(self):
        """Start the Flask backend server."""
        print("🚀 Starting backend server...")
        
        backend_dir = Path(__file__).parent / "backend"
        
        # Create virtual environment if needed
        venv_dir = backend_dir / "venv"
        if not venv_dir.exists():
            print("📦 Creating virtual environment for backend...")
            # Create venv at absolute path to avoid cwd race conditions
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        
        # Install requirements
        requirements_file = backend_dir / "requirements.txt"
        if requirements_file.exists():
            print("📦 Installing backend dependencies...")
            if os.name == 'nt':  # Windows
                # Use python -m pip to avoid path issues
                python_cmd = str(venv_dir / "Scripts" / "python.exe")
                subprocess.run([python_cmd, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
            else:  # Unix/Linux/Mac
                pip_cmd = [str(venv_dir / "bin" / "pip")]
                subprocess.run(pip_cmd + ["install", "-r", str(requirements_file)], check=True)
        
        # Start backend server
        if os.name == 'nt':  # Windows
            python_cmd = str(venv_dir / "Scripts" / "python.exe")
        else:  # Unix/Linux/Mac
            python_cmd = str(venv_dir / "bin" / "python")
        
        self.backend_process = subprocess.Popen(
            [python_cmd, "app.py"],
            cwd=backend_dir,
            stdout=None,
            stderr=None
        )
        
        print("✅ Backend server started on http://localhost:5000")
        
    def start_frontend(self):
        """Start the React frontend server."""
        print("🚀 Starting frontend server...")
        
        frontend_dir = Path(__file__).parent / "frontend"
        
        # Ensure dependencies (including newly added devDependencies) are installed
        def run_npm_install():
            print("📦 Installing/updating frontend dependencies...")
            if os.name == 'nt':
                subprocess.run("npm install --force", shell=True, check=True, cwd=frontend_dir)
            else:
                subprocess.run(["npm", "install", "--force"], check=True, cwd=frontend_dir)

        if not (frontend_dir / "node_modules").exists():
            run_npm_install()
        else:
            # If Tailwind isn't installed yet (new devDependency), install
            try:
                if os.name == 'nt':
                    result = subprocess.run("npm ls tailwindcss --depth=0", shell=True, cwd=frontend_dir)
                else:
                    result = subprocess.run(["npm", "ls", "tailwindcss", "--depth=0"], cwd=frontend_dir) 
                if result.returncode != 0:
                    run_npm_install()
            except Exception:
                run_npm_install()
        
        # Start frontend server
        if os.name == 'nt':  # Windows
            self.frontend_process = subprocess.Popen(
                "npm start",
                shell=True,
                cwd=frontend_dir,
                stdout=None,
                stderr=None
            )
        else:
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=frontend_dir,
                stdout=None,
                stderr=None
            )
        
        print("✅ Frontend server started on http://localhost:3000")
        
    def wait_for_servers(self):
        """Wait for servers to be ready."""
        # Prefer requests, but fall back to urllib if unavailable
        try:
            import requests  # type: ignore
            def http_get(url: str, timeout: float):
                return requests.get(url, timeout=timeout)
            def is_ok(resp) -> bool:
                return getattr(resp, 'status_code', None) == 200
        except Exception:
            import urllib.request
            def http_get(url: str, timeout: float):
                return urllib.request.urlopen(url, timeout=timeout)
            def is_ok(resp) -> bool:
                try:
                    return getattr(resp, 'status', None) == 200
                except Exception:
                    return False
        
        print("⏳ Waiting for servers to start...")
        
        # Wait for backend
        backend_ready = False
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = http_get("http://localhost:5000/api/health", timeout=1)
                if is_ok(response):
                    backend_ready = True
                    print("✅ Backend server is ready!")
                    break
            except:
                time.sleep(1)
        
        if not backend_ready:
            print("❌ Backend server failed to start")
            return False
        
        # Wait for frontend
        frontend_ready = False
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = http_get("http://localhost:3000", timeout=1)
                if is_ok(response):
                    frontend_ready = True
                    print("✅ Frontend server is ready!")
                    break
            except:
                time.sleep(1)
        
        if not frontend_ready:
            print("❌ Frontend server failed to start")
            return False
        
        return True
    
    def open_browser(self):
        """Open the application in the default browser."""
        print("🌐 Opening application in browser...")
        time.sleep(2)  # Give servers a moment to fully start
        webbrowser.open("http://localhost:3000")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down Alpha Signal Engine...")
        self.stop()
        sys.exit(0)
        
    def stop(self):
        """Stop all servers."""
        self.running = False
        
        if self.backend_process:
            print("🛑 Stopping backend server...")
            self.backend_process.terminate()
            self.backend_process.wait()
            
        if self.frontend_process:
            print("🛑 Stopping frontend server...")
            self.frontend_process.terminate()
            self.frontend_process.wait()
            
        print("✅ All servers stopped")
        
    def run(self):
        """Run the complete application."""
        print("🚀 Alpha Signal Engine - Starting...")
        print("=" * 50)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start backend in a separate thread
            backend_thread = threading.Thread(target=self.start_backend)
            backend_thread.daemon = True
            backend_thread.start()
            
            # Start frontend in a separate thread
            frontend_thread = threading.Thread(target=self.start_frontend)
            frontend_thread.daemon = True
            frontend_thread.start()
            
            # Wait for servers to be ready
            if self.wait_for_servers():
                self.running = True
                self.open_browser()
                
                print("\n🎉 Alpha Signal Engine is running!")
                print("📊 Frontend: http://localhost:3000")
                print("🔧 Backend:  http://localhost:5000")
                print("\nPress Ctrl+C to stop the application")
                
                # Keep the main thread alive
                while self.running:
                    time.sleep(1)
                    
            else:
                print("❌ Failed to start servers")
                self.stop()
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            self.stop()
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            self.stop()

def check_prerequisites():
    """Check if all prerequisites are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    # Check Node.js
    try:
        # Use shell=True on Windows to ensure PATH is properly resolved
        if os.name == 'nt':  # Windows
            result = subprocess.run("node --version", shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Node.js is not installed")
            return False
        print(f"✅ Node.js: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Node.js is not installed")
        return False
    
    # Check npm
    try:
        # Use shell=True on Windows to ensure PATH is properly resolved
        if os.name == 'nt':  # Windows
            result = subprocess.run("npm --version", shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ npm is not installed")
            return False
        print(f"✅ npm: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ npm is not installed")
        return False
    
    print("✅ All prerequisites are satisfied")
    return True

def main():
    """Main entry point."""
    print("🎯 Alpha Signal Engine - Startup Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Please install the missing prerequisites and try again")
        sys.exit(1)
    
    # Create and run the application
    app = AlphaSignalApp()
    app.run()

if __name__ == "__main__":
    main()
