#!/usr/bin/env python3
"""
Script de instalacion rapida para Windows
"""
import os
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"OK: {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {cmd}")
        print(f"   {e.stderr}")
        return False

def test_installation():
    print("Probando instalacion...")
    
    test_script = '''
try:
    import numpy
    import pandas
    import h5py
    import requests
    import PySide6
    import pyqtgraph
    print("Todos los imports funcionan")
except ImportError as e:
    print(f"Error en import: {e}")
    exit(1)
'''
    
    with open("test_imports.py", "w", encoding='utf-8') as f:
        f.write(test_script)
    
    success = run_cmd("python test_imports.py")
    Path("test_imports.py").unlink(missing_ok=True)
    return success

def download_mun104():
    print("Descargando dataset MUN104...")
    try:
        from core.dataset_loader import DatasetManager
        manager = DatasetManager()
        path = manager.ensure_dataset("MUN104")
        if path:
            print("MUN104 descargado exitosamente")
            return True
    except Exception as e:
        print(f"Error descargando MUN104: {e}")
        print("Se puede descargar despues desde la UI")
    return False

def main():
    print("CONFIGURACION RAPIDA DE FOOTLAB")
    
    # Test instalacion
    if not test_installation():
        print("Faltan dependencias")
        return False
    
    # Test dataset loader
    if Path("core/dataset_loader.py").exists() and Path("core/dataset_loader.py").stat().st_size > 100:
        download_mun104()
    else:
        print("PASO FALTANTE: Copiar core/dataset_loader.py del artifact primero")
    
    print("\nCONFIGURACION COMPLETA")
    print("SIGUIENTES PASOS:")
    print("1. Copiar codigo de artifacts a:")
    print("   - core/dataset_loader.py")
    print("   - ui/dataset_integration.py")
    print("2. Ejecutar: python app.py")
    print("3. O probar: python ui/dataset_integration.py")
    
    return True

if __name__ == "__main__":
    main()
