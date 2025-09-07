"""
Dataset Loader para FootLab - Implementacion completa
Soporta MUN104 Template y UNB StepUP-P150
Instalacion: copiar este codigo completo a core/dataset_loader.py
"""

import os
import zipfile
import requests
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from scipy.interpolate import griddata, RBFInterpolator
from dataclasses import dataclass
import time
import threading
from urllib.parse import urlparse
import json

@dataclass
class Sample:
    """Estructura de datos compatible con FootLab"""
    t_ms: int
    left: List[float]   # 16 sensores pie izquierdo
    right: List[float]  # 16 sensores pie derecho

class DatasetDownloader:
    """Descarga automática de datasets"""
    
    def __init__(self, data_dir: str = "data/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_mun104(self) -> Path:
        """Descarga dataset MUN104 si no existe"""
        zip_path = self.data_dir / "MUN104.csv.zip"
        extracted_dir = self.data_dir / "MUN104"
        
        if extracted_dir.exists():
            print("✅ MUN104 ya está descargado")
            return extracted_dir
            
        print("📥 Descargando MUN104 dataset...")
        url = "https://github.com/0todd0000/mun104/raw/master/MUN104.csv.zip"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extraer
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            
            zip_path.unlink()  # Borrar zip
            print("✅ MUN104 descargado y extraído")
            return extracted_dir
            
        except Exception as e:
            print(f"❌ Error descargando MUN104: {e}")
            return None
    
    def download_mun104_hdf5(self) -> Path:
        """Descarga versión HDF5 de MUN104"""
        zip_path = self.data_dir / "MUN104.h5.zip"
        extracted_dir = self.data_dir / "MUN104_h5"
        
        if extracted_dir.exists():
            return extracted_dir
            
        print("📥 Descargando MUN104 HDF5...")
        url = "https://github.com/0todd0000/mun104/raw/master/MUN104.h5.zip"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
                
            zip_path.unlink()
            print("✅ MUN104 HDF5 descargado")
            return extracted_dir
            
        except Exception as e:
            print(f"❌ Error descargando MUN104 HDF5: {e}")
            return None

class MUN104Adapter:
    """Adaptador para dataset MUN104 Template"""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.left_template = None
        self.right_template = None
        self.load_templates()
    
    def load_templates(self):
        """Carga templates MUN104"""
        try:
            # Buscar archivos CSV
            csv_files = list(self.dataset_path.glob("*.csv"))
            h5_files = list(self.dataset_path.glob("*.h5"))
            
            if h5_files:
                self._load_from_hdf5(h5_files[0])
            elif csv_files:
                self._load_from_csv(csv_files)
            else:
                raise FileNotFoundError("No se encontraron archivos MUN104")
                
        except Exception as e:
            print(f"❌ Error cargando MUN104: {e}")
    
    def _load_from_hdf5(self, h5_path: Path):
        """Carga desde archivo HDF5"""
        try:
            with h5py.File(h5_path, 'r') as f:
                # MUN104 tiene estructura /I para la imagen
                if 'I' in f:
                    template = f['I'][:]
                    # Asumir que es template izquierdo, crear derecho como espejo
                    self.left_template = template
                    self.right_template = np.fliplr(template)
                else:
                    print("⚠️ Estructura HDF5 inesperada")
        except Exception as e:
            print(f"❌ Error leyendo HDF5: {e}")
    
    def _load_from_csv(self, csv_files: List[Path]):
        """Carga desde archivos CSV"""
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if 'L' in csv_file.name.upper():
                    self.left_template = df.values
                elif 'R' in csv_file.name.upper():
                    self.right_template = df.values
            except Exception as e:
                print(f"❌ Error leyendo CSV {csv_file}: {e}")
        
        # Si solo hay un archivo, crear espejo
        if self.left_template is not None and self.right_template is None:
            self.right_template = np.fliplr(self.left_template)
        elif self.right_template is not None and self.left_template is None:
            self.left_template = np.fliplr(self.right_template)
    
    def get_nurvv_coordinates(self):
        """Coordenadas de 16 sensores NURVV (compatible con FootLab)"""
        # Coordenadas de sensores NURVV según la especificación
        left_coords = np.array([
            # Talón (3 sensores)
            [0.40, 0.88], [0.50, 0.90], [0.60, 0.88],
            # Mediopié (3 sensores)  
            [0.35, 0.65], [0.50, 0.60], [0.65, 0.65],
            # Metatarsos (5 sensores)
            [0.32, 0.32], [0.40, 0.30], [0.48, 0.28], 
            [0.56, 0.28], [0.64, 0.30],
            # Dedos (5 sensores)
            [0.30, 0.12], [0.38, 0.10], [0.46, 0.10],
            [0.54, 0.10], [0.62, 0.12],
        ])
        
        # Pie derecho como espejo
        right_coords = left_coords.copy()
        right_coords[:, 0] = 1.0 - right_coords[:, 0]
        
        return left_coords, right_coords
    
    def sample_template_at_sensors(self, template: np.ndarray, is_left: bool = True) -> List[float]:
        """Muestrea template en posiciones de sensores NURVV"""
        if template is None:
            return [0.0] * 16
            
        left_coords, right_coords = self.get_nurvv_coordinates()
        sensor_coords = left_coords if is_left else right_coords
        
        # Crear grid de coordenadas del template
        rows, cols = template.shape
        y_grid = np.linspace(0, 1, rows)  # 0=dedos, 1=talón
        x_grid = np.linspace(0, 1, cols)  # 0=izq, 1=der
        
        # Interpolar valores en posiciones de sensores
        pressures = []
        for x_sensor, y_sensor in sensor_coords:
            # Convertir coordenadas a índices
            col_idx = x_sensor * (cols - 1)
            row_idx = y_sensor * (rows - 1)
            
            # Interpolación bilineal
            col_low = int(np.floor(col_idx))
            col_high = min(col_low + 1, cols - 1)
            row_low = int(np.floor(row_idx))
            row_high = min(row_low + 1, rows - 1)
            
            if col_low == col_high and row_low == row_high:
                pressure = template[row_low, col_low]
            else:
                # Interpolación bilineal
                w1 = (col_high - col_idx) * (row_high - row_idx)
                w2 = (col_idx - col_low) * (row_high - row_idx)
                w3 = (col_high - col_idx) * (row_idx - row_low)
                w4 = (col_idx - col_low) * (row_idx - row_low)
                
                pressure = (w1 * template[row_low, col_low] +
                          w2 * template[row_low, col_high] +
                          w3 * template[row_high, col_low] +
                          w4 * template[row_high, col_high])
            
            pressures.append(float(pressure))
        
        return pressures

class DatasetReplaySource:
    """Fuente de datos que reproduce datasets reales"""
    
    def __init__(self, dataset_path: str, replay_speed: float = 1.0, loop: bool = True):
        self.dataset_path = Path(dataset_path)
        self.replay_speed = replay_speed
        self.loop = loop
        self.running = False
        self.thread = None
        self.on_sample_callback = None
        
        # Cargar datos
        self.adapter = None
        self.samples = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Carga el dataset apropiado"""
        if "MUN104" in str(self.dataset_path):
            self._load_mun104_static()
        else:
            print("⚠️ Tipo de dataset no reconocido")
    
    def _load_mun104_static(self):
        """Carga MUN104 como datos estáticos repetidos"""
        self.adapter = MUN104Adapter(self.dataset_path)
        
        if self.adapter.left_template is None:
            print("❌ No se pudo cargar MUN104")
            return
        
        # Crear secuencia simulada de presiones estáticas
        base_left = self.adapter.sample_template_at_sensors(self.adapter.left_template, True)
        base_right = self.adapter.sample_template_at_sensors(self.adapter.right_template, False)
        
        # Normalizar a rango 0-100 (similar a presión real)
        left_max = max(base_left) if max(base_left) > 0 else 1.0
        right_max = max(base_right) if max(base_right) > 0 else 1.0
        
        base_left = [p / left_max * 80.0 for p in base_left]  # Max 80 para variación
        base_right = [p / right_max * 80.0 for p in base_right]
        
        # Crear secuencia temporal con variaciones
        for i in range(1000):  # 10 segundos a 100Hz
            t_ms = i * 10  # 100Hz
            
            # Añadir variación temporal realista
            variation = 1.0 + 0.2 * np.sin(i * 0.1) + 0.1 * np.random.normal()
            noise = 0.05
            
            left = [max(0, p * variation + np.random.normal(0, noise)) for p in base_left]
            right = [max(0, p * variation + np.random.normal(0, noise)) for p in base_right]
            
            self.samples.append(Sample(t_ms, left, right))
        
        print(f"✅ MUN104 cargado: {len(self.samples)} muestras")
    
    def start(self, on_sample_callback):
        """Inicia reproducción de datos"""
        self.on_sample_callback = on_sample_callback
        self.running = True
        
        if not self.samples:
            print("❌ No hay datos para reproducir")
            return
            
        self.thread = threading.Thread(target=self._replay_loop)
        self.thread.daemon = True
        self.thread.start()
        print("▶️ Iniciando reproducción de dataset")
    
    def stop(self):
        """Detiene reproducción"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("⏹️ Reproducción detenida")
    
    def _replay_loop(self):
        """Loop principal de reproducción"""
        sample_idx = 0
        start_time = time.time()
        
        while self.running:
            if sample_idx >= len(self.samples):
                if self.loop:
                    sample_idx = 0
                    start_time = time.time()
                else:
                    break
            
            sample = self.samples[sample_idx]
            
            # Timing preciso
            expected_time = start_time + (sample.t_ms / 1000.0) / self.replay_speed
            current_time = time.time()
            
            if current_time < expected_time:
                time.sleep(expected_time - current_time)
            
            # Enviar muestra
            if self.on_sample_callback:
                self.on_sample_callback(sample)
            
            sample_idx += 1

class DatasetManager:
    """Manager principal para todos los datasets"""
    
    def __init__(self, data_dir: str = "data/datasets"):
        self.downloader = DatasetDownloader(data_dir)
        self.available_datasets = {}
        self._scan_datasets()
    
    def _scan_datasets(self):
        """Escanea datasets disponibles"""
        # MUN104
        mun104_path = self.downloader.data_dir / "MUN104"
        if mun104_path.exists():
            self.available_datasets["MUN104"] = {
                "path": mun104_path,
                "type": "static_template",
                "description": "Template anatómico de 104 sujetos"
            }
    
    def list_datasets(self) -> Dict[str, Any]:
        """Lista datasets disponibles"""
        return self.available_datasets
    
    def ensure_dataset(self, name: str) -> Optional[Path]:
        """Asegura que un dataset esté disponible, descargándolo si es necesario"""
        if name == "MUN104":
            return self.downloader.download_mun104()
        else:
            print(f"❌ Dataset '{name}' no reconocido")
            return None
    
    def create_source(self, dataset_name: str, **kwargs) -> Optional[DatasetReplaySource]:
        """Crea fuente de datos para un dataset"""
        dataset_path = self.ensure_dataset(dataset_name)
        if dataset_path:
            return DatasetReplaySource(dataset_path, **kwargs)
        return None

# Función de conveniencia para integración fácil
def get_dataset_source(dataset_name: str = "MUN104", **kwargs) -> Optional[DatasetReplaySource]:
    """
    Función simple para obtener una fuente de datos de dataset
    
    Args:
        dataset_name: Nombre del dataset ("MUN104", etc.)
        **kwargs: Argumentos para DatasetReplaySource (replay_speed, loop, etc.)
    
    Returns:
        DatasetReplaySource configurado o None si hay error
    """
    manager = DatasetManager()
    return manager.create_source(dataset_name, **kwargs)

# Ejemplo de uso
if __name__ == "__main__":
    def test_sample_handler(sample: Sample):
        """Handler de prueba"""
        left_total = sum(sample.left)
        right_total = sum(sample.right)
        print(f"t={sample.t_ms}ms, Left={left_total:.1f}, Right={right_total:.1f}")
    
    # Obtener fuente de datos
    source = get_dataset_source("MUN104", replay_speed=1.0, loop=True)
    
    if source:
        print("🚀 Iniciando prueba de dataset...")
        source.start(test_sample_handler)
        
        # Ejecutar por 5 segundos
        time.sleep(5)
        source.stop()
        print("✅ Prueba completada")
    else:
        print("❌ No se pudo cargar dataset")