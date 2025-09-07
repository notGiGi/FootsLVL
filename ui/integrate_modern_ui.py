# integrate_modern_ui.py
"""
Script para integrar la nueva UI moderna con el sistema existente
"""

import os
import shutil

def backup_old_ui():
    """Hace backup de la UI anterior"""
    if os.path.exists("ui/main_window.py"):
        shutil.copy("ui/main_window.py", "ui/main_window_old.py")
        print("✓ Backup creado: ui/main_window_old.py")

def create_app_modern():
    """Crea nuevo punto de entrada para la app moderna"""
    
    app_code = '''# app_modern.py
"""
FootLab Premium - Aplicación con UI moderna
"""

import sys
from PySide6.QtWidgets import QApplication
from ui.modern_ui import ModernFootLabUI
from core.simulator import SimulatorSource
from core.ble_nurvv import NurvvBleSource
import numpy as np

class FootLabApp(ModernFootLabUI):
    """Aplicación FootLab con integración completa"""
    
    def __init__(self):
        super().__init__()
        
        # Fuente de datos
        self.data_source = None
        
        # Buffer de datos
        self.pressure_buffer_left = []
        self.pressure_buffer_right = []
        
        # Configurar callbacks
        self.setup_connections()
        
    def setup_connections(self):
        """Conecta señales adicionales"""
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        
    def on_source_changed(self, source_type):
        """Maneja cambio de fuente de datos"""
        if source_type == "NURVV BLE":
            # Preparado para BLE real
            self.status_text.setText("Ready for BLE")
        elif source_type == "Simulator":
            self.status_text.setText("Simulator Mode")
        
    def start_acquisition(self):
        """Inicia adquisición con la fuente seleccionada"""
        source_type = self.source_combo.currentText()
        
        # Detener fuente anterior si existe
        if self.data_source:
            self.data_source.stop()
            
        if source_type == "Simulator":
            # Usar simulador mejorado
            self.data_source = SimulatorSource(
                n_sensors=16,
                freq=100,
                base_amp=35.0,  # Ajustado para visualización
                noise=0.1,      # Bajo ruido
                cadence_hz=1.0  # Ciclo completo cada segundo
            )
            self.data_source.start(self.on_sample_received)
            
        elif source_type == "NURVV BLE":
            # Usar BLE real
            self.data_source = NurvvBleSource(
                n_sensors=16,
                freq=100,
                auto_connect=True
            )
            self.data_source.start(self.on_sample_received)
            
        # Actualizar UI
        super().start_acquisition()
        
    def stop_acquisition(self):
        """Detiene adquisición"""
        if self.data_source:
            self.data_source.stop()
            self.data_source = None
            
        super().stop_acquisition()
        
    def on_sample_received(self, sample):
        """Callback cuando llega una muestra de datos"""
        # sample = {"t_ms": int, "left": list[16], "right": list[16]}
        
        left_pressures = np.array(sample["left"])
        right_pressures = np.array(sample["right"])
        
        # Crear grid de presión interpolado
        left_grid = self.interpolate_to_grid(left_pressures, is_left=True)
        right_grid = self.interpolate_to_grid(right_pressures, is_left=False)
        
        # Calcular CoP
        cop_left = self.calculate_cop(left_pressures, is_left=True)
        cop_right = self.calculate_cop(right_pressures, is_left=False)
        
        # Actualizar heatmaps
        self.heatmap_left.update_heatmap(left_grid, cop_left[0], cop_left[1])
        self.heatmap_right.update_heatmap(right_grid, cop_right[0], cop_right[1])
        
        # Calcular métricas
        self.update_metrics_from_data(left_pressures, right_pressures)
        
        # Detectar fase de marcha
        self.detect_gait_phase(left_pressures, right_pressures)
        
    def interpolate_to_grid(self, pressures, is_left=True, grid_size=(48, 32)):
        """
        Interpola los 16 sensores a un grid denso para visualización
        """
        from scipy.interpolate import griddata
        
        # Posiciones de los 16 sensores NURVV (normalizadas 0-1)
        if is_left:
            sensor_positions = np.array([
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
        else:
            # Espejo para pie derecho
            sensor_positions = np.array([
                [0.60, 0.88], [0.50, 0.90], [0.40, 0.88],
                [0.65, 0.65], [0.50, 0.60], [0.35, 0.65],
                [0.68, 0.32], [0.60, 0.30], [0.52, 0.28],
                [0.44, 0.28], [0.36, 0.30],
                [0.70, 0.12], [0.62, 0.10], [0.54, 0.10],
                [0.46, 0.10], [0.38, 0.12],
            ])
        
        # Escalar posiciones al tamaño del grid
        sensor_x = sensor_positions[:, 0] * grid_size[1]
        sensor_y = sensor_positions[:, 1] * grid_size[0]
        
        # Crear grid de salida
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, grid_size[1], grid_size[1]),
            np.linspace(0, grid_size[0], grid_size[0])
        )
        
        # Interpolar usando RBF (Radial Basis Function)
        try:
            grid_values = griddata(
                (sensor_x, sensor_y),
                pressures[:16],  # Asegurar 16 valores
                (grid_x, grid_y),
                method='cubic',
                fill_value=0
            )
            
            # Aplicar suavizado gaussiano para mejor visualización
            from scipy.ndimage import gaussian_filter
            grid_values = gaussian_filter(grid_values, sigma=1.5)
            
            # Enmascarar valores negativos
            grid_values[grid_values < 0] = 0
            
        except:
            # Fallback si falla la interpolación
            grid_values = np.zeros((grid_size[0], grid_size[1]))
            
        return grid_values
    
    def calculate_cop(self, pressures, is_left=True):
        """Calcula centro de presión"""
        # Usar las mismas posiciones de sensores
        if is_left:
            positions = np.array([
                [0.40, 0.88], [0.50, 0.90], [0.60, 0.88],
                [0.35, 0.65], [0.50, 0.60], [0.65, 0.65],
                [0.32, 0.32], [0.40, 0.30], [0.48, 0.28],
                [0.56, 0.28], [0.64, 0.30],
                [0.30, 0.12], [0.38, 0.10], [0.46, 0.10],
                [0.54, 0.10], [0.62, 0.12],
            ])
        else:
            positions = np.array([
                [0.60, 0.88], [0.50, 0.90], [0.40, 0.88],
                [0.65, 0.65], [0.50, 0.60], [0.35, 0.65],
                [0.68, 0.32], [0.60, 0.30], [0.52, 0.28],
                [0.44, 0.28], [0.36, 0.30],
                [0.70, 0.12], [0.62, 0.10], [0.54, 0.10],
                [0.46, 0.10], [0.38, 0.12],
            ])
        
        total_pressure = np.sum(pressures[:16])
        if total_pressure > 0:
            cop_x = np.sum(positions[:, 0] * pressures[:16]) / total_pressure
            cop_y = np.sum(positions[:, 1] * pressures[:16]) / total_pressure
            return (cop_x * 32, cop_y * 48)  # Escalar al grid
        return (16, 24)  # Centro por defecto
    
    def update_metrics_from_data(self, left_pressures, right_pressures):
        """Actualiza métricas basadas en datos reales"""
        # Peak pressure
        peak_left = np.max(left_pressures)
        peak_right = np.max(right_pressures)
        peak_total = max(peak_left, peak_right)
        self.metrics_panel.update_metric("Peak Pressure", peak_total)
        
        # Contact area (número de sensores activos)
        threshold = 5.0  # Umbral mínimo de presión
        active_left = np.sum(left_pressures > threshold)
        active_right = np.sum(right_pressures > threshold)
        contact_area = (active_left + active_right) * 2.5  # cm² aproximados por sensor
        self.metrics_panel.update_metric("Contact Area", contact_area)
        
        # Asymmetry
        total_left = np.sum(left_pressures)
        total_right = np.sum(right_pressures)
        if total_left + total_right > 0:
            asymmetry = abs(total_left - total_right) / (total_left + total_right) * 100
            self.metrics_panel.update_metric("Asymmetry", asymmetry)
    
    def detect_gait_phase(self, left_pressures, right_pressures):
        """Detecta la fase de marcha basado en distribución de presión"""
        # Dividir pie en zonas
        heel_indices = [0, 1, 2]  # Sensores del talón
        midfoot_indices = [3, 4, 5]  # Mediopié
        forefoot_indices = [6, 7, 8, 9, 10]  # Antepié
        toe_indices = [11, 12, 13, 14, 15]  # Dedos
        
        # Calcular presión por zona (pie con más carga)
        if np.sum(left_pressures) > np.sum(right_pressures):
            pressures = left_pressures
        else:
            pressures = right_pressures
            
        heel_pressure = np.mean([pressures[i] for i in heel_indices])
        midfoot_pressure = np.mean([pressures[i] for i in midfoot_indices])
        forefoot_pressure = np.mean([pressures[i] for i in forefoot_indices])
        toe_pressure = np.mean([pressures[i] for i in toe_indices])
        
        # Estimar fase basado en distribución
        total = heel_pressure + midfoot_pressure + forefoot_pressure + toe_pressure
        
        if total > 0:
            heel_ratio = heel_pressure / total
            toe_ratio = toe_pressure / total
            
            # Mapear a porcentaje del ciclo
            if heel_ratio > 0.6:  # Heel strike
                phase_percent = 5
            elif heel_ratio > 0.3 and toe_ratio < 0.2:  # Loading/Midstance
                phase_percent = 25
            elif toe_ratio > 0.4:  # Toe off
                phase_percent = 55
            else:  # Terminal stance
                phase_percent = 40
                
            self.gait_phase.update_phase(phase_percent)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = FootLabApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
'''
    
    with open("app_modern.py", "w", encoding="utf-8") as f:
        f.write(app_code)
    
    print("✓ Creado: app_modern.py")

def create_launch_script():
    """Crea script de lanzamiento rápido"""
    
    launch_code = '''@echo off
echo ========================================
echo   FootLab Premium - Modern UI
echo ========================================
echo.
python app_modern.py
pause
'''
    
    with open("launch_modern.bat", "w") as f:
        f.write(launch_code)
    
    print("✓ Creado: launch_modern.bat (Windows)")
    
    # Versión Linux/Mac
    launch_sh = '''#!/bin/bash
echo "========================================"
echo "  FootLab Premium - Modern UI"
echo "========================================"
echo ""
python3 app_modern.py
'''
    
    with open("launch_modern.sh", "w") as f:
        f.write(launch_sh)
    
    os.chmod("launch_modern.sh", 0o755)
    print("✓ Creado: launch_modern.sh (Linux/Mac)")

def main():
    print("=" * 50)
    print("INSTALANDO UI MODERNA FOOTLAB")
    print("=" * 50)
    
    # 1. Backup
    backup_old_ui()
    
    # 2. Crear app moderna
    create_app_modern()
    
    # 3. Scripts de lanzamiento
    create_launch_script()
    
    print("\n" + "=" * 50)
    print("✅ INSTALACIÓN COMPLETA")
    print("=" * 50)
    
    print("\nPara ejecutar la nueva UI moderna:")
    print("\nOpción 1 (directo):")
    print("  python app_modern.py")
    
    print("\nOpción 2 (Windows):")
    print("  Doble click en launch_modern.bat")
    
    print("\nOpción 3 (Linux/Mac):")
    print("  ./launch_modern.sh")
    
    print("\nCaracterísticas:")
    print("  • Diseño premium oscuro")
    print("  • Indicador de fases de marcha")
    print("  • Heatmaps de alta calidad")
    print("  • Métricas en tiempo real")
    print("  • 100% listo para sensores NURVV")

if __name__ == "__main__":
    main()