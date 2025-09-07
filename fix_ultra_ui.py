# ui/fixed_ultra_ui.py
"""
FootLab Ultra Premium UI - Versión Corregida
Optimizada para pantallas 1920x1080 y menores
"""

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QPushButton, QComboBox, QGridLayout, QGraphicsOpacityEffect,
    QGraphicsDropShadowEffect, QProgressBar, QApplication
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, Signal, QThread
)
from PySide6.QtGui import (
    QColor, QFont
)

import pyqtgraph as pg
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import time

# ============ PALETA DE COLORES ============
class UltraTheme:
    BACKGROUND = "#030712"
    SURFACE = "#0A1628"
    SURFACE_LIGHT = "#162032"
    SURFACE_BRIGHT = "#1E293B"
    
    GRADIENT_PRIMARY = ["#00D9FF", "#0099FF", "#0066FF"]
    GRADIENT_SUCCESS = ["#10F896", "#00E887", "#00D074"]
    GRADIENT_WARNING = ["#FFD700", "#FFC700", "#FFB700"]
    GRADIENT_DANGER = ["#FF6B6B", "#FF5252", "#FF3838"]
    
    CYAN = "#00D9FF"
    PURPLE = "#9F7AEA"
    GREEN = "#10F896"
    GOLD = "#FFD700"
    RED = "#FF5252"
    
    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#CBD5E1"
    TEXT_MUTED = "#64748B"
    TEXT_DIM = "#475569"


# ============ SIMULADOR MEJORADO ============
class ImprovedGaitSimulator(QThread):
    """Simulador optimizado con heatmaps visibles"""
    
    data_ready = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.time = 0
        
        # Grid más pequeño para mejor rendimiento
        self.grid_size = (64, 96)
        
        # Posiciones de sensores
        self.setup_sensor_positions()
        
    def setup_sensor_positions(self):
        """Define posiciones anatómicas de sensores"""
        # 16 sensores por pie en posiciones realistas
        self.sensors_left = np.array([
            # Talón (3 sensores)
            [0.45, 0.85], [0.50, 0.88], [0.55, 0.85],
            # Mediopié (4 sensores)
            [0.38, 0.65], [0.45, 0.62], [0.55, 0.62], [0.62, 0.65],
            # Metatarsos (5 sensores)
            [0.35, 0.35], [0.42, 0.32], [0.50, 0.30], [0.58, 0.32], [0.65, 0.35],
            # Dedos (4 sensores)
            [0.40, 0.15], [0.47, 0.12], [0.53, 0.12], [0.60, 0.15],
        ])
        
        # Pie derecho (espejo)
        self.sensors_right = self.sensors_left.copy()
        self.sensors_right[:, 0] = 1.0 - self.sensors_right[:, 0]
    
    def run(self):
        """Thread principal del simulador"""
        self.running = True
        self.time = 0
        
        while self.running:
            # Generar frame
            data = self.generate_frame()
            
            # Emitir datos
            self.data_ready.emit(data)
            
            # Control de velocidad (30 FPS)
            time.sleep(0.033)
            self.time += 0.033
    
    def generate_frame(self):
        """Genera datos realistas de presión"""
        # Fase del ciclo (0-100%)
        cycle_phase = ((self.time * 0.8) % 1.0) * 100  # Ciclo más lento
        
        # Generar presiones según fase
        left_pressures = self.calculate_pressures(cycle_phase, True)
        right_pressures = self.calculate_pressures((cycle_phase + 50) % 100, False)
        
        # Crear heatmaps visibles
        left_heatmap = self.create_heatmap(left_pressures, True)
        right_heatmap = self.create_heatmap(right_pressures, False)
        
        # Calcular CoP
        cop_left = self.calculate_cop(left_pressures, True)
        cop_right = self.calculate_cop(right_pressures, False)
        
        return {
            'left_pressures': left_pressures,
            'right_pressures': right_pressures,
            'left_heatmap': left_heatmap,
            'right_heatmap': right_heatmap,
            'cop_left': cop_left,
            'cop_right': cop_right,
            'gait_phase': cycle_phase
        }
    
    def calculate_pressures(self, phase, is_left):
        """Calcula presiones según fase de marcha"""
        pressures = np.zeros(16)
        
        # Patrones de activación según fase
        if phase < 15:  # Heel strike
            pressures[0:3] = [80, 100, 80]
        elif phase < 40:  # Mid stance
            pressures[0:3] = [60, 70, 60]
            pressures[3:7] = [50, 70, 70, 50]
        elif phase < 60:  # Propulsion
            pressures[7:12] = [70, 85, 100, 85, 70]
        elif phase < 80:  # Toe off
            pressures[12:16] = [90, 70, 70, 60]
        
        # Añadir ruido suave
        pressures += np.random.normal(0, 5, 16)
        pressures = np.maximum(0, pressures)
        
        return pressures
    
    def create_heatmap(self, pressures, is_left):
        """Crea heatmap visible de alta calidad"""
        sensors = self.sensors_left if is_left else self.sensors_right
        
        # Crear grid
        x = np.linspace(0, 1, self.grid_size[0])
        y = np.linspace(0, 1, self.grid_size[1])
        xx, yy = np.meshgrid(x, y)
        
        # Método alternativo: suma de gaussianas para cada sensor
        heatmap = np.zeros(self.grid_size)
        
        for i, (sx, sy) in enumerate(sensors):
            if pressures[i] > 0:
                # Crear gaussiana para cada sensor activo
                dist = np.sqrt((xx - sx)**2 + (yy - sy)**2)
                gaussian = np.exp(-dist**2 / 0.05) * pressures[i]
                heatmap += gaussian
        
        # Suavizar
        heatmap = gaussian_filter(heatmap, sigma=2)
        
        # Normalizar
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def calculate_cop(self, pressures, is_left):
        """Calcula centro de presión"""
        sensors = self.sensors_left if is_left else self.sensors_right
        
        total = np.sum(pressures)
        if total > 10:
            cop_x = np.sum(sensors[:, 0] * pressures) / total
            cop_y = np.sum(sensors[:, 1] * pressures) / total
            return (cop_x * self.grid_size[0], cop_y * self.grid_size[1])
        
        return (self.grid_size[0] / 2, self.grid_size[1] / 2)
    
    def stop(self):
        self.running = False
        self.wait()


# ============ WIDGET DE HEATMAP SIMPLE ============
class SimpleHeatmapWidget(pg.GraphicsLayoutWidget):
    """Heatmap simplificado que funciona"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setBackground(UltraTheme.BACKGROUND)
        
        # Plot
        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        
        if title:
            self.plot.setTitle(title, color=UltraTheme.TEXT_MUTED, size='10pt')
        
        # Imagen para heatmap
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        
        # Colormap médico
        colors = [
            (0, 0, 30),      # Negro azulado
            (0, 0, 100),     # Azul oscuro
            (0, 100, 200),   # Azul
            (0, 200, 200),   # Cyan
            (0, 255, 100),   # Verde
            (200, 255, 0),   # Amarillo
            (255, 100, 0),   # Naranja
            (255, 0, 0),     # Rojo
        ]
        positions = np.linspace(0, 1, len(colors))
        cmap = pg.ColorMap(positions, colors)
        self.img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        
        # Línea CoP
        self.cop_trail = self.plot.plot([], [], 
            pen=pg.mkPen(color=UltraTheme.CYAN, width=2))
        self.cop_point = self.plot.plot([], [],
            pen=None, symbol='o', symbolBrush=UltraTheme.CYAN, symbolSize=10)
        
        # Historial de CoP
        self.cop_history = []
        self.max_history = 20
    
    def update_display(self, heatmap, cop_x, cop_y):
        """Actualiza visualización"""
        if heatmap is not None and heatmap.size > 0:
            # Mostrar heatmap
            self.img.setImage(heatmap.T, autoLevels=False, levels=[0, 1])
        
        # Actualizar CoP
        if cop_x is not None and cop_y is not None:
            self.cop_history.append((cop_x, cop_y))
            if len(self.cop_history) > self.max_history:
                self.cop_history.pop(0)
            
            if len(self.cop_history) > 1:
                trail = np.array(self.cop_history)
                self.cop_trail.setData(trail[:, 0], trail[:, 1])
            
            self.cop_point.setData([cop_x], [cop_y])


# ============ INDICADOR DE FASE SIMPLE ============
class SimpleGaitPhaseWidget(QWidget):
    """Indicador de fase simplificado"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        
        self.phases = [
            {"name": "Heel Strike", "range": (0, 15), "color": UltraTheme.CYAN},
            {"name": "Mid Stance", "range": (15, 40), "color": UltraTheme.GREEN},
            {"name": "Propulsion", "range": (40, 60), "color": UltraTheme.GOLD},
            {"name": "Toe Off", "range": (60, 80), "color": UltraTheme.RED},
            {"name": "Swing", "range": (80, 100), "color": UltraTheme.PURPLE},
        ]
        
        self.current_phase = -1
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("GAIT PHASE")
        title.setStyleSheet(f"color: {UltraTheme.TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        layout.addWidget(title)
        
        # Contenedor de fases
        phases_container = QWidget()
        phases_layout = QHBoxLayout(phases_container)
        
        self.phase_widgets = []
        for phase in self.phases:
            widget = QFrame()
            widget.setFixedSize(100, 50)
            widget.setStyleSheet(f"""
                background: {UltraTheme.SURFACE_LIGHT};
                border-radius: 8px;
                border: 2px solid transparent;
            """)
            
            widget_layout = QVBoxLayout(widget)
            label = QLabel(phase["name"])
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(f"color: {UltraTheme.TEXT_SECONDARY}; font-size: 12px;")
            widget_layout.addWidget(label)
            
            widget.label = label
            widget.phase = phase
            self.phase_widgets.append(widget)
            phases_layout.addWidget(widget)
        
        layout.addWidget(phases_container)
        
        # Barra de progreso
        self.progress = QProgressBar()
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: {UltraTheme.SURFACE};
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {UltraTheme.CYAN};
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.progress)
    
    def update_phase(self, progress):
        """Actualiza fase actual"""
        self.progress.setValue(int(progress))
        
        # Determinar fase
        for i, phase in enumerate(self.phases):
            if phase["range"][0] <= progress <= phase["range"][1]:
                if i != self.current_phase:
                    # Desactivar anterior
                    if 0 <= self.current_phase < len(self.phase_widgets):
                        w = self.phase_widgets[self.current_phase]
                        w.setStyleSheet(f"""
                            background: {UltraTheme.SURFACE_LIGHT};
                            border-radius: 8px;
                            border: 2px solid transparent;
                        """)
                        w.label.setStyleSheet(f"color: {UltraTheme.TEXT_SECONDARY}; font-size: 12px;")
                    
                    # Activar nueva
                    w = self.phase_widgets[i]
                    color = phase["color"]
                    w.setStyleSheet(f"""
                        background: {UltraTheme.SURFACE_BRIGHT};
                        border-radius: 8px;
                        border: 2px solid {color};
                    """)
                    w.label.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold;")
                    
                    self.current_phase = i
                break


# ============ PANEL DE MÉTRICAS SIMPLE ============
class SimpleMetricsPanel(QFrame):
    """Panel de métricas simplificado"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            background: {UltraTheme.SURFACE};
            border-radius: 12px;
        """)
        
        self.metrics = {}
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        
        title = QLabel("METRICS")
        title.setStyleSheet(f"color: {UltraTheme.TEXT_MUTED}; font-size: 11px; font-weight: 600;")
        layout.addWidget(title)
        
        # Grid de métricas
        grid = QGridLayout()
        
        metrics_config = [
            ("Peak Pressure", "kPa", UltraTheme.RED),
            ("Contact Area", "cm²", UltraTheme.GREEN),
            ("CoP Velocity", "mm/s", UltraTheme.CYAN),
            ("Asymmetry", "%", UltraTheme.GOLD),
        ]
        
        for i, (name, unit, color) in enumerate(metrics_config):
            row = i // 2
            col = i % 2
            
            card = QFrame()
            card.setStyleSheet(f"background: {UltraTheme.BACKGROUND}; border-radius: 8px; padding: 10px;")
            
            card_layout = QVBoxLayout(card)
            
            name_label = QLabel(name)
            name_label.setStyleSheet(f"color: {UltraTheme.TEXT_MUTED}; font-size: 10px;")
            card_layout.addWidget(name_label)
            
            value_label = QLabel("0.0")
            value_label.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")
            card_layout.addWidget(value_label)
            
            unit_label = QLabel(unit)
            unit_label.setStyleSheet(f"color: {UltraTheme.TEXT_DIM}; font-size: 10px;")
            card_layout.addWidget(unit_label)
            
            card.value_label = value_label
            self.metrics[name] = card
            
            grid.addWidget(card, row, col)
        
        layout.addLayout(grid)
    
    def update_metric(self, name, value):
        if name in self.metrics:
            self.metrics[name].value_label.setText(f"{value:.1f}")


# ============ VENTANA PRINCIPAL OPTIMIZADA ============
class OptimizedFootLabUI(QMainWindow):
    """Ventana principal optimizada para cualquier pantalla"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("FootLab Ultra • Premium Baropodometry")
        self.setStyleSheet(f"background: {UltraTheme.BACKGROUND};")
        
        # Detectar tamaño de pantalla y ajustar
        self.setup_window_size()
        
        # Simulador
        self.simulator = ImprovedGaitSimulator()
        self.simulator.data_ready.connect(self.update_visualization)
        
        # Setup UI
        self.setup_ui()
        
        # Estado
        self.is_running = False
    
    def setup_window_size(self):
        """Configura tamaño de ventana según pantalla"""
        screen = QApplication.primaryScreen().geometry()
        
        # Usar 80% del tamaño de pantalla, máximo 1600x900
        width = min(int(screen.width() * 0.8), 1600)
        height = min(int(screen.height() * 0.8), 900)
        
        self.resize(width, height)
        
        # Centrar ventana
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.move(x, y)
    
    def setup_ui(self):
        """Construye la interfaz"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Panel izquierdo (controles)
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Panel central (visualización)
        center_panel = self.create_visualization_panel()
        main_layout.addWidget(center_panel, 2)
        
        # Panel derecho (métricas)
        right_panel = self.create_metrics_panel()
        main_layout.addWidget(right_panel, 1)
    
    def create_control_panel(self):
        """Panel de control"""
        panel = QFrame()
        panel.setMaximumWidth(300)
        panel.setStyleSheet(f"""
            background: {UltraTheme.SURFACE};
            border-radius: 12px;
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Logo
        logo = QLabel("FOOTLAB")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"""
            color: {UltraTheme.CYAN};
            font-size: 24px;
            font-weight: bold;
            letter-spacing: 3px;
        """)
        layout.addWidget(logo)
        
        # Separador
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {UltraTheme.SURFACE_LIGHT};")
        layout.addWidget(line)
        
        # Selector de fuente
        layout.addWidget(QLabel("DATA SOURCE"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Simulator", "NURVV BLE", "File Replay"])
        self.source_combo.setStyleSheet(f"""
            QComboBox {{
                background: {UltraTheme.BACKGROUND};
                color: {UltraTheme.TEXT_PRIMARY};
                border: 1px solid {UltraTheme.SURFACE_LIGHT};
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            }}
        """)
        layout.addWidget(self.source_combo)
        
        # Botones
        self.start_btn = QPushButton("▶  START")
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background: {UltraTheme.GREEN};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {UltraTheme.GRADIENT_SUCCESS[1]};
            }}
        """)
        self.start_btn.clicked.connect(self.start_session)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("■  STOP")
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background: {UltraTheme.RED};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {UltraTheme.GRADIENT_DANGER[1]};
            }}
        """)
        self.stop_btn.clicked.connect(self.stop_session)
        layout.addWidget(self.stop_btn)
        
        # Estado
        self.status_label = QLabel("● System Ready")
        self.status_label.setStyleSheet(f"color: {UltraTheme.TEXT_MUTED}; font-size: 12px; margin-top: 10px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Panel de visualización"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Indicador de fase
        self.gait_phase = SimpleGaitPhaseWidget()
        layout.addWidget(self.gait_phase)
        
        # Heatmaps
        heatmap_container = QFrame()
        heatmap_container.setStyleSheet(f"""
            background: {UltraTheme.SURFACE};
            border-radius: 12px;
        """)
        
        heatmap_layout = QHBoxLayout(heatmap_container)
        heatmap_layout.setContentsMargins(10, 10, 10, 10)
        
        self.heatmap_left = SimpleHeatmapWidget("LEFT FOOT")
        self.heatmap_right = SimpleHeatmapWidget("RIGHT FOOT")
        
        heatmap_layout.addWidget(self.heatmap_left)
        heatmap_layout.addWidget(self.heatmap_right)
        
        layout.addWidget(heatmap_container)
        
        return panel
    
    def create_metrics_panel(self):
        """Panel de métricas"""
        panel = QFrame()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        self.metrics = SimpleMetricsPanel()
        layout.addWidget(self.metrics)
        
        layout.addStretch()
        
        return panel
    
    def start_session(self):
        """Inicia sesión"""
        if not self.is_running:
            self.is_running = True
            self.simulator.start()
            self.status_label.setText("● Recording")
            self.status_label.setStyleSheet(f"color: {UltraTheme.GREEN}; font-size: 12px;")
    
    def stop_session(self):
        """Detiene sesión"""
        if self.is_running:
            self.is_running = False
            self.simulator.stop()
            self.status_label.setText("● System Ready")
            self.status_label.setStyleSheet(f"color: {UltraTheme.TEXT_MUTED}; font-size: 12px;")
    
    def update_visualization(self, data):
        """Actualiza visualización con datos"""
        # Actualizar heatmaps
        self.heatmap_left.update_display(
            data['left_heatmap'],
            data['cop_left'][0],
            data['cop_left'][1]
        )
        
        self.heatmap_right.update_display(
            data['right_heatmap'],
            data['cop_right'][0],
            data['cop_right'][1]
        )
        
        # Actualizar fase
        self.gait_phase.update_phase(data['gait_phase'])
        
        # Actualizar métricas
        peak = max(np.max(data['left_pressures']), np.max(data['right_pressures']))
        self.metrics.update_metric("Peak Pressure", peak)
        
        area = (np.sum(data['left_pressures'] > 10) + np.sum(data['right_pressures'] > 10)) * 2.5
        self.metrics.update_metric("Contact Area", area)
        
        velocity = 20 + np.random.normal(0, 3)
        self.metrics.update_metric("CoP Velocity", abs(velocity))
        
        total_left = np.sum(data['left_pressures'])
        total_right = np.sum(data['right_pressures'])
        if total_left + total_right > 0:
            asymmetry = abs(total_left - total_right) / (total_left + total_right) * 100
            self.metrics.update_metric("Asymmetry", asymmetry)


# ============ MAIN ============
def main():
    import sys
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = OptimizedFootLabUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()