# ui/modern_ui.py
"""
Interfaz de usuario moderna y elegante para FootLab
Diseño minimalista profesional con indicadores de fase de marcha
"""

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, 
    QLabel, QPushButton, QComboBox, QSlider, QGridLayout,
    QGraphicsDropShadowEffect, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Property, QRect
from PySide6.QtGui import QPalette, QColor, QFont, QLinearGradient, QPainter, QBrush, QPen

import pyqtgraph as pg

# Paleta de colores moderna
class ModernTheme:
    # Colores principales
    BACKGROUND = "#0A0E27"      # Azul oscuro profundo
    SURFACE = "#151B3D"         # Azul oscuro superficie
    SURFACE_LIGHT = "#1E2751"   # Superficie elevada
    
    # Acentos
    PRIMARY = "#00D4FF"         # Cyan brillante
    SECONDARY = "#7B61FF"       # Púrpura
    SUCCESS = "#00FF88"         # Verde neón
    WARNING = "#FFB800"         # Amarillo dorado
    DANGER = "#FF3366"          # Rosa/Rojo
    
    # Texto
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#B8BCC8"
    TEXT_MUTED = "#6C7293"
    
    # Gradientes para heatmap
    HEATMAP_GRADIENT = [
        (0.0, (0, 0, 20)),       # Negro azulado
        (0.2, (0, 0, 100)),      # Azul oscuro
        (0.4, (0, 150, 255)),    # Azul cyan
        (0.6, (0, 255, 150)),    # Verde cyan
        (0.8, (255, 200, 0)),    # Amarillo
        (1.0, (255, 50, 50)),    # Rojo
    ]

# Widget de fase de marcha con animación
class GaitPhaseIndicator(QWidget):
    """Indicador visual elegante de la fase de marcha"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        
        # Fases de la marcha
        self.phases = [
            {"name": "Initial Contact", "abbr": "IC", "range": (0, 2), "color": ModernTheme.PRIMARY},
            {"name": "Loading Response", "abbr": "LR", "range": (2, 12), "color": ModernTheme.SECONDARY},
            {"name": "Mid Stance", "abbr": "MSt", "range": (12, 31), "color": ModernTheme.SUCCESS},
            {"name": "Terminal Stance", "abbr": "TSt", "range": (31, 50), "color": ModernTheme.WARNING},
            {"name": "Pre-Swing", "abbr": "PSw", "range": (50, 62), "color": ModernTheme.DANGER},
            {"name": "Swing Phase", "abbr": "Sw", "range": (62, 100), "color": ModernTheme.PRIMARY},
        ]
        
        self.current_phase = 0
        self.progress = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Título
        title = QLabel("GAIT PHASE")
        title.setStyleSheet(f"""
            color: {ModernTheme.TEXT_MUTED};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 2px;
        """)
        layout.addWidget(title)
        
        # Contenedor de fases
        phases_layout = QHBoxLayout()
        phases_layout.setSpacing(8)
        
        self.phase_widgets = []
        for i, phase in enumerate(self.phases):
            phase_widget = self.create_phase_widget(phase)
            phases_layout.addWidget(phase_widget)
            self.phase_widgets.append(phase_widget)
        
        layout.addLayout(phases_layout)
        
        # Barra de progreso general
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {ModernTheme.SURFACE};
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {ModernTheme.PRIMARY},
                    stop:1 {ModernTheme.SECONDARY});
                border-radius: 2px;
            }}
        """)
        layout.addWidget(self.progress_bar)
        
    def create_phase_widget(self, phase):
        """Crea widget individual para cada fase"""
        widget = QFrame()
        widget.setFixedHeight(50)
        widget.setStyleSheet(f"""
            QFrame {{
                background: {ModernTheme.SURFACE};
                border-radius: 8px;
                border: 2px solid transparent;
            }}
        """)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        
        # Abreviatura
        abbr_label = QLabel(phase["abbr"])
        abbr_label.setAlignment(Qt.AlignCenter)
        abbr_label.setStyleSheet(f"""
            color: {ModernTheme.TEXT_SECONDARY};
            font-size: 16px;
            font-weight: bold;
        """)
        layout.addWidget(abbr_label)
        
        # Porcentaje
        percent_label = QLabel(f"{phase['range'][0]}-{phase['range'][1]}%")
        percent_label.setAlignment(Qt.AlignCenter)
        percent_label.setStyleSheet(f"""
            color: {ModernTheme.TEXT_MUTED};
            font-size: 10px;
        """)
        layout.addWidget(percent_label)
        
        # Guardar referencias
        widget.abbr_label = abbr_label
        widget.phase_data = phase
        
        return widget
    
    def update_phase(self, progress_percent):
        """Actualiza el indicador de fase basado en el progreso del ciclo"""
        self.progress = progress_percent
        self.progress_bar.setValue(int(progress_percent))
        
        # Determinar fase actual
        for i, phase in enumerate(self.phases):
            if phase["range"][0] <= progress_percent <= phase["range"][1]:
                if i != self.current_phase:
                    self.highlight_phase(i)
                break
    
    def highlight_phase(self, phase_index):
        """Resalta la fase activa con animación"""
        # Desactivar fase anterior
        if self.current_phase < len(self.phase_widgets):
            old_widget = self.phase_widgets[self.current_phase]
            old_widget.setStyleSheet(f"""
                QFrame {{
                    background: {ModernTheme.SURFACE};
                    border-radius: 8px;
                    border: 2px solid transparent;
                }}
            """)
            old_widget.abbr_label.setStyleSheet(f"""
                color: {ModernTheme.TEXT_SECONDARY};
                font-size: 16px;
                font-weight: bold;
            """)
        
        # Activar nueva fase
        if phase_index < len(self.phase_widgets):
            new_widget = self.phase_widgets[phase_index]
            phase_color = self.phases[phase_index]["color"]
            new_widget.setStyleSheet(f"""
                QFrame {{
                    background: {ModernTheme.SURFACE_LIGHT};
                    border-radius: 8px;
                    border: 2px solid {phase_color};
                }}
            """)
            new_widget.abbr_label.setStyleSheet(f"""
                color: {phase_color};
                font-size: 18px;
                font-weight: bold;
            """)
            
        self.current_phase = phase_index

# Widget de heatmap mejorado
class PremiumHeatmapWidget(pg.GraphicsLayoutWidget):
    """Widget de heatmap con diseño premium"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setBackground(ModernTheme.BACKGROUND)
        
        self.title = title
        self.setup_plot()
        
    def setup_plot(self):
        # Configurar plot
        self.plot = self.addPlot()
        self.plot.setLabel('left', '', units='', 
                          **{'color': ModernTheme.TEXT_SECONDARY, 'font-size': '10pt'})
        self.plot.setLabel('bottom', '',
                          **{'color': ModernTheme.TEXT_SECONDARY, 'font-size': '10pt'})
        
        # Ocultar ejes para look más limpio
        self.plot.getAxis('left').setStyle(showValues=False)
        self.plot.getAxis('bottom').setStyle(showValues=False)
        self.plot.getAxis('left').setPen(pg.mkPen(color=ModernTheme.TEXT_MUTED, width=0.5))
        self.plot.getAxis('bottom').setPen(pg.mkPen(color=ModernTheme.TEXT_MUTED, width=0.5))
        
        # Grid sutil
        self.plot.showGrid(x=True, y=True, alpha=0.05)
        
        # Imagen para heatmap
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        
        # Configurar colormap
        colors = []
        positions = []
        for pos, color in ModernTheme.HEATMAP_GRADIENT:
            positions.append(pos)
            colors.append(color)
        
        cmap = pg.ColorMap(positions, colors)
        self.img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        
        # Línea de CoP
        self.cop_line = self.plot.plot([], [], 
                                       pen=pg.mkPen(color=ModernTheme.PRIMARY, width=2))
        self.cop_point = self.plot.plot([], [], 
                                        pen=None,
                                        symbol='o', 
                                        symbolBrush=ModernTheme.PRIMARY,
                                        symbolSize=8)
        
        # Título
        if self.title:
            title_item = pg.TextItem(self.title, 
                                     color=ModernTheme.TEXT_PRIMARY,
                                     anchor=(0.5, 1))
            title_item.setPos(50, 100)
            self.plot.addItem(title_item)
    
    def update_heatmap(self, data, cop_x=None, cop_y=None, cop_trail=None):
        """Actualiza el heatmap con los datos"""
        # Normalizar datos
        if data.max() > 0:
            normalized = data / data.max()
        else:
            normalized = data
            
        # Actualizar imagen
        self.img.setImage(normalized.T)
        
        # Actualizar CoP
        if cop_x is not None and cop_y is not None:
            self.cop_point.setData([cop_x], [cop_y])
            
        if cop_trail is not None and len(cop_trail) > 1:
            trail = np.array(cop_trail)
            self.cop_line.setData(trail[:, 0], trail[:, 1])

# Panel de métricas elegante
class MetricsPanel(QFrame):
    """Panel de métricas con diseño moderno"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {ModernTheme.SURFACE};
                border-radius: 12px;
            }}
        """)
        
        self.metrics = {}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Título
        title = QLabel("REAL-TIME METRICS")
        title.setStyleSheet(f"""
            color: {ModernTheme.TEXT_MUTED};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 2px;
        """)
        layout.addWidget(title, 0, 0, 1, 2)
        
        # Métricas
        metrics_config = [
            ("Peak Pressure", "kPa", ModernTheme.DANGER),
            ("Contact Area", "cm²", ModernTheme.SUCCESS),
            ("CoP Velocity", "mm/s", ModernTheme.PRIMARY),
            ("Asymmetry", "%", ModernTheme.WARNING),
        ]
        
        for i, (name, unit, color) in enumerate(metrics_config):
            row = i // 2 + 1
            col = i % 2
            
            metric_widget = self.create_metric_widget(name, unit, color)
            layout.addWidget(metric_widget, row, col)
            self.metrics[name] = metric_widget
    
    def create_metric_widget(self, name, unit, color):
        """Crea widget individual para métrica"""
        widget = QFrame()
        widget.setStyleSheet(f"""
            QFrame {{
                background: {ModernTheme.BACKGROUND};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        
        # Nombre
        name_label = QLabel(name)
        name_label.setStyleSheet(f"""
            color: {ModernTheme.TEXT_MUTED};
            font-size: 11px;
        """)
        layout.addWidget(name_label)
        
        # Valor
        value_label = QLabel("0.0")
        value_label.setStyleSheet(f"""
            color: {color};
            font-size: 24px;
            font-weight: bold;
        """)
        layout.addWidget(value_label)
        
        # Unidad
        unit_label = QLabel(unit)
        unit_label.setStyleSheet(f"""
            color: {ModernTheme.TEXT_SECONDARY};
            font-size: 10px;
        """)
        layout.addWidget(unit_label)
        
        widget.value_label = value_label
        return widget
    
    def update_metric(self, name, value):
        """Actualiza valor de métrica"""
        if name in self.metrics:
            self.metrics[name].value_label.setText(f"{value:.1f}")

# Ventana principal moderna
class ModernFootLabUI(QMainWindow):
    """Ventana principal con diseño premium"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FootLab Premium - Baropodometry System")
        self.setStyleSheet(f"background: {ModernTheme.BACKGROUND};")
        self.resize(1600, 900)
        
        # Estado
        self.gait_cycle_progress = 0
        self.cop_trail_left = []
        self.cop_trail_right = []
        
        self.setup_ui()
        
        # Timer para simulación
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(50)  # 20 FPS
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Panel izquierdo (controles)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Panel central (heatmaps)
        center_panel = self.create_center_panel()
        main_layout.addWidget(center_panel, 3)
        
        # Panel derecho (métricas)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
    
    def create_left_panel(self):
        """Panel de control izquierdo"""
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background: {ModernTheme.SURFACE};
                border-radius: 12px;
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # Logo/Título
        title = QLabel("FOOTLAB")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {ModernTheme.PRIMARY};
            font-size: 28px;
            font-weight: bold;
            letter-spacing: 4px;
        """)
        layout.addWidget(title)
        
        # Línea divisora
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background: {ModernTheme.SURFACE_LIGHT}; max-height: 2px;")
        layout.addWidget(line)
        
        # Selector de fuente
        source_label = QLabel("DATA SOURCE")
        source_label.setStyleSheet(f"""
            color: {ModernTheme.TEXT_MUTED};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 2px;
        """)
        layout.addWidget(source_label)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Simulator", "NURVV BLE", "File Replay"])
        self.source_combo.setStyleSheet(f"""
            QComboBox {{
                background: {ModernTheme.BACKGROUND};
                color: {ModernTheme.TEXT_PRIMARY};
                border: 2px solid {ModernTheme.SURFACE_LIGHT};
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }}
            QComboBox:hover {{
                border-color: {ModernTheme.PRIMARY};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {ModernTheme.TEXT_SECONDARY};
                margin-right: 10px;
            }}
        """)
        layout.addWidget(self.source_combo)
        
        # Botones de control
        self.start_btn = self.create_button("START", ModernTheme.SUCCESS)
        self.stop_btn = self.create_button("STOP", ModernTheme.DANGER)
        self.calibrate_btn = self.create_button("CALIBRATE", ModernTheme.WARNING)
        
        self.start_btn.clicked.connect(self.start_acquisition)
        self.stop_btn.clicked.connect(self.stop_acquisition)
        
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.calibrate_btn)
        
        # Estado de conexión
        self.status_widget = QFrame()
        self.status_widget.setStyleSheet(f"""
            QFrame {{
                background: {ModernTheme.BACKGROUND};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        status_layout = QHBoxLayout(self.status_widget)
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet(f"color: {ModernTheme.TEXT_MUTED}; font-size: 20px;")
        status_layout.addWidget(self.status_indicator)
        
        self.status_text = QLabel("Disconnected")
        self.status_text.setStyleSheet(f"color: {ModernTheme.TEXT_SECONDARY};")
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(self.status_widget)
        
        layout.addStretch()
        
        return panel
    
    def create_center_panel(self):
        """Panel central con heatmaps"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Indicador de fase de marcha
        self.gait_phase = GaitPhaseIndicator()
        layout.addWidget(self.gait_phase)
        
        # Heatmaps
        heatmap_container = QFrame()
        heatmap_container.setStyleSheet(f"""
            QFrame {{
                background: {ModernTheme.SURFACE};
                border-radius: 12px;
            }}
        """)
        
        heatmap_layout = QHBoxLayout(heatmap_container)
        heatmap_layout.setContentsMargins(20, 20, 20, 20)
        heatmap_layout.setSpacing(20)
        
        # Heatmap izquierdo
        self.heatmap_left = PremiumHeatmapWidget("LEFT FOOT")
        heatmap_layout.addWidget(self.heatmap_left)
        
        # Heatmap derecho  
        self.heatmap_right = PremiumHeatmapWidget("RIGHT FOOT")
        heatmap_layout.addWidget(self.heatmap_right)
        
        layout.addWidget(heatmap_container)
        
        return panel
    
    def create_right_panel(self):
        """Panel derecho con métricas"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Panel de métricas
        self.metrics_panel = MetricsPanel()
        layout.addWidget(self.metrics_panel)
        
        # Gráfico temporal
        graph_panel = QFrame()
        graph_panel.setStyleSheet(f"""
            QFrame {{
                background: {ModernTheme.SURFACE};
                border-radius: 12px;
            }}
        """)
        graph_layout = QVBoxLayout(graph_panel)
        graph_layout.setContentsMargins(20, 20, 20, 20)
        
        graph_title = QLabel("FORCE PROFILE")
        graph_title.setStyleSheet(f"""
            color: {ModernTheme.TEXT_MUTED};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 2px;
        """)
        graph_layout.addWidget(graph_title)
        
        # Aquí iría el gráfico temporal
        self.force_graph = pg.PlotWidget()
        self.force_graph.setBackground(ModernTheme.BACKGROUND)
        self.force_graph.showGrid(x=True, y=True, alpha=0.1)
        graph_layout.addWidget(self.force_graph)
        
        layout.addWidget(graph_panel)
        
        return panel
    
    def create_button(self, text, color):
        """Crea botón con estilo moderno"""
        btn = QPushButton(text)
        btn.setFixedHeight(45)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{
                background: {color}CC;
            }}
            QPushButton:pressed {{
                background: {color}99;
            }}
        """)
        
        # Sombra
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(color).darker())
        shadow.setOffset(0, 5)
        btn.setGraphicsEffect(shadow)
        
        return btn
    
    def start_acquisition(self):
        """Inicia adquisición de datos"""
        self.timer.start()
        self.status_indicator.setStyleSheet(f"color: {ModernTheme.SUCCESS}; font-size: 20px;")
        self.status_text.setText("Connected")
    
    def stop_acquisition(self):
        """Detiene adquisición"""
        self.timer.stop()
        self.status_indicator.setStyleSheet(f"color: {ModernTheme.TEXT_MUTED}; font-size: 20px;")
        self.status_text.setText("Disconnected")
    
    def update_simulation(self):
        """Actualiza simulación (temporal, será reemplazado por datos reales)"""
        # Simular progreso del ciclo de marcha
        self.gait_cycle_progress = (self.gait_cycle_progress + 2) % 100
        self.gait_phase.update_phase(self.gait_cycle_progress)
        
        # Simular datos de presión (ESTO SERÁ REEMPLAZADO POR DATOS REALES)
        # Por ahora solo para demostrar el diseño
        t = self.gait_cycle_progress / 100.0 * 2 * np.pi
        
        # Generar heatmap simple
        x = np.linspace(0, 100, 32)
        y = np.linspace(0, 100, 48)
        X, Y = np.meshgrid(x, y)
        
        # Simular presiones en diferentes zonas según fase
        if self.gait_cycle_progress < 30:  # Heel strike
            center_x, center_y = 50, 80
        elif self.gait_cycle_progress < 60:  # Midstance
            center_x, center_y = 50, 50
        else:  # Toe off
            center_x, center_y = 50, 20
            
        Z = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / 200) * np.sin(t)
        
        # Actualizar heatmaps
        self.heatmap_left.update_heatmap(Z.T, center_x, center_y)
        self.heatmap_right.update_heatmap(Z.T, center_x, center_y)
        
        # Actualizar métricas (valores de ejemplo)
        self.metrics_panel.update_metric("Peak Pressure", 150 + np.sin(t) * 30)
        self.metrics_panel.update_metric("Contact Area", 45 + np.cos(t) * 5)
        self.metrics_panel.update_metric("CoP Velocity", 25 + np.sin(t * 2) * 10)
        self.metrics_panel.update_metric("Asymmetry", 5 + np.abs(np.sin(t * 3)) * 3)

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = ModernFootLabUI()
    window.show()
    
    sys.exit(app.exec())
