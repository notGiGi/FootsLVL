# ui/ultra_premium_ui.py
"""
FootLab Ultra Premium UI - Diseño de clase mundial
60 FPS | Heatmaps fluidos | Animaciones cinematográficas
"""

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QPushButton, QComboBox, QGridLayout, QGraphicsOpacityEffect,
    QGraphicsDropShadowEffect, QProgressBar, QSlider, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
    QSequentialAnimationGroup, Property, QRect, QPoint, QSize, Signal, QThread
)
from PySide6.QtGui import (QColor, 
    QPalette, QColor, QFont, QLinearGradient, QPainter, QBrush, QPen,
    QRadialGradient, QConicalGradient, QPixmap, QPainterPath
)

import pyqtgraph as pg
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import time

# ============ PALETA DE COLORES ULTRA PREMIUM ============
class UltraTheme:
    # Fondos
    BACKGROUND = "#030712"        # Negro azulado profundo
    SURFACE = "#0A1628"          # Azul medianoche
    SURFACE_LIGHT = "#162032"    # Azul superficie
    SURFACE_BRIGHT = "#1E293B"   # Azul claro
    
    # Gradientes principales
    GRADIENT_PRIMARY = ["#00D9FF", "#0099FF", "#0066FF"]
    GRADIENT_SECONDARY = ["#B794F6", "#9F7AEA", "#805AD5"]
    GRADIENT_SUCCESS = ["#10F896", "#00E887", "#00D074"]
    GRADIENT_WARNING = ["#FFD700", "#FFC700", "#FFB700"]
    GRADIENT_DANGER = ["#FF6B6B", "#FF5252", "#FF3838"]
    
    # Colores sólidos
    CYAN = "#00D9FF"
    PURPLE = "#9F7AEA"
    GREEN = "#10F896"
    GOLD = "#FFD700"
    RED = "#FF5252"
    
    # Texto
    TEXT_PRIMARY = "#F8FAFC"
    TEXT_SECONDARY = "#CBD5E1"
    TEXT_MUTED = "#64748B"
    TEXT_DIM = "#475569"
    
    # Efectos
    GLOW_CYAN = QColor(0, 217, 255, 102)  # 40% alpha = 102/255
    GLOW_PURPLE = QColor(159, 122, 234, 102)
    SHADOW = QColor(0, 0, 0, 204)


# ============ SIMULADOR REALISTA DE MARCHA ============
class RealisticGaitSimulator(QThread):
    """Simulador ultra-realista de marcha con 60 FPS garantizados"""
    
    data_ready = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.fps = 60  # 60 FPS objetivo
        self.frame_time = 1.0 / self.fps
        
        # Parámetros de marcha realistas
        self.gait_cycle_duration = 1.2  # segundos por ciclo
        self.stance_ratio = 0.62  # 62% stance, 38% swing
        
        # Posiciones anatómicas de sensores (16 por pie)
        self.sensor_positions_left = np.array([
            # Talón (3)
            [0.45, 0.88], [0.50, 0.92], [0.55, 0.88],
            # Mediopié (4)
            [0.38, 0.68], [0.45, 0.65], [0.55, 0.65], [0.62, 0.68],
            # Metatarsos (5)
            [0.35, 0.35], [0.42, 0.32], [0.50, 0.30], [0.58, 0.32], [0.65, 0.35],
            # Dedos (4)
            [0.38, 0.15], [0.46, 0.12], [0.54, 0.12], [0.62, 0.15],
        ])
        
        self.sensor_positions_right = self.sensor_positions_left.copy()
        self.sensor_positions_right[:, 0] = 1.0 - self.sensor_positions_right[:, 0]
        
        # Grid de alta resolución para interpolación
        self.grid_size = (80, 120)  # Mayor resolución para suavidad
        
        # Pre-calcular grids para optimización
        self._setup_interpolation_grids()
        
        # Estado del simulador
        self.time = 0
        self.last_frame_time = 0
        
    def _setup_interpolation_grids(self):
        """Pre-calcula grids de interpolación para máximo rendimiento"""
        self.grid_x = np.linspace(0, 1, self.grid_size[1])
        self.grid_y = np.linspace(0, 1, self.grid_size[0])
        self.grid_xx, self.grid_yy = np.meshgrid(self.grid_x, self.grid_y)
        
    def run(self):
        """Thread principal del simulador a 60 FPS"""
        self.running = True
        self.time = 0
        frame_count = 0
        start_time = time.perf_counter()
        
        while self.running:
            frame_start = time.perf_counter()
            
            # Generar datos de presión
            data = self.generate_frame()
            
            # Emitir datos
            self.data_ready.emit(data)
            
            # Control preciso de FPS
            frame_elapsed = time.perf_counter() - frame_start
            sleep_time = self.frame_time - frame_elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Actualizar tiempo
            self.time += self.frame_time
            frame_count += 1
            
            # Debug FPS cada segundo
            if frame_count % 60 == 0:
                elapsed = time.perf_counter() - start_time
                actual_fps = frame_count / elapsed
                print(f"FPS actual: {actual_fps:.1f}")
    
    def generate_frame(self):
        """Genera un frame realista de datos de presión"""
        # Calcular fase del ciclo de marcha
        cycle_phase = (self.time % self.gait_cycle_duration) / self.gait_cycle_duration
        
        # Generar presiones para cada pie
        left_pressures = self._generate_foot_pressures(cycle_phase, is_left=True)
        right_pressures = self._generate_foot_pressures(
            (cycle_phase + 0.5) % 1.0,  # Pies desfasados 180°
            is_left=False
        )
        
        # Interpolar a grid de alta resolución
        left_heatmap = self._interpolate_to_heatmap(left_pressures, is_left=True)
        right_heatmap = self._interpolate_to_heatmap(right_pressures, is_left=False)
        
        # Calcular CoP
        cop_left = self._calculate_cop(left_pressures, is_left=True)
        cop_right = self._calculate_cop(right_pressures, is_left=False)
        
        return {
            'left_pressures': left_pressures,
            'right_pressures': right_pressures,
            'left_heatmap': left_heatmap,
            'right_heatmap': right_heatmap,
            'cop_left': cop_left,
            'cop_right': cop_right,
            'gait_phase': cycle_phase * 100,
            'timestamp': self.time
        }
    
    def _generate_foot_pressures(self, phase, is_left=True):
        """Genera presiones realistas según la fase de marcha"""
        pressures = np.zeros(16)
        
        if phase < self.stance_ratio:  # Fase de apoyo
            stance_phase = phase / self.stance_ratio
            
            if stance_phase < 0.15:  # Initial contact (heel strike)
                # Activación fuerte del talón
                pressures[0:3] = self._smooth_activation(stance_phase / 0.15) * np.array([0.8, 1.0, 0.8]) * 100
                
            elif stance_phase < 0.4:  # Loading response
                # Transición talón → mediopié
                t = (stance_phase - 0.15) / 0.25
                heel_factor = 1.0 - t * 0.5
                mid_factor = t
                
                pressures[0:3] = heel_factor * np.array([0.7, 0.9, 0.7]) * 80
                pressures[3:7] = mid_factor * np.array([0.5, 0.7, 0.7, 0.5]) * 60
                
            elif stance_phase < 0.7:  # Mid stance
                # Peso distribuido, inicio de propulsión
                t = (stance_phase - 0.4) / 0.3
                mid_factor = 1.0 - t * 0.3
                fore_factor = t
                
                pressures[3:7] = mid_factor * np.array([0.6, 0.8, 0.8, 0.6]) * 70
                pressures[7:12] = fore_factor * np.array([0.7, 0.9, 1.0, 0.9, 0.7]) * 90
                
            elif stance_phase < 0.9:  # Terminal stance
                # Propulsión desde antepié
                t = (stance_phase - 0.7) / 0.2
                fore_factor = 1.0
                toe_factor = t
                
                pressures[7:12] = fore_factor * np.array([0.8, 0.95, 1.0, 0.95, 0.8]) * 100
                pressures[12:16] = toe_factor * np.array([0.9, 0.7, 0.7, 0.6]) * 70
                
            else:  # Pre-swing (toe-off)
                # Solo dedos en contacto
                t = (stance_phase - 0.9) / 0.1
                toe_factor = 1.0 - t
                
                pressures[12:16] = toe_factor * np.array([1.0, 0.8, 0.7, 0.5]) * 50
        
        # Añadir ruido realista
        noise = np.random.normal(0, 2, 16)
        pressures = np.maximum(0, pressures + noise)
        
        # Suavizado temporal (simulación de inercia del tejido)
        if not hasattr(self, 'pressure_history'):
            self.pressure_history = {}
        
        key = 'left' if is_left else 'right'
        if key not in self.pressure_history:
            self.pressure_history[key] = pressures
        else:
            # Filtro de media móvil exponencial
            alpha = 0.3
            self.pressure_history[key] = alpha * pressures + (1 - alpha) * self.pressure_history[key]
            pressures = self.pressure_history[key]
        
        return pressures
    
    def _smooth_activation(self, t):
        """Función de activación suave (sigmoide)"""
        return 1 / (1 + np.exp(-10 * (t - 0.5)))
    
    def _interpolate_to_heatmap(self, pressures, is_left=True):
        """Interpola sensores a heatmap de alta resolución con máximo rendimiento"""
        positions = self.sensor_positions_left if is_left else self.sensor_positions_right
        
        # Interpolación cúbica rápida
        try:
            heatmap = griddata(
                positions,
                pressures,
                (self.grid_xx, self.grid_yy),
                method='cubic',
                fill_value=0
            )
            
            # Suavizado gaussiano para eliminar artefactos
            heatmap = gaussian_filter(heatmap, sigma=2.0)
            
            # Normalizar para visualización
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
        except:
            heatmap = np.zeros(self.grid_size)
        
        return heatmap
    
    def _calculate_cop(self, pressures, is_left=True):
        """Calcula centro de presión con suavizado"""
        positions = self.sensor_positions_left if is_left else self.sensor_positions_right
        
        total = np.sum(pressures)
        if total > 5:  # Umbral mínimo
            cop_x = np.sum(positions[:, 0] * pressures) / total
            cop_y = np.sum(positions[:, 1] * pressures) / total
            return (cop_x * self.grid_size[1], cop_y * self.grid_size[0])
        return (self.grid_size[1] / 2, self.grid_size[0] / 2)
    
    def stop(self):
        self.running = False
        self.wait()


# ============ WIDGET DE HEATMAP ULTRA PREMIUM ============
class UltraPremiumHeatmap(pg.GraphicsLayoutWidget):
    """Heatmap con renderizado ultra suave y efectos visuales"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setBackground(UltraTheme.BACKGROUND)
        self.title = title
        
        # Configuración de renderizado de alta calidad
        self.setAntialiasing(True)
        self.setRenderHint(QPainter.Antialiasing)
        
        self.setup_plot()
        
        # Historial de CoP para trail suave
        self.cop_trail = []
        self.max_trail_length = 30
        
    def setup_plot(self):
        """Configura el plot con estilo ultra premium"""
        # Plot principal
        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)
        
        # Eliminar ejes para look más limpio
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        
        # Título elegante
        if self.title:
            title_html = f'<span style="color: {UltraTheme.TEXT_MUTED}; font-size: 11pt; font-weight: 300;">{self.title}</span>'
            self.plot.setTitle(title_html)
        
        # Imagen del heatmap
        self.img = pg.ImageItem(border='w')
        self.plot.addItem(self.img)
        
        # Configurar colormap premium
        self.setup_colormap()
        
        # Trail del CoP con gradiente
        self.cop_trails = []
        for i in range(10):  # Múltiples líneas para efecto de gradiente
            pen = pg.mkPen(
                color=UltraTheme.CYAN,
                width=3 - i * 0.2,
                style=Qt.SolidLine
            )
            trail = self.plot.plot([], [], pen=pen)
            self.cop_trails.append(trail)
        
        # Punto actual del CoP con glow
        self.cop_point = self.plot.plot(
            [], [],
            pen=None,
            symbol='o',
            symbolBrush=pg.mkBrush(UltraTheme.CYAN),
            symbolPen=pg.mkPen(color=QColor(0, 217, 255, 102), width=2),
            symbolSize=12
        )
        
        # Efecto de glow adicional
        self.cop_glow = self.plot.plot(
            [], [],
            pen=None,
            symbol='o',
            symbolBrush=pg.mkBrush(QColor(0, 217, 255, 51)),
            symbolPen=None,
            symbolSize=20
        )
    
    def setup_colormap(self):
        """Configura un colormap ultra suave y elegante"""
        # Colormap tipo "thermal" mejorado
        colors = [
            (0, 5, 15),      # Negro azulado
            (0, 20, 50),     # Azul muy oscuro
            (0, 50, 100),    # Azul oscuro
            (0, 100, 150),   # Azul medio
            (0, 150, 200),   # Azul claro
            (0, 200, 150),   # Cyan-verde
            (50, 255, 100),  # Verde brillante
            (200, 255, 0),   # Amarillo-verde
            (255, 200, 0),   # Amarillo
            (255, 100, 0),   # Naranja
            (255, 50, 50),   # Rojo
            (255, 255, 255), # Blanco (máxima presión)
        ]
        
        positions = np.linspace(0, 1, len(colors))
        
        cmap = pg.ColorMap(positions, colors)
        lut = cmap.getLookupTable(0.0, 1.0, 512)  # Más niveles para suavidad
        self.img.setLookupTable(lut)
    
    def update_display(self, heatmap, cop_x=None, cop_y=None):
        """Actualiza el display con animación suave"""
        if heatmap is not None:
            # Aplicar un poco más de suavizado para efecto visual
            smooth_heatmap = gaussian_filter(heatmap, sigma=1.0)
            
            # Aplicar curva de gamma para mejor contraste
            gamma = 0.8
            smooth_heatmap = np.power(smooth_heatmap, gamma)
            
            self.img.setImage(smooth_heatmap.T, autoLevels=False)
        
        # Actualizar CoP con trail
        if cop_x is not None and cop_y is not None:
            # Añadir al trail
            self.cop_trail.append((cop_x, cop_y))
            if len(self.cop_trail) > self.max_trail_length:
                self.cop_trail.pop(0)
            
            # Actualizar líneas del trail con efecto de fade
            if len(self.cop_trail) > 1:
                trail_array = np.array(self.cop_trail)
                
                # Dividir el trail en segmentos para efecto de gradiente
                for i, line in enumerate(self.cop_trails):
                    start_idx = max(0, len(trail_array) - (i + 1) * 3)
                    if start_idx < len(trail_array) - 1:
                        segment = trail_array[start_idx:]
                        line.setData(segment[:, 0], segment[:, 1])
                    else:
                        line.setData([], [])
            
            # Actualizar punto actual con glow
            self.cop_point.setData([cop_x], [cop_y])
            self.cop_glow.setData([cop_x], [cop_y])


# ============ INDICADOR DE FASE ANIMADO ============
class AnimatedGaitPhaseWidget(QWidget):
    """Indicador de fase de marcha con animaciones cinematográficas"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(140)
        
        self.phases = [
            {"name": "Initial Contact", "abbr": "IC", "range": (0, 2), "color": UltraTheme.CYAN},
            {"name": "Loading Response", "abbr": "LR", "range": (2, 12), "color": UltraTheme.PURPLE},
            {"name": "Mid Stance", "abbr": "MSt", "range": (12, 31), "color": UltraTheme.GREEN},
            {"name": "Terminal Stance", "abbr": "TSt", "range": (31, 50), "color": UltraTheme.GOLD},
            {"name": "Pre-Swing", "abbr": "PSw", "range": (50, 62), "color": UltraTheme.RED},
            {"name": "Swing Phase", "abbr": "Sw", "range": (62, 100), "color": UltraTheme.GRADIENT_PRIMARY[0]},
        ]
        
        self.current_phase = -1
        self.progress = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 15, 30, 15)
        
        # Título con estilo
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 10)
        
        title = QLabel("GAIT PHASE ANALYSIS")
        title.setStyleSheet(f"""
            color: {UltraTheme.TEXT_MUTED};
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 3px;
        """)
        title_layout.addWidget(title)
        
        # Indicador de porcentaje
        self.percent_label = QLabel("0%")
        self.percent_label.setStyleSheet(f"""
            color: {UltraTheme.CYAN};
            font-size: 14px;
            font-weight: bold;
        """)
        title_layout.addStretch()
        title_layout.addWidget(self.percent_label)
        
        layout.addWidget(title_container)
        
        # Contenedor de fases con layout personalizado
        phases_container = QWidget()
        phases_container.setStyleSheet(f"""
            background: {UltraTheme.SURFACE};
            border-radius: 12px;
        """)
        phases_layout = QHBoxLayout(phases_container)
        phases_layout.setContentsMargins(15, 10, 15, 10)
        phases_layout.setSpacing(8)
        
        self.phase_widgets = []
        for i, phase in enumerate(self.phases):
            widget = self.create_phase_indicator(phase)
            phases_layout.addWidget(widget)
            self.phase_widgets.append(widget)
        
        layout.addWidget(phases_container)
        
        # Barra de progreso elegante
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {UltraTheme.SURFACE_LIGHT};
                border-radius: 3px;
                border: 1px solid {UltraTheme.SURFACE_BRIGHT};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {UltraTheme.GRADIENT_PRIMARY[0]},
                    stop:0.5 {UltraTheme.GRADIENT_PRIMARY[1]},
                    stop:1 {UltraTheme.GRADIENT_PRIMARY[2]});
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.progress_bar)
    
    def create_phase_indicator(self, phase):
        """Crea un indicador individual con efectos visuales"""
        container = QFrame()
        container.setFixedSize(80, 60)
        container.setStyleSheet(f"""
            QFrame {{
                background: {UltraTheme.SURFACE_LIGHT};
                border: 2px solid transparent;
                border-radius: 10px;
            }}
        """)
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 8, 5, 8)
        layout.setSpacing(2)
        
        # Abreviación
        abbr = QLabel(phase["abbr"])
        abbr.setAlignment(Qt.AlignCenter)
        abbr.setStyleSheet(f"""
            color: {UltraTheme.TEXT_SECONDARY};
            font-size: 18px;
            font-weight: bold;
        """)
        layout.addWidget(abbr)
        
        # Rango
        range_label = QLabel(f"{phase['range'][0]}-{phase['range'][1]}%")
        range_label.setAlignment(Qt.AlignCenter)
        range_label.setStyleSheet(f"""
            color: {UltraTheme.TEXT_DIM};
            font-size: 9px;
        """)
        layout.addWidget(range_label)
        
        # Guardar referencias
        container.phase_data = phase
        container.abbr_label = abbr
        container.is_active = False
        
        # Añadir efecto de sombra
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 2)
        container.setGraphicsEffect(shadow)
        
        return container
    
    def update_phase(self, progress):
        """Actualiza con animación suave"""
        self.progress = progress
        self.progress_bar.setValue(int(progress))
        self.percent_label.setText(f"{int(progress)}%")
        
        # Determinar fase actual
        for i, phase in enumerate(self.phases):
            if phase["range"][0] <= progress <= phase["range"][1]:
                if i != self.current_phase:
                    self.activate_phase(i)
                break
    
    def activate_phase(self, index):
        """Activa una fase con animación"""
        # Desactivar anterior
        if 0 <= self.current_phase < len(self.phase_widgets):
            self.deactivate_widget(self.phase_widgets[self.current_phase])
        
        # Activar nueva
        if 0 <= index < len(self.phase_widgets):
            self.activate_widget(self.phase_widgets[index])
            self.current_phase = index
    
    def activate_widget(self, widget):
        """Aplica efectos de activación"""
        phase_color = widget.phase_data["color"]
        
        widget.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {UltraTheme.SURFACE_BRIGHT},
                    stop:1 {UltraTheme.SURFACE_LIGHT});
                border: 2px solid {phase_color};
                border-radius: 10px;
            }}
        """)
        
        widget.abbr_label.setStyleSheet(f"""
            color: {phase_color};
            font-size: 20px;
            font-weight: bold;
        """)
        
        # Efecto de glow
        shadow = widget.graphicsEffect()
        if shadow:
            shadow.setColor(QColor(phase_color))
            shadow.setBlurRadius(20)
            shadow.setOffset(0, 0)
        
        widget.is_active = True
    
    def deactivate_widget(self, widget):
        """Quita efectos de activación"""
        widget.setStyleSheet(f"""
            QFrame {{
                background: {UltraTheme.SURFACE_LIGHT};
                border: 2px solid transparent;
                border-radius: 10px;
            }}
        """)
        
        widget.abbr_label.setStyleSheet(f"""
            color: {UltraTheme.TEXT_SECONDARY};
            font-size: 18px;
            font-weight: bold;
        """)
        
        # Quitar glow
        shadow = widget.graphicsEffect()
        if shadow:
            shadow.setColor(QColor(0, 0, 0, 50))
            shadow.setBlurRadius(10)
            shadow.setOffset(0, 2)
        
        widget.is_active = False


# ============ PANEL DE MÉTRICAS ANIMADAS ============
class AnimatedMetricsPanel(QFrame):
    """Panel de métricas con animaciones y efectos visuales"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {UltraTheme.SURFACE_LIGHT},
                    stop:1 {UltraTheme.SURFACE});
                border-radius: 16px;
                border: 1px solid {UltraTheme.SURFACE_BRIGHT};
            }}
        """)
        
        # Añadir sombra
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)
        
        self.metrics = {}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(20)
        
        # Título elegante
        title = QLabel("BIOMECHANICAL METRICS")
        title.setStyleSheet(f"""
            color: {UltraTheme.TEXT_MUTED};
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 2px;
        """)
        layout.addWidget(title)
        
        # Grid de métricas
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(15)
        
        metrics_config = [
            ("Peak Pressure", "kPa", UltraTheme.GRADIENT_DANGER, "↑"),
            ("Contact Area", "cm²", UltraTheme.GRADIENT_SUCCESS, "◆"),
            ("CoP Velocity", "mm/s", UltraTheme.GRADIENT_PRIMARY, "→"),
            ("Asymmetry", "%", UltraTheme.GRADIENT_WARNING, "⚖"),
        ]
        
        for i, (name, unit, gradient, icon) in enumerate(metrics_config):
            row = i // 2
            col = i % 2
            
            metric_widget = self.create_metric_card(name, unit, gradient, icon)
            metrics_grid.addWidget(metric_widget, row, col)
            self.metrics[name] = metric_widget
        
        layout.addLayout(metrics_grid)
    
    def create_metric_card(self, name, unit, gradient, icon):
        """Crea una tarjeta de métrica con diseño premium"""
        card = QFrame()
        card.setFixedHeight(100)
        card.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {UltraTheme.BACKGROUND},
                    stop:1 {UltraTheme.SURFACE});
                border-radius: 12px;
                border: 1px solid {UltraTheme.SURFACE_LIGHT};
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(5)
        
        # Header con icono y nombre
        header = QHBoxLayout()
        
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"""
            color: {gradient[0]};
            font-size: 16px;
        """)
        header.addWidget(icon_label)
        
        name_label = QLabel(name)
        name_label.setStyleSheet(f"""
            color: {UltraTheme.TEXT_MUTED};
            font-size: 11px;
            font-weight: 500;
        """)
        header.addWidget(name_label)
        header.addStretch()
        
        layout.addLayout(header)
        
        # Valor con animación
        value_container = QHBoxLayout()
        
        value_label = QLabel("0.0")
        value_label.setStyleSheet(f"""
            color: {gradient[0]};
            font-size: 28px;
            font-weight: bold;
        """)
        value_container.addWidget(value_label)
        
        unit_label = QLabel(unit)
        unit_label.setStyleSheet(f"""
            color: {UltraTheme.TEXT_DIM};
            font-size: 12px;
            padding-top: 8px;
        """)
        value_container.addWidget(unit_label)
        value_container.addStretch()
        
        layout.addLayout(value_container)
        
        # Mini gráfico de tendencia
        trend_widget = QWidget()
        trend_widget.setFixedHeight(3)
        trend_widget.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {gradient[0]},
                stop:0.5 {gradient[1]},
                stop:1 {gradient[2]});
            border-radius: 2px;
        """)
        layout.addWidget(trend_widget)
        
        # Guardar referencias
        card.value_label = value_label
        card.trend_widget = trend_widget
        
        return card
    
    def update_metric(self, name, value):
        """Actualiza con animación suave"""
        if name in self.metrics:
            card = self.metrics[name]
            card.value_label.setText(f"{value:.1f}")
            
            # Animar el indicador de tendencia
            self.animate_trend(card.trend_widget)
    
    def animate_trend(self, widget):
        """Anima el indicador de tendencia"""
        # Crear efecto de pulso
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(500)
        animation.setStartValue(0.3)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutQuad)
        animation.start()


# ============ VENTANA PRINCIPAL ULTRA PREMIUM ============
class UltraPremiumFootLabUI(QMainWindow):
    """Interfaz principal con diseño de clase mundial"""
    
    def __init__(self):
        super().__init__()
        
        # Configuración de ventana
        self.setWindowTitle("FootLab Ultra • Advanced Baropodometry System")
        self.setStyleSheet(f"background: {UltraTheme.BACKGROUND};")
        self.resize(1920, 1080)
        
        # Centrar ventana
        self.center_window()
        
        # Simulador
        self.simulator = RealisticGaitSimulator()
        self.simulator.data_ready.connect(self.update_visualization)
        
        # Setup UI
        self.setup_ui()
        
        # Estado
        self.is_running = False
        
    def center_window(self):
        """Centra la ventana en la pantalla"""
        from PySide6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
    
    def setup_ui(self):
        """Construye la interfaz ultra premium"""
        central = QWidget()
        self.setCentralWidget(central)
        
        # Layout principal con márgenes elegantes
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(25)
        
        # Panel izquierdo
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Panel central
        center_panel = self.create_visualization_panel()
        main_layout.addWidget(center_panel, 3)
        
        # Panel derecho
        right_panel = self.create_metrics_panel()
        main_layout.addWidget(right_panel, 1)
    
    def create_control_panel(self):
        """Panel de control con diseño elegante"""
        panel = QFrame()
        panel.setMaximumWidth(350)
        panel.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {UltraTheme.SURFACE_LIGHT},
                    stop:1 {UltraTheme.SURFACE});
                border-radius: 20px;
                border: 1px solid {UltraTheme.SURFACE_BRIGHT};
            }}
        """)
        
        # Sombra elegante
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 10)
        panel.setGraphicsEffect(shadow)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        # Logo/Branding
        logo_container = QWidget()
        logo_layout = QVBoxLayout(logo_container)
        
        logo = QLabel("FOOTLAB")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"""
            color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {UltraTheme.CYAN},
                stop:1 {UltraTheme.GRADIENT_PRIMARY[1]});
            font-size: 32px;
            font-weight: 800;
            letter-spacing: 5px;
        """)
        logo_layout.addWidget(logo)
        
        subtitle = QLabel("ULTRA PREMIUM EDITION")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"""
            color: {UltraTheme.TEXT_MUTED};
            font-size: 10px;
            letter-spacing: 3px;
            margin-top: -5px;
        """)
        logo_layout.addWidget(subtitle)
        
        layout.addWidget(logo_container)
        
        # Separador elegante
        separator = QFrame()
        separator.setFixedHeight(2)
        separator.setStyleSheet(f"""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 transparent,
                stop:0.5 {UltraTheme.SURFACE_BRIGHT},
                stop:1 transparent);
        """)
        layout.addWidget(separator)
        
        # Selector de fuente
        source_label = QLabel("DATA SOURCE")
        source_label.setStyleSheet(f"""
            color: {UltraTheme.TEXT_MUTED};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 2px;
            margin-bottom: -10px;
        """)
        layout.addWidget(source_label)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Ultra Simulator", "NURVV BLE Pro", "Clinical Replay"])
        self.source_combo.setFixedHeight(45)
        self.source_combo.setStyleSheet(f"""
            QComboBox {{
                background: {UltraTheme.BACKGROUND};
                color: {UltraTheme.TEXT_PRIMARY};
                border: 2px solid {UltraTheme.SURFACE_BRIGHT};
                border-radius: 10px;
                padding: 10px 15px;
                font-size: 14px;
                font-weight: 500;
            }}
            QComboBox:hover {{
                border-color: {UltraTheme.CYAN};
                background: {UltraTheme.SURFACE};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid {UltraTheme.CYAN};
                margin-right: 10px;
            }}
        """)
        layout.addWidget(self.source_combo)
        
        # Botones principales
        layout.addSpacing(10)
        
        self.start_btn = self.create_premium_button(
            "START SESSION",
            UltraTheme.GRADIENT_SUCCESS,
            "▶"
        )
        self.start_btn.clicked.connect(self.start_session)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = self.create_premium_button(
            "STOP SESSION",
            UltraTheme.GRADIENT_DANGER,
            "■"
        )
        self.stop_btn.clicked.connect(self.stop_session)
        layout.addWidget(self.stop_btn)
        
        self.calibrate_btn = self.create_premium_button(
            "CALIBRATE",
            UltraTheme.GRADIENT_WARNING,
            "⚙"
        )
        layout.addWidget(self.calibrate_btn)
        
        # Estado de conexión
        layout.addSpacing(20)
        
        status_container = QFrame()
        status_container.setStyleSheet(f"""
            QFrame {{
                background: {UltraTheme.BACKGROUND};
                border-radius: 12px;
                border: 1px solid {UltraTheme.SURFACE_LIGHT};
            }}
        """)
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(15, 12, 15, 12)
        
        self.status_led = QLabel("●")
        self.status_led.setStyleSheet(f"""
            color: {UltraTheme.TEXT_MUTED};
            font-size: 24px;
        """)
        status_layout.addWidget(self.status_led)
        
        status_text_container = QVBoxLayout()
        self.status_text = QLabel("System Ready")
        self.status_text.setStyleSheet(f"""
            color: {UltraTheme.TEXT_PRIMARY};
            font-size: 14px;
            font-weight: 500;
        """)
        status_text_container.addWidget(self.status_text)
        
        self.fps_label = QLabel("0 FPS")
        self.fps_label.setStyleSheet(f"""
            color: {UltraTheme.TEXT_MUTED};
            font-size: 11px;
        """)
        status_text_container.addWidget(self.fps_label)
        
        status_layout.addLayout(status_text_container)
        status_layout.addStretch()
        
        layout.addWidget(status_container)
        
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Panel central de visualización"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Indicador de fase de marcha
        self.gait_phase_widget = AnimatedGaitPhaseWidget()
        layout.addWidget(self.gait_phase_widget)
        
        # Container para heatmaps
        heatmap_container = QFrame()
        heatmap_container.setStyleSheet(f"""
            QFrame {{
                background: {UltraTheme.SURFACE};
                border-radius: 20px;
                border: 1px solid {UltraTheme.SURFACE_LIGHT};
            }}
        """)
        
        # Sombra
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(0, 10)
        heatmap_container.setGraphicsEffect(shadow)
        
        heatmap_layout = QHBoxLayout(heatmap_container)
        heatmap_layout.setContentsMargins(20, 20, 20, 20)
        heatmap_layout.setSpacing(20)
        
        # Heatmaps ultra premium
        self.heatmap_left = UltraPremiumHeatmap("LEFT FOOT")
        self.heatmap_right = UltraPremiumHeatmap("RIGHT FOOT")
        
        heatmap_layout.addWidget(self.heatmap_left)
        heatmap_layout.addWidget(self.heatmap_right)
        
        layout.addWidget(heatmap_container)
        
        return panel
    
    def create_metrics_panel(self):
        """Panel de métricas y análisis"""
        panel = QFrame()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        
        # Panel de métricas animadas
        self.metrics_panel = AnimatedMetricsPanel()
        layout.addWidget(self.metrics_panel)
        
        # Gráfico de fuerza
        force_container = QFrame()
        force_container.setStyleSheet(f"""
            QFrame {{
                background: {UltraTheme.SURFACE};
                border-radius: 16px;
                border: 1px solid {UltraTheme.SURFACE_LIGHT};
            }}
        """)
        
        force_layout = QVBoxLayout(force_container)
        force_layout.setContentsMargins(20, 15, 20, 15)
        
        force_title = QLabel("GROUND REACTION FORCE")
        force_title.setStyleSheet(f"""
            color: {UltraTheme.TEXT_MUTED};
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 2px;
        """)
        force_layout.addWidget(force_title)
        
        # Gráfico placeholder (aquí iría pyqtgraph)
        force_graph = QWidget()
        force_graph.setFixedHeight(200)
        force_graph.setStyleSheet(f"""
            background: {UltraTheme.BACKGROUND};
            border-radius: 8px;
        """)
        force_layout.addWidget(force_graph)
        
        layout.addWidget(force_container)
        
        layout.addStretch()
        
        return panel
    
    def create_premium_button(self, text, gradient, icon=""):
        """Crea un botón ultra premium con efectos"""
        btn = QPushButton(f"{icon}  {text}" if icon else text)
        btn.setFixedHeight(50)
        btn.setCursor(Qt.PointingHandCursor)
        
        btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {gradient[0]},
                    stop:0.5 {gradient[1]},
                    stop:1 {gradient[2]});
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 13px;
                font-weight: 600;
                letter-spacing: 1px;
                padding: 0 20px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {gradient[1]},
                    stop:0.5 {gradient[2]},
                    stop:1 {gradient[0]});
            }}
            QPushButton:pressed {{
                background: {gradient[2]};
                padding-top: 2px;
            }}
        """)
        
        # Sombra con color del gradiente
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(gradient[0]))
        shadow.setOffset(0, 5)
        btn.setGraphicsEffect(shadow)
        
        return btn
    
    def start_session(self):
        """Inicia la sesión con animación"""
        if not self.is_running:
            self.is_running = True
            self.simulator.start()
            
            # Actualizar UI
            self.status_led.setStyleSheet(f"color: {UltraTheme.GREEN}; font-size: 24px;")
            self.status_text.setText("Recording Active")
            
            # Animación del LED
            self.animate_status_led()
    
    def stop_session(self):
        """Detiene la sesión"""
        if self.is_running:
            self.is_running = False
            self.simulator.stop()
            
            # Actualizar UI
            self.status_led.setStyleSheet(f"color: {UltraTheme.TEXT_MUTED}; font-size: 24px;")
            self.status_text.setText("System Ready")
    
    def animate_status_led(self):
        """Anima el LED de estado con pulso"""
        if not self.is_running:
            return
        
        # Crear efecto de pulso
        effect = QGraphicsOpacityEffect()
        self.status_led.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(1000)
        animation.setStartValue(0.3)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.setLoopCount(-1)  # Loop infinito
        animation.start()
    
    def update_visualization(self, data):
        """Actualiza toda la visualización con datos del simulador"""
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
        
        # Actualizar fase de marcha
        self.gait_phase_widget.update_phase(data['gait_phase'])
        
        # Actualizar métricas (valores simulados)
        self.update_metrics(data)
        
        # Actualizar FPS
        self.update_fps_counter()
    
    def update_metrics(self, data):
        """Actualiza las métricas con valores realistas"""
        # Peak pressure
        peak_left = np.max(data['left_pressures'])
        peak_right = np.max(data['right_pressures'])
        self.metrics_panel.update_metric("Peak Pressure", max(peak_left, peak_right))
        
        # Contact area
        threshold = 10
        area_left = np.sum(data['left_pressures'] > threshold) * 2.5
        area_right = np.sum(data['right_pressures'] > threshold) * 2.5
        self.metrics_panel.update_metric("Contact Area", area_left + area_right)
        
        # CoP velocity (simulado)
        velocity = 20 + np.random.normal(0, 5)
        self.metrics_panel.update_metric("CoP Velocity", abs(velocity))
        
        # Asymmetry
        total_left = np.sum(data['left_pressures'])
        total_right = np.sum(data['right_pressures'])
        if total_left + total_right > 0:
            asymmetry = abs(total_left - total_right) / (total_left + total_right) * 100
            self.metrics_panel.update_metric("Asymmetry", asymmetry)
    
    def update_fps_counter(self):
        """Actualiza contador de FPS"""
        if not hasattr(self, 'fps_counter'):
            self.fps_counter = 0
            self.fps_time = time.perf_counter()
        
        self.fps_counter += 1
        
        current_time = time.perf_counter()
        if current_time - self.fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_time)
            self.fps_label.setText(f"{fps:.0f} FPS")
            self.fps_counter = 0
            self.fps_time = current_time


# ============ APLICACIÓN PRINCIPAL ============
def main():
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Configurar para high DPI
    # app.setAttribute(Qt.AA_EnableHighDpiScaling, True) # Not needed in Qt6
    # app.setAttribute(Qt.AA_UseHighDpiPixmaps, True) # Not needed in Qt6
    
    window = UltraPremiumFootLabUI()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()