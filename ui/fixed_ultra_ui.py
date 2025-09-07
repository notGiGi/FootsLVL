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
pg.setConfigOptions(imageAxisOrder='row-major')

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

        self.grid_size = (64, 96)
        self.setup_sensor_positions()
        # --- NUEVO: utilidades y parámetros de “realismo” ---
        self.alpha_heatmap = getattr(self, "alpha_heatmap", 0.88)  # ya lo usas: suavidad temporal
        self.alpha_press   = getattr(self, "alpha_press", 0.88)
        self.alpha_cop     = getattr(self, "alpha_cop", 0.82)
        self.toe_medial_shift = 0.09               # cuánto mover el cluster hacia medial
        self.toe_cluster_weights = (0.60, 0.28, 0.18)  # (hallux, medio, lateral)

        self.foot_axis_anisotropy = 2.2  # gaussiana más larga a lo largo del pie
        self.metatarsal_ridge_gain = 0.20  # realce leve zona metatarsal
        self.toe_cluster_gain = 0.15       # un pelín extra en “dedos”

        # Precalcula grillas normalizadas (y: filas, x: columnas) para speed
        rows, cols = self.grid_size
        ridx, cidx = np.indices((rows, cols))
        self.row_norm = ridx / (rows - 1)  # y ∈ [0,1], 0 arriba (dedos), 1 abajo (talón)
        self.col_norm = cidx / (cols - 1)  # x ∈ [0,1], 0 izq (lateral), 1 der (medial)

        # Máscaras de silueta por pie (ligera asimetría medial/lateral)
        self.left_mask  = self._build_foot_mask(is_left=True)
        self.right_mask = self._build_foot_mask(is_left=False)

        # --- NUEVO: filtros temporales ---
        self.alpha_heatmap = 0.85   # suaviza mapa (0.7–0.9 según gusto)
        self.alpha_press   = 0.85   # suaviza presiones por sensor
        self.alpha_cop     = 0.80   # suaviza trayectoria del CoP

        self.prev_left_heatmap  = None
        self.prev_right_heatmap = None
        self.prev_left_press    = np.zeros(16, dtype=float)
        self.prev_right_press   = np.zeros(16, dtype=float)
        self.prev_cop_left      = None
        self.prev_cop_right     = None

        
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
    
    def _smooth_pulse(self, phase, start, end, feather=6):
        """
        Pulso suave en % de ciclo [0..100].
        Retorna 0..1 activación con borde cosenoidal (feather ~ ms).
        """
        # normaliza a [0,1] dentro del segmento
        if end < start:
            end += 100  # wrap
        p = phase
        if p < start: p += 100
        if p > end:   return 0.0
        t = (p - start) / (end - start)
        # ease-in/out cosenoidal
        return 0.5 - 0.5 * np.cos(np.clip(t, 0, 1) * np.pi)

    def _build_foot_mask(self, is_left=True):
        """
        Silueta del pie en (rows, cols), 0..1.
        Heel elíptico, arco estrecho, antepié ancho, almohadillas de dedos.
        Ligera asimetría medial/lateral por pie y cluster de dedos desplazado hacia medial.
        """
        y = self.row_norm  # 0 dedos (arriba), 1 talón (abajo)
        x = self.col_norm  # 0 lateral externo, 1 medial (arco)

        # Orientación anatómica: antepié (y≈0.2-0.45) ancho; arco (y≈0.5-0.7) estrecho; talón (y>0.75) ovalado.
        # Ancho base como función de y (cónico con curvatura):
        width_base = 0.38 - 0.08 * (y - 0.5)**2  # ~0.3–0.38

        # Ajuste de asimetría medial/lateral por pie (pronación leve):
        medial_bias = 0.02 if is_left else -0.02  # izquierda un poco hacia medial
        x_center = 0.5 + medial_bias

        # Distancia horizontal normalizada al “centro longitudinal” con ancho variable
        dx = (x - x_center) / (width_base + 1e-6)

        # Perfil longitudinal (y): define “alto” de la silueta por elípticas superpuestas
        # Talón (elipse en y>0.75):
        heel = np.exp(-((y - 0.86) / 0.12)**2 - (dx**2) * 0.6)
        # Arco (y≈0.55–0.7) más estrecho:
        arch = np.exp(-((y - 0.62) / 0.15)**2 - (dx**2) * 1.3)
        # Antepié (y≈0.35–0.55) más ancho:
        fore = np.exp(-((y - 0.42) / 0.17)**2 - (dx**2) * 0.9)

        # ---------------------------
        # Dedos (pads) (y≈0.12–0.22)
        # ---------------------------
        # Parámetros (defínelos en __init__):
        #   self.toe_medial_shift = 0.09
        #   self.toe_cluster_weights = (0.60, 0.28, 0.18)  # (hallux, medio, lateral)
        toes_base = np.exp(-((y - 0.16) / 0.06)**2)

        # Centros de los 3 lóbulos (medial = hallux, middle, lateral), desplazados hacia medial
        w_med, w_mid, w_lat = getattr(self, "toe_cluster_weights", (0.60, 0.28, 0.18))
        shift = getattr(self, "toe_medial_shift", 0.09)
        sigma_med, sigma_mid, sigma_lat = 0.055, 0.060, 0.070

        if is_left:
            # En el pie izquierdo, "medial" está hacia x mayor (derecha de la imagen)
            medial_center  = x_center + shift
            middle_center  = x_center + shift * 0.25
            lateral_center = x_center - 0.11
        else:
            # En el pie derecho, "medial" está hacia x menor (izquierda de la imagen)
            medial_center  = x_center - shift
            middle_center  = x_center - shift * 0.25
            lateral_center = x_center + 0.11

        lobes = (
            w_med * np.exp(-((x - medial_center)  / sigma_med)**2) +
            w_mid * np.exp(-((x - middle_center)  / sigma_mid)**2) +
            w_lat * np.exp(-((x - lateral_center) / sigma_lat)**2)
        )

        # Compresión leve horizontal en la banda de dedos para marcar hallux
        toe_narrow = 1.0 - 0.12 * np.abs(x - medial_center) / (width_base + 1e-6)
        toes = toes_base * lobes * np.clip(toe_narrow, 0.85, 1.05)

        # Combinación y feathering
        mask = np.clip(0.55*heel + 0.65*arch + 0.85*fore + 0.75*toes, 0.0, 1.0)
        mask = gaussian_filter(mask, sigma=1.5)  # borde suave
        mask = np.clip(mask, 0.0, 1.0)
        return mask





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
        cycle_phase = ((self.time * 0.8) % 1.0) * 100

        raw_left  = self.calculate_pressures(cycle_phase, True)
        raw_right = self.calculate_pressures((cycle_phase + 50) % 100, False)

        # --- NUEVO: suavizado de presiones por sensor (EMA) ---
        left_pressures  = self.alpha_press * self.prev_left_press  + (1 - self.alpha_press) * raw_left
        right_pressures = self.alpha_press * self.prev_right_press + (1 - self.alpha_press) * raw_right
        self.prev_left_press  = left_pressures
        self.prev_right_press = right_pressures

        # Heatmaps del frame actual
        left_heatmap_now  = self.create_heatmap(left_pressures, True)
        right_heatmap_now = self.create_heatmap(right_pressures, False)

        # --- NUEVO: suavizado temporal de heatmaps (EMA) ---
        if self.prev_left_heatmap is None:
            left_heatmap = left_heatmap_now
        else:
            left_heatmap = self.alpha_heatmap * self.prev_left_heatmap + (1 - self.alpha_heatmap) * left_heatmap_now
        self.prev_left_heatmap = left_heatmap

        if self.prev_right_heatmap is None:
            right_heatmap = right_heatmap_now
        else:
            right_heatmap = self.alpha_heatmap * self.prev_right_heatmap + (1 - self.alpha_heatmap) * right_heatmap_now
        self.prev_right_heatmap = right_heatmap

        # CoP crudo
        cop_left_now  = self.calculate_cop(left_pressures, True)
        cop_right_now = self.calculate_cop(right_pressures, False)

        # --- NUEVO: suavizado de CoP (EMA) ---
        if self.prev_cop_left is None:
            cop_left = cop_left_now
        else:
            cop_left = ( self.alpha_cop * np.array(self.prev_cop_left)
                    + (1 - self.alpha_cop) * np.array(cop_left_now) )
            cop_left = (float(cop_left[0]), float(cop_left[1]))
        self.prev_cop_left = cop_left

        if self.prev_cop_right is None:
            cop_right = cop_right_now
        else:
            cop_right = ( self.alpha_cop * np.array(self.prev_cop_right)
                        + (1 - self.alpha_cop) * np.array(cop_right_now) )
            cop_right = (float(cop_right[0]), float(cop_right[1]))
        self.prev_cop_right = cop_right

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
        """
        Presiones por fase, con envolventes suaves y pequeñas modulaciones.
        phase: 0..100
        """
        p = np.zeros(16, dtype=float)

        # Envolventes por región (porcentajes de ciclo típicos)
        heel_env   = self._smooth_pulse(phase, 0, 25)        # heel strike → early stance
        mid_env    = self._smooth_pulse(phase, 15, 50)       # mid-stance (mediopié)
        meta_env   = self._smooth_pulse(phase, 38, 70)       # metatarsos (propulsión)
        toes_env   = self._smooth_pulse(phase, 55, 80)       # toe-off

        # Heel (0..2)
        p[0:3] = np.array([0.8, 1.0, 0.8]) * 100 * heel_env

        # Midfoot (3..6)
        p[3:7] = np.array([0.6, 0.85, 0.85, 0.6]) * 80 * mid_env

        # Metatarsals (7..11)
        p[7:12] = np.array([0.7, 0.88, 1.0, 0.88, 0.72]) * 95 * meta_env

        # Toes (12..15)
        p[12:16] = np.array([0.95, 0.75, 0.75, 0.65]) * 85 * toes_env

        # Micro-modulación senoidal para continuidad y “rolling”
        roll = 0.04 * np.sin(2 * np.pi * ((self.time * 0.8) % 1.0))
        p *= (1.0 + roll)

        # Ruido leve y no granulado
        p += np.random.normal(0, 1.2, p.shape)

        # Simular ligera asimetría izquierda/derecha (distribución de carga)
        asym = 1.03 if is_left else 0.97
        p *= asym

        return np.maximum(0.0, p)


    
    # --- Reemplaza create_heatmap completo por esto ---
    def create_heatmap(self, pressures, is_left):
        """
        Heatmap con gaussianas anisotrópicas (largo a lo largo del pie),
        realce sutil en metatarsos y máscara de silueta de pie.
        """
        sensors = self.sensors_left if is_left else self.sensors_right
        mask    = self.left_mask     if is_left else self.right_mask

        rows, cols = self.grid_size
        y = self.row_norm  # (rows, cols)  0 arriba (dedos), 1 abajo (talón)
        x = self.col_norm  # (rows, cols)  0 izq, 1 der

        # Ejes anisotrópicos: más largo en y (longitud del pie)
        # σ^2 efectivos en cada dirección:
        sig2_y = 0.025 * self.foot_axis_anisotropy   # along-foot
        sig2_x = 0.025 / self.foot_axis_anisotropy   # across-foot

        heatmap = np.zeros((rows, cols), dtype=float)

        for i, (sx, sy) in enumerate(sensors):
            p = pressures[i]
            if p <= 0: 
                continue

            dx2 = (x - sx) ** 2
            dy2 = (y - sy) ** 2
            g = np.exp(-(dx2 / (sig2_x + 1e-9) + dy2 / (sig2_y + 1e-9))) * p
            heatmap += g

        # Realce sutil en “metatarsal ridge” (y ~ 0.35–0.5)
        meta_band = np.exp(-((y - 0.42) / 0.10) ** 2)
        heatmap *= (1.0 + self.metatarsal_ridge_gain * meta_band)

        # Toes boost leve (y ~ 0.12–0.22)
        toes_band = np.exp(-((y - 0.16) / 0.06) ** 2)
        heatmap *= (1.0 + self.toe_cluster_gain * toes_band)

        # Suavizado espacial leve
        heatmap = gaussian_filter(heatmap, sigma=2.4)

        # Aplica máscara de silueta para “recortar” fuera del pie
        heatmap *= mask

        # Normaliza
        maxv = heatmap.max()
        if maxv > 1e-6:
            heatmap = heatmap / maxv

        return heatmap






    # --- Reemplaza calculate_cop completo por esto ---
    def calculate_cop(self, pressures, is_left):
        """Calcula Centro de Presión en pixeles: x=columnas, y=filas"""
        sensors = self.sensors_left if is_left else self.sensors_right

        total = float(np.sum(pressures))
        if total > 10:
            # x = promedio ponderado de la coord x de sensores (col)
            cop_x_norm = float(np.sum(sensors[:, 0] * pressures) / total)
            # y = promedio ponderado de la coord y de sensores (fila)
            cop_y_norm = float(np.sum(sensors[:, 1] * pressures) / total)

            x_pix = cop_x_norm * (self.grid_size[1] - 1)  # columnas (ancho)
            y_pix = cop_y_norm * (self.grid_size[0] - 1)  # filas (alto)
            return (x_pix, y_pix)

        return ((self.grid_size[1] - 1) / 2.0, (self.grid_size[0] - 1) / 2.0)


    
    def stop(self):
        self.running = False
        self.wait()


# ============ WIDGET DE HEATMAP SIMPLE ============
class SimpleHeatmapWidget(pg.GraphicsLayoutWidget):
    """Heatmap simplificado que funciona (1 sola ImageItem, rango estable)"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setBackground(UltraTheme.BACKGROUND)

        # Plot
        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)       # pixel aspect fijo
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.invertY(True)               # origen arriba-izquierda

        # Una sola ImageItem
        self.img = pg.ImageItem()
        self.img.setAutoDownsample(True)      # menos shimmering al escalar
        self.plot.addItem(self.img)

        if title:
            self.plot.setTitle(title, color=UltraTheme.TEXT_MUTED, size='10pt')

        # Colormap médico
        colors = [
            (0, 0, 30),
            (0, 0, 100),
            (0, 100, 200),
            (0, 200, 200),
            (0, 255, 100),
            (200, 255, 0),
            (255, 100, 0),
            (255, 0, 0),
        ]
        positions = np.linspace(0, 1, len(colors))
        cmap = pg.ColorMap(positions, colors)
        self.img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))

        # Línea y punto de CoP
        self.cop_trail = self.plot.plot([], [], pen=pg.mkPen(color=UltraTheme.CYAN, width=2))
        self.cop_point = self.plot.plot([], [], pen=None, symbol='o',
                                        symbolBrush=UltraTheme.CYAN, symbolSize=10)

        # Estado para bloqueo de rango en primer frame
        self._range_locked = False
        self.plot.getViewBox().setMouseEnabled(False, False)
        self.plot.disableAutoRange()

        # Historial de CoP
        self.cop_history = []
        self.max_history = 20

    def update_display(self, heatmap, cop_x, cop_y):
        """Actualiza visualización"""
        if heatmap is not None and heatmap.size > 0:
            self.img.setImage(heatmap, autoLevels=False, levels=[0, 1])

            if not self._range_locked:
                h, w = heatmap.shape  # (rows, cols)
                vb = self.plot.getViewBox()
                vb.enableAutoRange(True, True)  # autorange 1 vez
                vb.autoRange()
                # Fija rango exacto al tamaño de la imagen y bloquea
                self.plot.setXRange(0, w, padding=0)
                self.plot.setYRange(0, h, padding=0)
                vb.enableAutoRange(False, False)
                self._range_locked = True

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

    def shutdown(self):
        if self.simulator.isRunning():
            self.simulator.stop()  # ya llama wait() por dentro

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