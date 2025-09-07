import os, asyncio
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QFileDialog, QComboBox, QMessageBox, QSlider, QCheckBox
)
from PySide6.QtCore import QObject, Signal, QTimer, Qt
from core.ble_nurvv import NurvvBleSource
from ui.heatmap_view import HeatmapView
from ui.charts import RollingGRF
from core.simulator import SimulatorSource
from core.replay import ReplaySource
from core.session_store import SessionWriter
from core.processing import (
    FootState, center_of_pressure, grf, detect_step,
    zones_mask, update_pti, cadence_from_steps
)
from core.interpolation import foot_layout_24
from core.calibration import init_calibration, apply_calibration
from reports.report import export_pdf

DARK_QSS = """
QWidget { background-color: #15171a; color: #dfe3ea; font-size: 12.5px; }
QLabel#appTitle { font-size: 18px; font-weight: 700; color: #f1f3f7; }
QLabel#panelTitle { font-size: 14px; font-weight: 600; color: #e5e9f0; }
QFrame#sidePanel { background-color: #1c1f24; border: 1px solid #2a2f36; border-radius: 8px; }
QFrame#headerBar { background-color: #1c1f24; border: 1px solid #2a2f36; border-radius: 8px; }
QPushButton { background-color: #2a2f36; border: 1px solid #37404a; border-radius: 6px; padding: 6px 10px; }
QPushButton:hover { background-color: #313742; }
QPushButton:pressed { background-color: #272c33; }
QComboBox, QSlider, QCheckBox { background-color: #22262c; border: 1px solid #343b44; border-radius: 6px; }
"""

class UiBridge(QObject):
    sample_arrived = Signal(dict)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FootLab — Smart Insole Viewer (PC)")
        self.setStyleSheet(DARK_QSS)
        self.resize(1280, 820)

        self.n = 24
        # Coordenadas por pie (L/R) para CoP y zonas
        self.coordsL = foot_layout_24(left=True)
        self.coordsR = foot_layout_24(left=False)
        self.masksL = zones_mask(self.coordsL)
        self.masksR = zones_mask(self.coordsR)

        # Estado por pie
        self.stateL = FootState()
        self.stateR = FootState()

        # Calibración
        self.calib = init_calibration(self.n)

        # Fuente de datos
        self.source = None  # SimulatorSource | ReplaySource | (luego) BleSource
        self.bridge = UiBridge()
        self.bridge.sample_arrived.connect(self._on_sample)

        # ===== Layout general =====
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        # Side panel
        side = QFrame(); side.setObjectName("sidePanel")
        sideLay = QVBoxLayout(side); sideLay.setContentsMargins(12, 12, 12, 12); sideLay.setSpacing(10)

        app_title = QLabel("FootLab — Controls"); app_title.setObjectName("appTitle")
        sideLay.addWidget(app_title)

        # Fuente
        self.cmb_source = QComboBox()
        self.cmb_source.addItems(["Simulator", "NURVV BLE", "Replay (CSV)"])  # BLE se activará luego
        sideLay.addWidget(QLabel("Source:"))
        sideLay.addWidget(self.cmb_source)

        # Botones
        self.btn_start = QPushButton("Start")
        self.btn_stop  = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_zero  = QPushButton("Calibrate Zero (2s)")
        self.btn_export= QPushButton("Export PDF")
        for b in (self.btn_start, self.btn_stop, self.btn_zero, self.btn_export):
            sideLay.addWidget(b)

        sideLay.addSpacing(8)
        sideLay.addWidget(QLabel("Heatmap settings:"))

        # Colormap
        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["viridis", "plasma", "inferno", "magma"])
        sideLay.addWidget(QLabel("Colormap:"))
        sideLay.addWidget(self.cmb_cmap)

        # Intensidad
        self.sld_intensity = QSlider(Qt.Horizontal)
        self.sld_intensity.setRange(20, 300)  # 0.2..3.0x
        self.sld_intensity.setValue(100)
        sideLay.addWidget(QLabel("Intensity scale"))
        sideLay.addWidget(self.sld_intensity)

        # Smoothing
        self.chk_smooth = QCheckBox("Smooth heatmap (3×3)")
        self.chk_smooth.setChecked(True)
        sideLay.addWidget(self.chk_smooth)

        sideLay.addSpacing(12)
        self.lbl_cadL = QLabel("Cadence L: 0 spm")
        self.lbl_cadR = QLabel("Cadence R: 0 spm")
        sideLay.addWidget(self.lbl_cadL)
        sideLay.addWidget(self.lbl_cadR)

        sideLay.addStretch(1)

        # Área principal
        mainCol = QVBoxLayout(); mainCol.setSpacing(10)

        header = QFrame(); header.setObjectName("headerBar")
        hLay = QHBoxLayout(header); hLay.setContentsMargins(12, 8, 12, 8)
        hTitle = QLabel("Live Pressure Maps & Metrics"); hTitle.setObjectName("panelTitle")
        hLay.addWidget(hTitle); hLay.addStretch(1)
        mainCol.addWidget(header)

        self.heatmap = HeatmapView(grid_w=64, grid_h=96, n_sensors=self.n, title="Heatmap (Left / Right)")
        mainCol.addWidget(self.heatmap, 3)

        self.chart = RollingGRF()
        mainCol.addWidget(self.chart, 1)

        root.addWidget(side, 0)
        root.addLayout(mainCol, 1)

        self.setCentralWidget(central)

        # Señales
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_zero.clicked.connect(self.on_zero)
        self.btn_export.clicked.connect(self.on_export)

        self.cmb_cmap.currentTextChanged.connect(self._on_cmap_change)
        self.sld_intensity.valueChanged.connect(self._on_intensity_change)
        self.chk_smooth.toggled.connect(self._on_smooth_toggle)

        # Sesión (auto-guardado)
        self.writer = None
        self.session_active = False
        self.t0_ms = None

        # Timer UI (cadencia labels)
        self.ui_timer = QTimer(self); self.ui_timer.setInterval(500)
        self.ui_timer.timeout.connect(self._refresh_labels); self.ui_timer.start()

        # Buffers calib
        self._calib_buffer_L, self._calib_buffer_R = [], []
        self._calib_collecting = False

        # Settings iniciales
        self._on_cmap_change(self.cmb_cmap.currentText())
        self._on_intensity_change(self.sld_intensity.value())
        self._on_smooth_toggle(self.chk_smooth.isChecked())

    # ==== Controles ====
    def _on_cmap_change(self, name: str):
        self.heatmap.set_colormap(name)

    def _on_intensity_change(self, val: int):
        self.heatmap.set_intensity_scale(val / 100.0)

    def _on_smooth_toggle(self, state: bool):
        self.heatmap.set_smoothing(state)

    # ==== Flujo ====
    def on_start(self):
        src = self.cmb_source.currentText()
        if src == "Simulator":
            self.source = SimulatorSource(n_sensors=self.n, freq=100, base_amp=40.0)
            self._start_source(self.source)
            self.writer = SessionWriter(); self.writer.open(self.n)
            self.session_active = True; self.t0_ms = None
            
        elif src == "NURVV BLE":
            # Crear fuente BLE con configuración
            self.source = NurvvBleSource(
                n_sensors=self.n, 
                freq=100,
                auto_connect=True,
                sync_window_ms=50.0
            )
            self._start_source(self.source)
            self.writer = SessionWriter(); self.writer.open(self.n)
            self.session_active = True; self.t0_ms = None
            
            # Mostrar estado de conexión
            QMessageBox.information(self, "NURVV BLE", 
                "Buscando dispositivos NURVV...\n"
                "Asegúrese de que los sensores estén encendidos y en modo emparejamiento.\n"
                "Si no se encuentran dispositivos, se usará el simulador.")
            
        elif src == "Replay (CSV)":
            path, _ = QFileDialog.getOpenFileName(self, "Open session CSV", "sessions", "CSV Files (*.csv)")
            if not path: return
            self.source = ReplaySource(path=path, n_sensors=self.n)
            self._start_source(self.source)
            self.writer = None; self.session_active = False; self.t0_ms = None


    def _start_source(self, source):
        def on_sample(sample):
            self.bridge.sample_arrived.emit(sample)
        source.start(on_sample)

    def on_stop(self):
        if self.source: self.source.stop()
        if self.writer: self.writer.close()
        self.session_active = False
        self.source = None

    def on_zero(self):
        self._calib_buffer_L.clear(); self._calib_buffer_R.clear()
        self._calib_collecting = True
        QTimer.singleShot(2000, self._finish_zero)

    def _finish_zero(self):
        if len(self._calib_buffer_L) > 0:
            arrL = np.vstack(self._calib_buffer_L); self.calib.zero_L = np.mean(arrL, axis=0)
        if len(self._calib_buffer_R) > 0:
            arrR = np.vstack(self._calib_buffer_R); self.calib.zero_R = np.mean(arrR, axis=0)
        self._calib_collecting = False
        QMessageBox.information(self, "Calibración", "Baseline (zero) actualizado.")

    def on_export(self):
        if not self.t0_ms:
            QMessageBox.information(self, "Info", "No hay sesión activa ni replay para exportar."); return
        duration_s = (self.stateL.time_hist[-1] - self.stateL.time_hist[0]) if self.stateL.time_hist else 0.0
        summary = {
            "duration_s": duration_s,
            "cadence_L": cadence_from_steps(self.stateL.step_times),
            "cadence_R": cadence_from_steps(self.stateR.step_times),
            "pti_L": f"{self.stateL.pti_heel:.1f}/{self.stateL.pti_mid:.1f}/{self.stateL.pti_fore:.1f}",
            "pti_R": f"{self.stateR.pti_heel:.1f}/{self.stateR.pti_mid:.1f}/{self.stateR.pti_fore:.1f}",
        }
        out = export_pdf("reports_out", getattr(self.writer, "path", "replay.csv"), summary)
        QMessageBox.information(self, "PDF", f"Reporte generado:\n{out}")

    def _on_sample(self, sample: dict):
        if self._calib_collecting:
            self._calib_buffer_L.append(np.array(sample["left"], dtype=float))
            self._calib_buffer_R.append(np.array(sample["right"], dtype=float))

        # Calibración de baseline
        left  = apply_calibration(np.array(sample["left"], dtype=float),  self.calib.zero_L)
        right = apply_calibration(np.array(sample["right"], dtype=float), self.calib.zero_R)

        # CoP & GRF usando coords específicas por pie
        gL = grf(left); gR = grf(right)
        copL = center_of_pressure(left,  self.coordsL)
        copR = center_of_pressure(right, self.coordsR)

        if self.t0_ms is None: self.t0_ms = sample["t_ms"]
        t_s = (sample["t_ms"] - self.t0_ms) / 1000.0

        self.stateL.grf_hist.append(gL); self.stateL.time_hist.append(t_s)
        self.stateR.grf_hist.append(gR); self.stateR.time_hist.append(t_s)

        onL  = max(5.0, 0.2 * (np.max(self.stateL.grf_hist[-50:]) if len(self.stateL.grf_hist)>10 else 50))
        offL = 0.5 * onL
        onR  = max(5.0, 0.2 * (np.max(self.stateR.grf_hist[-50:]) if len(self.stateR.grf_hist)>10 else 50))
        offR = 0.5 * onR

        evL, inL = detect_step(gL, onL, offL, self.stateL.in_contact)
        evR, inR = detect_step(gR, onR, offR, self.stateR.in_contact)
        self.stateL.in_contact = inL; self.stateR.in_contact = inR
        if evL and inL: self.stateL.step_times.append(t_s)
        if evR and inR: self.stateR.step_times.append(t_s)

        dtL = self.stateL.time_hist[-1] - self.stateL.time_hist[-2] if len(self.stateL.time_hist) >= 2 else 0.0
        dtR = self.stateR.time_hist[-1] - self.stateR.time_hist[-2] if len(self.stateR.time_hist) >= 2 else 0.0
        update_pti(self.stateL, left,  dtL, self.masksL)
        update_pti(self.stateR, right, dtR, self.masksR)

        # Render
        self.heatmap.update_with_sample({"left": left.tolist(), "right": right.tolist()}, copL=copL, copR=copR)
        self.chart.push(sample["t_ms"], gL, gR)

        if self.session_active and self.writer:
            self.writer.write(sample, gL, copL, gR, copR)

    def _refresh_labels(self):
        cL = cadence_from_steps(self.stateL.step_times)
        cR = cadence_from_steps(self.stateR.step_times)
        self.lbl_cadL.setText(f"Cadence L: {cL:.1f} spm")
        self.lbl_cadR.setText(f"Cadence R: {cR:.1f} spm")

    def show_ble_status(self):
        """Muestra el estado de la conexión BLE"""
        if isinstance(self.source, NurvvBleSource):
            status = self.source.get_status()
            msg = f"Estado BLE:\n"
            msg += f"• Pie izquierdo: {'✓ Conectado' if status['connected_left'] else '✗ Desconectado'}\n"
            msg += f"• Pie derecho: {'✓ Conectado' if status['connected_right'] else '✗ Desconectado'}\n"
            msg += f"• Frecuencia: {status['frequency']} Hz"
            QMessageBox.information(self, "Estado NURVV BLE", msg)
