"""
Integracion de datasets con la UI de FootLab
Archivo: ui/dataset_integration.py
Instalacion: copiar este codigo completo a ui/dataset_integration.py
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLabel, QComboBox, QSlider, QCheckBox, QProgressBar,
                               QGroupBox, QFileDialog, QMessageBox, QSpinBox)
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
import sys
import os

# Agregar el directorio raíz al path para importar core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset_loader import DatasetManager, DatasetReplaySource, Sample

class DatasetControlWidget(QWidget):
    """Widget de control para datasets en la UI principal"""
    
    # Señales
    dataset_sample = Signal(object)  # Emite Sample
    dataset_started = Signal(str)    # Emite nombre del dataset
    dataset_stopped = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_manager = DatasetManager()
        self.current_source = None
        self.setup_ui()
        self.refresh_datasets()
    
    def setup_ui(self):
        """Configura la interfaz"""
        layout = QVBoxLayout(self)
        
        # Título
        title = QLabel("📊 Dataset Player")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Selector de dataset
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # Combo box de datasets
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        dataset_layout.addWidget(QLabel("Available Datasets:"))
        dataset_layout.addWidget(self.dataset_combo)
        
        # Botón de descarga/refresh
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_datasets)
        self.download_btn = QPushButton("📥 Download")
        self.download_btn.clicked.connect(self.download_dataset)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.download_btn)
        dataset_layout.addLayout(btn_layout)
        
        layout.addWidget(dataset_group)
        
        # Controles de reproducción
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout(playback_group)
        
        # Velocidad de reproducción
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 300)  # 0.1x a 3.0x
        self.speed_slider.setValue(100)  # 1.0x
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        playback_layout.addLayout(speed_layout)
        
        # Loop checkbox
        self.loop_checkbox = QCheckBox("Loop playback")
        self.loop_checkbox.setChecked(True)
        playback_layout.addWidget(self.loop_checkbox)
        
        # Controles de reproducción
        controls_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self.play_dataset)
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_dataset)
        self.stop_btn.setEnabled(False)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        playback_layout.addLayout(controls_layout)
        
        layout.addWidget(playback_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        
        layout.addWidget(status_group)
        
        # Info del dataset actual
        info_group = QGroupBox("Dataset Info")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("No dataset selected")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_group)
    
    def refresh_datasets(self):
        """Actualiza lista de datasets disponibles"""
        self.dataset_combo.clear()
        
        # Datasets predefinidos
        predefined = [
            ("MUN104", "Anatomical template (104 subjects)"),
            ("Load Custom...", "Load custom dataset file")
        ]
        
        for name, desc in predefined:
            self.dataset_combo.addItem(f"{name} - {desc}", name)
        
        # Datasets ya descargados
        available = self.dataset_manager.list_datasets()
        for name, info in available.items():
            if name not in [p[0] for p in predefined]:
                self.dataset_combo.addItem(f"✅ {name} - {info['description']}", name)
    
    def on_dataset_changed(self):
        """Maneja cambio de dataset seleccionado"""
        current_data = self.dataset_combo.currentData()
        if current_data == "Load Custom...":
            self.load_custom_dataset()
        else:
            self.update_dataset_info(current_data)
    
    def update_dataset_info(self, dataset_name):
        """Actualiza información del dataset"""
        if not dataset_name:
            self.info_label.setText("No dataset selected")
            return
            
        if dataset_name == "MUN104":
            self.info_label.setText(
                "📋 MUN104 Anatomical Template\n"
                "• Source: University of Münster\n"
                "• Subjects: 104 healthy individuals\n"
                "• Type: Static pressure template\n"
                "• Format: Interpolated to 16 NURVV sensors\n"
                "• Usage: Reference/validation data"
            )
        else:
            available = self.dataset_manager.list_datasets()
            if dataset_name in available:
                info = available[dataset_name]
                self.info_label.setText(f"📊 {dataset_name}\n{info['description']}")
            else:
                self.info_label.setText(f"Dataset: {dataset_name}\nNot downloaded yet.")
    
    def download_dataset(self):
        """Descarga el dataset seleccionado"""
        dataset_name = self.dataset_combo.currentData()
        if not dataset_name or dataset_name == "Load Custom...":
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText(f"Downloading {dataset_name}...")
        self.download_btn.setEnabled(False)
        
        # Usar QTimer para no bloquear UI
        QTimer.singleShot(100, lambda: self._download_dataset_async(dataset_name))
    
    def _download_dataset_async(self, dataset_name):
        """Descarga dataset en background"""
        try:
            path = self.dataset_manager.ensure_dataset(dataset_name)
            if path:
                self.status_label.setText(f"✅ {dataset_name} downloaded successfully")
                self.refresh_datasets()
            else:
                self.status_label.setText(f"❌ Failed to download {dataset_name}")
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            self.download_btn.setEnabled(True)
    
    def load_custom_dataset(self):
        """Carga dataset personalizado"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Dataset", "", 
            "Dataset files (*.csv *.h5 *.npz *.mat);;All files (*)"
        )
        
        if file_path:
            # TODO: Implementar carga de datasets personalizados
            QMessageBox.information(self, "Info", 
                                   f"Custom dataset loading not implemented yet.\n"
                                   f"Selected: {file_path}")
    
    def on_speed_changed(self):
        """Maneja cambio de velocidad"""
        speed = self.speed_slider.value() / 100.0
        self.speed_label.setText(f"{speed:.1f}x")
        
        # Actualizar velocidad si está reproduciendo
        if self.current_source:
            self.current_source.replay_speed = speed
    
    def play_dataset(self):
        """Inicia reproducción del dataset"""
        dataset_name = self.dataset_combo.currentData()
        if not dataset_name or dataset_name == "Load Custom...":
            QMessageBox.warning(self, "Warning", "Please select a valid dataset")
            return
        
        # Parámetros de reproducción
        speed = self.speed_slider.value() / 100.0
        loop = self.loop_checkbox.isChecked()
        
        try:
            self.current_source = self.dataset_manager.create_source(
                dataset_name, 
                replay_speed=speed, 
                loop=loop
            )
            
            if self.current_source:
                self.current_source.start(self._on_sample_received)
                
                # Actualizar UI
                self.play_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.status_label.setText(f"▶️ Playing {dataset_name}")
                
                # Emitir señal
                self.dataset_started.emit(dataset_name)
            else:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {dataset_name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error starting dataset: {str(e)}")
    
    def stop_dataset(self):
        """Detiene reproducción"""
        if self.current_source:
            self.current_source.stop()
            self.current_source = None
        
        # Actualizar UI
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("⏹️ Stopped")
        
        # Emitir señal
        self.dataset_stopped.emit()
    
    def _on_sample_received(self, sample: Sample):
        """Maneja muestra recibida del dataset"""
        # Convertir Sample a formato dict compatible
        sample_dict = {
            "t_ms": sample.t_ms,
            "left": sample.left,
            "right": sample.right
        }
        
        # Emitir señal para que la UI principal la procese
        self.dataset_sample.emit(sample_dict)

class DatasetWindow(QWidget):
    """Ventana independiente para control de datasets"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FootLab - Dataset Player")
        self.setFixedSize(400, 600)
        
        layout = QVBoxLayout(self)
        
        # Control widget
        self.control_widget = DatasetControlWidget()
        layout.addWidget(self.control_widget)
        
        # Conectar señales para demo
        self.control_widget.dataset_sample.connect(self.on_sample_demo)
        self.control_widget.dataset_started.connect(self.on_dataset_started)
        self.control_widget.dataset_stopped.connect(self.on_dataset_stopped)
    
    def on_sample_demo(self, sample_dict):
        """Demo handler para mostrar datos recibidos"""
        left_total = sum(sample_dict["left"])
        right_total = sum(sample_dict["right"])
        print(f"Sample t={sample_dict['t_ms']}ms: L={left_total:.1f}, R={right_total:.1f}")
    
    def on_dataset_started(self, dataset_name):
        """Maneja inicio de dataset"""
        print(f"🚀 Dataset started: {dataset_name}")
    
    def on_dataset_stopped(self):
        """Maneja parada de dataset"""
        print("⏹️ Dataset stopped")

# Función para integrar con main_window.py existente
def integrate_with_main_window(main_window):
    """
    Integra el control de datasets con la ventana principal existente
    
    Args:
        main_window: Instancia de la ventana principal de FootLab
    """
    
    # Crear widget de control
    dataset_control = DatasetControlWidget()
    
    # Conectar señales con los métodos existentes de la ventana principal
    def on_dataset_sample(sample_dict):
        """Adaptador para sample de dataset"""
        # Convertir a formato esperado por la UI existente
        if hasattr(main_window, 'on_sample_received'):
            main_window.on_sample_received(sample_dict)
        elif hasattr(main_window, 'update_heatmaps'):
            # Si usa update_heatmaps directamente
            left_pressures = sample_dict["left"]
            right_pressures = sample_dict["right"]
            main_window.update_heatmaps(left_pressures, right_pressures)
    
    def on_dataset_started(dataset_name):
        """Cuando inicia dataset, detener simulador si existe"""
        if hasattr(main_window, 'stop_acquisition'):
            main_window.stop_acquisition()
        print(f"📊 Switched to dataset: {dataset_name}")
    
    def on_dataset_stopped():
        """Cuando para dataset"""
        print("📊 Dataset playback stopped")
    
    # Conectar señales
    dataset_control.dataset_sample.connect(on_dataset_sample)
    dataset_control.dataset_started.connect(on_dataset_started) 
    dataset_control.dataset_stopped.connect(on_dataset_stopped)
    
    # Agregar a la UI principal (ajustar según layout existente)
    if hasattr(main_window, 'main_layout'):
        main_window.main_layout.addWidget(dataset_control)
    elif hasattr(main_window, 'addDockWidget'):
        # Si usa dock widgets
        from PySide6.QtWidgets import QDockWidget
        dock = QDockWidget("Dataset Player", main_window)
        dock.setWidget(dataset_control)
        main_window.addDockWidget(Qt.RightDockWidgetArea, dock)
    else:
        # Fallback: ventana separada
        dataset_window = DatasetWindow()
        dataset_window.show()
        return dataset_window
    
    return dataset_control

# Para testing independiente
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Crear ventana de prueba
    window = DatasetWindow()
    window.show()
    
    sys.exit(app.exec())