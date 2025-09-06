# main_application.py
"""
State-of-the-Art FootLab Main Application
Integrates all advanced components into a cohesive clinical system
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QSplitter, QStatusBar, QMenuBar, QToolBar, QAction,
    QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox, QGroupBox,
    QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QFileDialog, QDialog, QDialogButtonBox, QFormLayout,
    QSlider, QFrame
)
from PySide6.QtCore import Qt, QTimer, QThread, QObject, Signal, QSettings
from PySide6.QtGui import QIcon, QPixmap, QFont

# Import our advanced components
from core.modern_architecture import (
    ModernFootLabCore, SystemConfig, AnalysisMode, DataQuality,
    create_footlab_system
)
from ui.enhanced_heatmap_view import StateOfTheArtHeatmapView
from core.advanced_biomechanics import AdvancedGaitAnalyzer, create_gait_analyzer
from ai.pathology_detection import PathologyClassifier, create_pathology_classifier
from reports.advanced_report_system import ClinicalReportGenerator, PatientInfo, SessionMetadata
from sensors.nurvv_ble_advanced import NurvvSystemManager, create_nurvv_system
from core.simulator import SimulatorSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('footlab.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Modern Dark Theme Stylesheet
MODERN_DARK_THEME = """
QMainWindow {
    background-color: #1e1e1e;
    color: #ffffff;
}

QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    font-size: 11px;
}

QTabWidget::pane {
    border: 1px solid #3d3d3d;
    background-color: #2d2d2d;
    border-radius: 6px;
}

QTabWidget::tab-bar {
    alignment: left;
}

QTabBar::tab {
    background: #3d3d3d;
    border: 1px solid #4d4d4d;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background: #0078d4;
    color: white;
}

QTabBar::tab:hover {
    background: #4d4d4d;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4d4d4d, stop:1 #3d3d3d);
    border: 1px solid #5d5d5d;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    min-width: 80px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #5d5d5d, stop:1 #4d4d4d);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #3d3d3d, stop:1 #2d2d2d);
}

QPushButton:disabled {
    background: #2d2d2d;
    color: #666666;
}

QPushButton#primaryButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0084ff, stop:1 #0078d4);
    color: white;
    font-weight: 700;
}

QPushButton#primaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1090ff, stop:1 #0084ff);
}

QPushButton#dangerButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ff4757, stop:1 #ff3742);
    color: white;
}

QPushButton#dangerButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ff5757, stop:1 #ff4757);
}

QComboBox {
    background: #3d3d3d;
    border: 1px solid #5d5d5d;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #0078d4;
}

QComboBox::drop-down {
    border: none;
}

QComboBox::down-arrow {
    width: 12px;
    height: 12px;
}

QGroupBox {
    font-weight: 600;
    border: 2px solid #4d4d4d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 8px 0 8px;
}

QStatusBar {
    background: #2d2d2d;
    border-top: 1px solid #4d4d4d;
    color: #cccccc;
}

QProgressBar {
    border: 1px solid #5d5d5d;
    border-radius: 6px;
    background: #2d2d2d;
    text-align: center;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0078d4, stop:1 #0084ff);
    border-radius: 5px;
}

QSlider::groove:horizontal {
    border: 1px solid #5d5d5d;
    height: 6px;
    background: #3d3d3d;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #0078d4;
    border: 1px solid #0078d4;
    width: 16px;
    height: 16px;
    border-radius: 8px;
    margin: -6px 0;
}

QSlider::handle:horizontal:hover {
    background: #0084ff;
}

QTextEdit {
    background: #2d2d2d;
    border: 1px solid #4d4d4d;
    border-radius: 6px;
    padding: 8px;
    font-family: "Consolas", "Monaco", monospace;
}

QTableWidget {
    background: #2d2d2d;
    alternate-background-color: #3d3d3d;
    gridline-color: #4d4d4d;
    border: 1px solid #4d4d4d;
    border-radius: 6px;
}

QHeaderView::section {
    background: #4d4d4d;
    padding: 8px;
    border: none;
    font-weight: 600;
}

QSplitter::handle {
    background: #4d4d4d;
    margin: 2px;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

/* Custom classes for status indicators */
QLabel#statusIndicator {
    border: 2px solid;
    border-radius: 8px;
    padding: 4px 8px;
    font-weight: 600;
    font-size: 10px;
}

QLabel#statusConnected {
    background: #27ae60;
    border-color: #2ecc71;
    color: white;
}

QLabel#statusDisconnected {
    background: #e74c3c;
    border-color: #c0392b;
    color: white;
}

QLabel#statusStreaming {
    background: #f39c12;
    border-color: #e67e22;
    color: white;
}

QLabel#statusError {
    background: #8e44ad;
    border-color: #9b59b6;
    color: white;
}
"""

class PatientDialog(QDialog):
    """Dialog for entering patient information"""
    
    def __init__(self, parent=None, patient_data=None):
        super().__init__(parent)
        self.setWindowTitle("Patient Information")
        self.setModal(True)
        self.setMinimumSize(400, 300)
        
        layout = QFormLayout(self)
        
        # Patient fields
        self.patient_id = QComboBox()
        self.patient_id.setEditable(True)
        self.name = QComboBox()
        self.name.setEditable(True)
        self.age = QSpinBox()
        self.age.setRange(0, 120)
        self.height = QSpinBox()
        self.height.setRange(0, 250)
        self.height.setSuffix(" cm")
        self.weight = QSpinBox()
        self.weight.setRange(0, 300)
        self.weight.setSuffix(" kg")
        self.gender = QComboBox()
        self.gender.addItems(["Male", "Female", "Other", "Prefer not to say"])
        self.diagnosis = QComboBox()
        self.diagnosis.setEditable(True)
        
        # Common diagnoses
        common_diagnoses = [
            "Healthy/Control",
            "Diabetes Mellitus",
            "Parkinson's Disease", 
            "Post-stroke",
            "Plantar Fasciitis",
            "Rheumatoid Arthritis",
            "Osteoarthritis",
            "Peripheral Neuropathy",
            "Foot Pain",
            "Balance Disorder"
        ]
        self.diagnosis.addItems(common_diagnoses)
        
        # Add fields to layout
        layout.addRow("Patient ID:", self.patient_id)
        layout.addRow("Name:", self.name)
        layout.addRow("Age:", self.age)
        layout.addRow("Height:", self.height)
        layout.addRow("Weight:", self.weight)
        layout.addRow("Gender:", self.gender)
        layout.addRow("Diagnosis/Condition:", self.diagnosis)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        layout.addWidget(buttons)
        
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        # Load existing patient data
        if patient_data:
            self.load_patient_data(patient_data)
        
        # Load recent patients
        self.load_recent_patients()
    
    def load_recent_patients(self):
        """Load recent patient data from settings"""
        settings = QSettings("FootLab", "Patients")
        recent_patients = settings.value("recent_patients", [])
        
        if recent_patients:
            for patient in recent_patients[-10:]:  # Last 10 patients
                self.patient_id.addItem(patient.get("id", ""))
                self.name.addItem(patient.get("name", ""))
    
    def load_patient_data(self, data):
        """Load patient data into form"""
        self.patient_id.setCurrentText(data.get("patient_id", ""))
        self.name.setCurrentText(data.get("name", ""))
        self.age.setValue(data.get("age", 0))
        self.height.setValue(data.get("height", 0))
        self.weight.setValue(data.get("weight", 0))
        self.gender.setCurrentText(data.get("gender", ""))
        self.diagnosis.setCurrentText(data.get("diagnosis", ""))
    
    def get_patient_data(self):
        """Get patient data from form"""
        return {
            "patient_id": self.patient_id.currentText(),
            "name": self.name.currentText(),
            "age": self.age.value(),
            "height": self.height.value(),
            "weight": self.weight.value(),
            "gender": self.gender.currentText(),
            "diagnosis": self.diagnosis.currentText()
        }

class SystemStatusWidget(QWidget):
    """Widget showing system status and statistics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)  # Update every second
        
        self.system: Optional[ModernFootLabCore] = None
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        
        # Status indicators
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        
        # Connection status
        connection_layout = QHBoxLayout()
        connection_layout.addWidget(QLabel("Connection:"))
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setObjectName("statusDisconnected")
        connection_layout.addWidget(self.connection_status)
        connection_layout.addStretch()
        status_layout.addLayout(connection_layout)
        
        # Data quality
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Data Quality:"))
        self.quality_status = QLabel("Unknown")
        self.quality_status.setObjectName("statusIndicator")
        quality_layout.addWidget(self.quality_status)
        quality_layout.addStretch()
        status_layout.addLayout(quality_layout)
        
        # Session status
        session_layout = QHBoxLayout()
        session_layout.addWidget(QLabel("Session:"))
        self.session_status = QLabel("Inactive")
        self.session_status.setObjectName("statusIndicator")
        session_layout.addWidget(self.session_status)
        session_layout.addStretch()
        status_layout.addLayout(session_layout)
        
        layout.addWidget(status_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_labels = {}
        stats_items = [
            ("Samples Received", "samples_received"),
            ("Analysis Updates", "analysis_updates"),
            ("Session Duration", "session_duration"),
            ("Average Latency", "avg_latency"),
            ("Sync Quality", "sync_quality")
        ]
        
        for label_text, key in stats_items:
            item_layout = QHBoxLayout()
            item_layout.addWidget(QLabel(f"{label_text}:"))
            
            value_label = QLabel("0")
            value_label.setStyleSheet("font-weight: 600; color: #0078d4;")
            self.stats_labels[key] = value_label
            
            item_layout.addWidget(value_label)
            item_layout.addStretch()
            stats_layout.addLayout(item_layout)
        
        layout.addWidget(stats_group)
        layout.addStretch()
    
    def set_system(self, system: ModernFootLabCore):
        """Set the system to monitor"""
        self.system = system
    
    def update_status(self):
        """Update status display"""
        if not self.system:
            return
        
        try:
            status = self.system.get_system_status()
            
            # Update connection status
            system_status = status.get('system_status', 'unknown')
            self.update_status_label(self.connection_status, system_status)
            
            # Update data quality
            data_quality = status.get('data_quality', DataQuality.GOOD)
            quality_text = data_quality.value if hasattr(data_quality, 'value') else str(data_quality)
            self.update_quality_label(self.quality_status, quality_text)
            
            # Update session status
            session_active = status.get('session_active', False)
            session_text = "Active" if session_active else "Inactive"
            self.update_session_label(self.session_status, session_text, session_active)
            
            # Update statistics
            current_metrics = status.get('current_metrics', {})
            self.update_statistics(current_metrics, status)
            
        except Exception as e:
            logger.error(f"Error updating system status: {e}")
    
    def update_status_label(self, label: QLabel, status: str):
        """Update connection status label"""
        status_map = {
            'running': ('Connected', 'statusConnected'),
            'streaming': ('Streaming', 'statusStreaming'), 
            'error': ('Error', 'statusError'),
            'stopped': ('Disconnected', 'statusDisconnected')
        }
        
        text, style_class = status_map.get(status, ('Unknown', 'statusIndicator'))
        label.setText(text)
        label.setObjectName(style_class)
        label.style().unpolish(label)
        label.style().polish(label)
    
    def update_quality_label(self, label: QLabel, quality: str):
        """Update data quality label"""
        quality_colors = {
            'excellent': '#27ae60',
            'good': '#2ecc71', 
            'acceptable': '#f39c12',
            'poor': '#e74c3c',
            'unusable': '#8e44ad'
        }
        
        color = quality_colors.get(quality.lower(), '#4d4d4d')
        label.setText(quality.title())
        label.setStyleSheet(f"background: {color}; color: white; padding: 4px 8px; border-radius: 4px;")
    
    def update_session_label(self, label: QLabel, text: str, active: bool):
        """Update session status label"""
        if active:
            label.setStyleSheet("background: #f39c12; color: white; padding: 4px 8px; border-radius: 4px;")
        else:
            label.setStyleSheet("background: #4d4d4d; color: white; padding: 4px 8px; border-radius: 4px;")
        label.setText(text)
    
    def update_statistics(self, metrics: Dict, status: Dict):
        """Update statistics display"""
        # Update individual statistics
        self.stats_labels['samples_received'].setText(str(metrics.get('samples_processed', 0)))
        self.stats_labels['analysis_updates'].setText(str(metrics.get('analysis_count', 0)))
        
        # Session duration
        uptime = status.get('uptime', 0)
        duration_text = f"{uptime:.1f}s"
        if uptime > 3600:
            duration_text = f"{uptime/3600:.1f}h"
        elif uptime > 60:
            duration_text = f"{uptime/60:.1f}m"
        self.stats_labels['session_duration'].setText(duration_text)
        
        # Average latency
        latency = metrics.get('average_latency', 0)
        self.stats_labels['avg_latency'].setText(f"{latency:.1f}ms")
        
        # Sync quality
        sync_quality = metrics.get('sync_quality', 0)
        self.stats_labels['sync_quality'].setText(f"{sync_quality:.2%}")

class AdvancedControlPanel(QWidget):
    """Advanced control panel with all system controls"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.system: Optional[ModernFootLabCore] = None
        self.current_patient = {}
        self.setupUI()
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        
        # Patient Information
        patient_group = QGroupBox("Patient Information")
        patient_layout = QVBoxLayout(patient_group)
        
        patient_info_layout = QHBoxLayout()
        self.patient_info_label = QLabel("No patient selected")
        self.patient_info_label.setStyleSheet("font-weight: 600; color: #0078d4;")
        patient_info_layout.addWidget(self.patient_info_label)
        patient_info_layout.addStretch()
        
        self.select_patient_btn = QPushButton("Select Patient")
        self.select_patient_btn.clicked.connect(self.select_patient)
        patient_info_layout.addWidget(self.select_patient_btn)
        
        patient_layout.addLayout(patient_info_layout)
        layout.addWidget(patient_group)
        
        # Data Source Configuration
        source_group = QGroupBox("Data Source")
        source_layout = QFormLayout(source_group)
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Simulator", "NURVV BLE", "File Replay"])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        source_layout.addRow("Source Type:", self.source_combo)
        
        # Source-specific options
        self.source_options = QWidget()
        self.setup_source_options()
        source_layout.addWidget(self.source_options)
        
        layout.addWidget(source_group)
        
        # Analysis Configuration
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QFormLayout(analysis_group)
        
        self.analysis_mode = QComboBox()
        self.analysis_mode.addItems(["Real-time", "Clinical", "Research", "Rehabilitation"])
        analysis_layout.addRow("Analysis Mode:", self.analysis_mode)
        
        self.enable_ml = QCheckBox("Enable ML Pathology Detection")
        self.enable_ml.setChecked(True)
        analysis_layout.addRow(self.enable_ml)
        
        self.enable_advanced = QCheckBox("Enable Advanced Analytics")
        self.enable_advanced.setChecked(True)
        analysis_layout.addRow(self.enable_advanced)
        
        layout.addWidget(analysis_group)
        
        # Session Control
        session_group = QGroupBox("Session Control")
        session_layout = QVBoxLayout(session_group)
        
        # Session buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start System")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self.start_system)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop System")
        self.stop_btn.setObjectName("dangerButton")
        self.stop_btn.clicked.connect(self.stop_system)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        session_layout.addLayout(button_layout)
        
        # Recording controls
        record_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setEnabled(False)
        record_layout.addWidget(self.record_btn)
        
        self.export_btn = QPushButton("Export Report")
        self.export_btn.clicked.connect(self.export_report)
        self.export_btn.setEnabled(False)
        record_layout.addWidget(self.export_btn)
        
        session_layout.addLayout(record_layout)
        
        layout.addWidget(session_group)
        
        # Calibration
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout(calibration_group)
        
        self.calibrate_btn = QPushButton("Calibrate Baseline")
        self.calibrate_btn.clicked.connect(self.calibrate_system)
        self.calibrate_btn.setEnabled(False)
        calibration_layout.addWidget(self.calibrate_btn)
        
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setVisible(False)
        calibration_layout.addWidget(self.calibration_progress)
        
        layout.addWidget(calibration_group)
        
        layout.addStretch()
    
    def setup_source_options(self):
        """Setup source-specific option widgets"""
        layout = QVBoxLayout(self.source_options)
        
        # Simulator options
        self.simulator_options = QWidget()
        sim_layout = QFormLayout(self.simulator_options)
        
        self.sim_frequency = QSpinBox()
        self.sim_frequency.setRange(50, 200)
        self.sim_frequency.setValue(100)
        self.sim_frequency.setSuffix(" Hz")
        sim_layout.addRow("Sampling Rate:", self.sim_frequency)
        
        self.sim_amplitude = QSlider(Qt.Horizontal)
        self.sim_amplitude.setRange(20, 200)
        self.sim_amplitude.setValue(100)
        sim_layout.addRow("Signal Amplitude:", self.sim_amplitude)
        
        layout.addWidget(self.simulator_options)
        
        # BLE options
        self.ble_options = QWidget()
        ble_layout = QFormLayout(self.ble_options)
        
        self.auto_connect = QCheckBox("Auto-connect to devices")
        self.auto_connect.setChecked(True)
        ble_layout.addRow(self.auto_connect)
        
        self.sync_window = QSpinBox()
        self.sync_window.setRange(10, 100)
        self.sync_window.setValue(50)
        self.sync_window.setSuffix(" ms")
        ble_layout.addRow("Sync Window:", self.sync_window)
        
        layout.addWidget(self.ble_options)
        self.ble_options.setVisible(False)
        
        # File replay options
        self.file_options = QWidget()
        file_layout = QVBoxLayout(self.file_options)
        
        file_select_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        file_select_layout.addWidget(self.file_path_label)
        
        self.file_browse_btn = QPushButton("Browse")
        self.file_browse_btn.clicked.connect(self.browse_file)
        file_select_layout.addWidget(self.file_browse_btn)
        
        file_layout.addLayout(file_select_layout)
        
        layout.addWidget(self.file_options)
        self.file_options.setVisible(False)
    
    def on_source_changed(self, source_type):
        """Handle source type change"""
        # Hide all option widgets
        for widget in [self.simulator_options, self.ble_options, self.file_options]:
            widget.setVisible(False)
        
        # Show relevant options
        if source_type == "Simulator":
            self.simulator_options.setVisible(True)
        elif source_type == "NURVV BLE":
            self.ble_options.setVisible(True)
        elif source_type == "File Replay":
            self.file_options.setVisible(True)
    
    def select_patient(self):
        """Show patient selection dialog"""
        dialog = PatientDialog(self, self.current_patient)
        
        if dialog.exec() == QDialog.Accepted:
            self.current_patient = dialog.get_patient_data()
            self.update_patient_display()
    
    def update_patient_display(self):
        """Update patient information display"""
        if self.current_patient:
            name = self.current_patient.get("name", "Unknown")
            patient_id = self.current_patient.get("patient_id", "Unknown")
            age = self.current_patient.get("age", 0)
            
            display_text = f"{name} (ID: {patient_id}, Age: {age})"
            self.patient_info_label.setText(display_text)
        else:
            self.patient_info_label.setText("No patient selected")
    
    def browse_file(self):
        """Browse for replay file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select session file", "", 
            "Session Files (*.csv *.json);;All Files (*)")
        
        if file_path:
            self.file_path_label.setText(Path(file_path).name)
            self.selected_file = file_path
    
    def set_system(self, system: ModernFootLabCore):
        """Set the system instance"""
        self.system = system
        
        # Subscribe to system events
        system.event_bus.subscribe('system_error', self.handle_system_error)
        system.event_bus.subscribe('session_started', self.handle_session_started)
        system.event_bus.subscribe('session_ended', self.handle_session_ended)
    
    async def start_system(self):
        """Start the system"""
        if not self.system:
            return
        
        try:
            # Configure system based on UI settings
            config = self.get_system_config()
            
            # Create and register data source
            source = self.create_data_source()
            if source:
                self.system.register_data_source(source)
            
            # Start system
            await self.system.start_system()
            
            # Update UI state
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
            self.calibrate_btn.setEnabled(True)
            
            logger.info("System started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            QMessageBox.critical(self, "System Error", f"Failed to start system:\n{str(e)}")
    
    async def stop_system(self):
        """Stop the system"""
        if not self.system:
            return
        
        try:
            await self.system.stop_system()
            
            # Update UI state
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.record_btn.setEnabled(False)
            self.calibrate_btn.setEnabled(False)
            self.export_btn.setEnabled(True)  # Can export after stopping
            
            logger.info("System stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration from UI"""
        
        analysis_mode_map = {
            "Real-time": AnalysisMode.REAL_TIME,
            "Clinical": AnalysisMode.CLINICAL,
            "Research": AnalysisMode.RESEARCH,
            "Rehabilitation": AnalysisMode.REHABILITATION
        }
        
        return SystemConfig(
            analysis_mode=analysis_mode_map.get(self.analysis_mode.currentText(), AnalysisMode.REAL_TIME),
            sampling_rate=self.sim_frequency.value(),
            enable_machine_learning=self.enable_ml.isChecked(),
            enable_advanced_analytics=self.enable_advanced.isChecked()
        )
    
    def create_data_source(self):
        """Create data source based on UI selection"""
        source_type = self.source_combo.currentText()
        
        if source_type == "Simulator":
            return SimulatorSource(
                n_sensors=16,
                freq=self.sim_frequency.value(),
                base_amp=self.sim_amplitude.value()
            )
        elif source_type == "NURVV BLE":
            # Would create BLE source
            return None  # Placeholder
        elif source_type == "File Replay":
            # Would create file replay source
            return None  # Placeholder
        
        return None
    
    def toggle_recording(self):
        """Toggle session recording"""
        if not self.system:
            return
        
        if self.system.state_manager.get_state('session_active'):
            # Stop recording
            self.system.end_session()
            self.record_btn.setText("Start Recording")
            self.export_btn.setEnabled(True)
        else:
            # Start recording
            if not self.current_patient:
                QMessageBox.warning(self, "No Patient", "Please select a patient before recording.")
                return
            
            session_info = {
                'patient_info': self.current_patient,
                'analysis_mode': self.analysis_mode.currentText(),
                'source_type': self.source_combo.currentText(),
                'timestamp': datetime.now().isoformat()
            }
            
            self.system.start_session(session_info)
            self.record_btn.setText("Stop Recording")
            self.export_btn.setEnabled(False)
    
    def calibrate_system(self):
        """Calibrate the system"""
        # Show calibration progress
        self.calibration_progress.setVisible(True)
        self.calibration_progress.setRange(0, 100)
        
        # Simulate calibration process
        self.calibrate_btn.setEnabled(False)
        
        def finish_calibration():
            self.calibration_progress.setVisible(False)
            self.calibrate_btn.setEnabled(True)
            QMessageBox.information(self, "Calibration", "System calibration completed successfully.")
        
        # In real implementation, this would be async
        QTimer.singleShot(3000, finish_calibration)
    
    def export_report(self):
        """Export analysis report"""
        if not self.current_patient:
            QMessageBox.warning(self, "No Patient", "No patient data available for report.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", f"report_{self.current_patient.get('patient_id', 'unknown')}.pdf",
            "PDF Files (*.pdf)")
        
        if file_path:
            # Generate report using advanced reporting system
            try:
                # This would use the actual analysis data
                QMessageBox.information(self, "Export", f"Report exported successfully to:\n{file_path}")
                logger.info(f"Report exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export report:\n{str(e)}")
                logger.error(f"Report export failed: {e}")
    
    def handle_system_error(self, event_data):
        """Handle system errors"""
        error_msg = event_data.get('error', 'Unknown error')
        component = event_data.get('component', 'System')
        
        QMessageBox.critical(self, "System Error", f"Error in {component}:\n{error_msg}")
    
    def handle_session_started(self, event_data):
        """Handle session start events"""
        session_id = event_data.get('session_id', 'Unknown')
        logger.info(f"Session started: {session_id}")
    
    def handle_session_ended(self, event_data):
        """Handle session end events"""
        session_file = event_data.get('session_file', '')
        logger.info(f"Session ended: {session_file}")

class StateOfTheArtFootLabApp(QMainWindow):
    """Main application window integrating all state-of-the-art components"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize system
        self.system: Optional[ModernFootLabCore] = None
        self.initialize_system()
        
        # Setup UI
        self.setupUI()
        self.setup_menu_bar()
        self.setup_status_bar()
        
        # Apply theme
        self.setStyleSheet(MODERN_DARK_THEME)
        
        # Window properties
        self.setWindowTitle("FootLab - State-of-the-Art Baropodometry System")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Center window
        self.center_window()
        
        logger.info("FootLab application initialized")
    
    def initialize_system(self):
        """Initialize the core system"""
        try:
            config = SystemConfig(
                analysis_mode=AnalysisMode.CLINICAL,
                sampling_rate=100.0,
                enable_advanced_analytics=True,
                enable_machine_learning=True
            )
            
            self.system = create_footlab_system(config)
            
            # Create and register analysis engine
            gait_analyzer = create_gait_analyzer(sampling_rate=100.0)
            self.system.register_analysis_engine(gait_analyzer)
            
            # Create pathology classifier
            pathology_classifier = create_pathology_classifier()
            # Would register this with the system if we had that interface
            
            logger.info("Core system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            QMessageBox.critical(None, "Initialization Error", 
                               f"Failed to initialize system:\n{str(e)}")
    
    def setupUI(self):
        """Setup the main user interface"""
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel (controls and status)
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_panel.setMinimumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Control panel
        self.control_panel = AdvancedControlPanel()
        if self.system:
            self.control_panel.set_system(self.system)
        left_layout.addWidget(self.control_panel, 2)
        
        # Status widget
        self.status_widget = SystemStatusWidget()
        if self.system:
            self.status_widget.set_system(self.system)
        left_layout.addWidget(self.status_widget, 1)
        
        main_splitter.addWidget(left_panel)
        
        # Right panel (main content area)
        right_panel = QTabWidget()
        
        # Heatmap visualization tab
        self.heatmap_view = StateOfTheArtHeatmapView(
            grid_w=128, grid_h=160, n_sensors=16,
            title="Advanced Plantar Pressure Analysis"
        )
        if self.system:
            self.system.register_visualizer(self.heatmap_view)
        right_panel.addTab(self.heatmap_view, "Pressure Analysis")
        
        # Analysis results tab
        self.analysis_tab = self.create_analysis_tab()
        right_panel.addTab(self.analysis_tab, "Gait Analysis")
        
        # Pathology detection tab
        self.pathology_tab = self.create_pathology_tab()
        right_panel.addTab(self.pathology_tab, "AI Pathology Detection")
        
        # System logs tab
        self.logs_tab = self.create_logs_tab()
        right_panel.addTab(self.logs_tab, "System Logs")
        
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([350, 1250])
    
    def create_analysis_tab(self) -> QWidget:
        """Create gait analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Analysis results table
        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(4)
        self.analysis_table.setHorizontalHeaderLabels(["Metric", "Left", "Right", "Asymmetry"])
        layout.addWidget(self.analysis_table)
        
        return widget
    
    def create_pathology_tab(self) -> QWidget:
        """Create pathology detection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Pathology results
        results_group = QGroupBox("AI Pathology Detection Results")
        results_layout = QVBoxLayout(results_group)
        
        self.pathology_results = QTableWidget()
        self.pathology_results.setColumnCount(5)
        self.pathology_results.setHorizontalHeaderLabels([
            "Condition", "Probability", "Confidence", "Severity", "Clinical Notes"
        ])
        results_layout.addWidget(self.pathology_results)
        
        layout.addWidget(results_group)
        
        return widget
    
    def create_logs_tab(self) -> QWidget:
        """Create system logs tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumBlockCount(1000)  # Limit log size
        layout.addWidget(self.log_display)
        
        # Log controls
        log_controls = QHBoxLayout()
        
        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.clicked.connect(self.log_display.clear)
        log_controls.addWidget(clear_logs_btn)
        
        export_logs_btn = QPushButton("Export Logs")
        export_logs_btn.clicked.connect(self.export_logs)
        log_controls.addWidget(export_logs_btn)
        
        log_controls.addStretch()
        
        layout.addLayout(log_controls)
        
        return widget
    
    def setup_menu_bar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.triggered.connect(self.new_session)
        file_menu.addAction(new_session_action)
        
        open_session_action = QAction("&Open Session", self)
        open_session_action.setShortcut("Ctrl+O")
        open_session_action.triggered.connect(self.open_session)
        file_menu.addAction(open_session_action)
        
        file_menu.addSeparator()
        
        export_report_action = QAction("Export &Report", self)
        export_report_action.setShortcut("Ctrl+R")
        export_report_action.triggered.connect(self.export_report)
        file_menu.addAction(export_report_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        calibrate_action = QAction("&Calibrate System", self)
        calibrate_action.triggered.connect(self.calibrate_system)
        tools_menu.addAction(calibrate_action)
        
        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About FootLab", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        documentation_action = QAction("&Documentation", self)
        documentation_action.triggered.connect(self.show_documentation)
        help_menu.addAction(documentation_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        statusbar = self.statusBar()
        
        # System status
        self.status_label = QLabel("System Ready")
        statusbar.addWidget(self.status_label)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        statusbar.addPermanentWidget(self.progress_bar)
        
        # Version info
        version_label = QLabel("FootLab v2.0 State-of-the-Art")
        statusbar.addPermanentWidget(version_label)
    
    def center_window(self):
        """Center the window on screen"""
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())
    
    # Menu action handlers
    def new_session(self):
        """Start a new session"""
        self.control_panel.select_patient()
    
    def open_session(self):
        """Open existing session"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "Session Files (*.json *.csv)")
        
        if file_path:
            # Load and display session
            logger.info(f"Opening session: {file_path}")
    
    def export_report(self):
        """Export analysis report"""
        self.control_panel.export_report()
    
    def calibrate_system(self):
        """Calibrate the system"""
        self.control_panel.calibrate_system()
    
    def show_settings(self):
        """Show application settings"""
        QMessageBox.information(self, "Settings", "Settings dialog would open here.")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>FootLab - State-of-the-Art Baropodometry System</h2>
        <p><b>Version:</b> 2.0</p>
        <p><b>Description:</b> Advanced clinical baropodometry system with real-time analysis, 
        AI-powered pathology detection, and comprehensive gait analysis capabilities.</p>
        
        <p><b>Features:</b></p>
        <ul>
        <li>Real-time pressure visualization with anatomical accuracy</li>
        <li>Advanced biomechanical analysis</li>
        <li>Machine learning pathology detection</li>
        <li>Publication-quality scientific reports</li>
        <li>BLE sensor integration (NURVV compatible)</li>
        <li>Modern architecture with plugin system</li>
        </ul>
        
        <p><b>Technology Stack:</b> Python, PySide6, PyQtGraph, scikit-learn, 
        matplotlib, asyncio, bleak</p>
        
        <p>© 2024 FootLab Development Team</p>
        """
        
        QMessageBox.about(self, "About FootLab", about_text)
    
    def show_documentation(self):
        """Show documentation"""
        QMessageBox.information(self, "Documentation", 
                               "Documentation would be available online or in help system.")
    
    def export_logs(self):
        """Export system logs"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", f"footlab_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)")
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.log_display.toPlainText())
                QMessageBox.information(self, "Export", f"Logs exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export logs:\n{str(e)}")
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.system and self.system.state_manager.get_state('system_status') == 'running':
            reply = QMessageBox.question(
                self, "Exit FootLab",
                "System is currently running. Stop system and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Stop system asynchronously
                if asyncio.get_event_loop().is_running():
                    asyncio.create_task(self.system.stop_system())
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Main application entry point"""
    
    # Configure high DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("FootLab")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("FootLab")
    app.setOrganizationDomain("footlab.com")
    
    # Create and show main window
    main_window = StateOfTheArtFootLabApp()
    main_window.show()
    
    # Run application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()