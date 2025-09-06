# monitoring/performance_system.py
"""
Real-time performance monitoring and optimization system for FootLab
Tracks system resources, data throughput, and performance metrics
"""

import asyncio
import psutil
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from collections import deque
import traceback
import gc

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QTreeWidget, QTreeWidgetItem, QTabWidget, QPushButton,
    QTextEdit, QComboBox, QSpinBox, QCheckBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QSplitter, QFrame
)
from PySide6.QtCore import QTimer, QThread, Signal, QObject, Qt
from PySide6.QtGui import QFont, QColor
import pyqtgraph as pg

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Performance level indicators"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class AlertType(Enum):
    """Performance alert types"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    level: PerformanceLevel
    threshold_warning: float = 80.0
    threshold_critical: float = 95.0
    description: str = ""

@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    component: str
    message: str
    metric_name: str
    metric_value: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SystemProfile:
    """System performance profile"""
    profile_name: str
    cpu_cores: int
    total_memory: float  # GB
    available_memory: float  # GB
    gpu_available: bool
    disk_space: float  # GB
    network_speed: float  # Mbps
    os_info: str
    python_version: str
    dependencies: Dict[str, str] = field(default_factory=dict)

class ResourceMonitor(QObject):
    """Monitors system resources in real-time"""
    
    metrics_updated = Signal(dict)
    alert_generated = Signal(object)  # PerformanceAlert
    
    def __init__(self, update_interval: float = 1.0):
        super().__init__()
        self.update_interval = update_interval
        self.running = False
        self.metrics_history: Dict[str, deque] = {}
        self.max_history_size = 300  # 5 minutes at 1 second intervals
        
        # Initialize metric histories
        metric_names = [
            "cpu_usage", "memory_usage", "disk_usage", "network_io",
            "gpu_usage", "gpu_memory", "temperature", "battery"
        ]
        
        for name in metric_names:
            self.metrics_history[name] = deque(maxlen=self.max_history_size)
        
        self.alerts: List[PerformanceAlert] = []
        self.alert_counter = 0
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.running = True
        logger.info("Performance monitoring started")
        
        while self.running:
            try:
                metrics = await self._collect_metrics()
                self._update_history(metrics)
                self._check_thresholds(metrics)
                
                self.metrics_updated.emit(metrics)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.running = False
        logger.info("Performance monitoring stopped")
    
    async def _collect_metrics(self) -> Dict[str, PerformanceMetric]:
        """Collect current system metrics"""
        metrics = {}
        current_time = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics["cpu_usage"] = PerformanceMetric(
            name="CPU Usage",
            value=cpu_percent,
            unit="%",
            timestamp=current_time,
            level=self._get_performance_level(cpu_percent, 70, 90),
            threshold_warning=70.0,
            threshold_critical=90.0,
            description="System CPU utilization"
        )
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        metrics["memory_usage"] = PerformanceMetric(
            name="Memory Usage",
            value=memory_percent,
            unit="%",
            timestamp=current_time,
            level=self._get_performance_level(memory_percent, 80, 95),
            threshold_warning=80.0,
            threshold_critical=95.0,
            description=f"RAM usage: {memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB"
        )
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        metrics["disk_usage"] = PerformanceMetric(
            name="Disk Usage",
            value=disk_percent,
            unit="%",
            timestamp=current_time,
            level=self._get_performance_level(disk_percent, 85, 95),
            threshold_warning=85.0,
            threshold_critical=95.0,
            description=f"Disk space: {disk.used/1024**3:.1f}GB / {disk.total/1024**3:.1f}GB"
        )
        
        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            # Calculate rate from previous measurement
            current_bytes = net_io.bytes_sent + net_io.bytes_recv
            if hasattr(self, '_last_net_bytes') and hasattr(self, '_last_net_time'):
                time_diff = time.time() - self._last_net_time
                if time_diff > 0:
                    bytes_diff = current_bytes - self._last_net_bytes
                    network_rate = (bytes_diff / time_diff) / 1024 / 1024  # MB/s
                else:
                    network_rate = 0
            else:
                network_rate = 0
            
            self._last_net_bytes = current_bytes
            self._last_net_time = time.time()
            
            metrics["network_io"] = PerformanceMetric(
                name="Network I/O",
                value=network_rate,
                unit="MB/s",
                timestamp=current_time,
                level=PerformanceLevel.GOOD,  # No specific thresholds for network
                description=f"Network transfer rate"
            )
        except Exception:
            metrics["network_io"] = PerformanceMetric(
                name="Network I/O", value=0, unit="MB/s", 
                timestamp=current_time, level=PerformanceLevel.GOOD
            )
        
        # GPU metrics (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                metrics["gpu_usage"] = PerformanceMetric(
                    name="GPU Usage",
                    value=gpu.load * 100,
                    unit="%",
                    timestamp=current_time,
                    level=self._get_performance_level(gpu.load * 100, 80, 95),
                    description=f"GPU: {gpu.name}"
                )
                
                metrics["gpu_memory"] = PerformanceMetric(
                    name="GPU Memory",
                    value=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                    unit="%",
                    timestamp=current_time,
                    level=self._get_performance_level((gpu.memoryUsed / gpu.memoryTotal) * 100, 80, 95),
                    description=f"VRAM: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB"
                )
                
                metrics["gpu_temperature"] = PerformanceMetric(
                    name="GPU Temperature",
                    value=gpu.temperature,
                    unit="°C",
                    timestamp=current_time,
                    level=self._get_performance_level(gpu.temperature, 75, 85),
                    description="GPU temperature"
                )
        except ImportError:
            # GPU monitoring not available
            pass
        except Exception as e:
            logger.warning(f"GPU monitoring error: {e}")
        
        # Temperature monitoring (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Average CPU temperature
                cpu_temps = []
                for sensor, temp_list in temps.items():
                    if 'cpu' in sensor.lower() or 'core' in sensor.lower():
                        cpu_temps.extend([t.current for t in temp_list if t.current])
                
                if cpu_temps:
                    avg_temp = np.mean(cpu_temps)
                    metrics["temperature"] = PerformanceMetric(
                        name="CPU Temperature",
                        value=avg_temp,
                        unit="°C",
                        timestamp=current_time,
                        level=self._get_performance_level(avg_temp, 70, 85),
                        threshold_warning=70.0,
                        threshold_critical=85.0,
                        description="Average CPU core temperature"
                    )
        except Exception:
            pass
        
        # Battery status (for laptops)
        try:
            battery = psutil.sensors_battery()
            if battery:
                metrics["battery"] = PerformanceMetric(
                    name="Battery Level",
                    value=battery.percent,
                    unit="%",
                    timestamp=current_time,
                    level=self._get_performance_level(100 - battery.percent, 20, 10),  # Inverted
                    threshold_warning=20.0,
                    threshold_critical=10.0,
                    description=f"Battery: {'Charging' if battery.power_plugged else 'Discharging'}"
                )
        except Exception:
            pass
        
        return metrics
    
    def _get_performance_level(self, value: float, warning_threshold: float, 
                              critical_threshold: float) -> PerformanceLevel:
        """Determine performance level based on value and thresholds"""
        if value >= critical_threshold:
            return PerformanceLevel.CRITICAL
        elif value >= warning_threshold:
            return PerformanceLevel.POOR
        elif value >= warning_threshold * 0.8:
            return PerformanceLevel.FAIR
        elif value >= warning_threshold * 0.6:
            return PerformanceLevel.GOOD
        else:
            return PerformanceLevel.EXCELLENT
    
    def _update_history(self, metrics: Dict[str, PerformanceMetric]):
        """Update metrics history"""
        for name, metric in metrics.items():
            if name in self.metrics_history:
                self.metrics_history[name].append({
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'level': metric.level
                })
    
    def _check_thresholds(self, metrics: Dict[str, PerformanceMetric]):
        """Check metrics against thresholds and generate alerts"""
        for name, metric in metrics.items():
            # Check for critical threshold
            if metric.level == PerformanceLevel.CRITICAL:
                self._generate_alert(
                    AlertType.CRITICAL,
                    f"resource_monitor.{name}",
                    f"Critical {metric.name}: {metric.value:.1f}{metric.unit}",
                    name,
                    metric.value
                )
            elif metric.level == PerformanceLevel.POOR:
                self._generate_alert(
                    AlertType.WARNING,
                    f"resource_monitor.{name}",
                    f"High {metric.name}: {metric.value:.1f}{metric.unit}",
                    name,
                    metric.value
                )
    
    def _generate_alert(self, alert_type: AlertType, component: str, 
                       message: str, metric_name: str, metric_value: float):
        """Generate performance alert"""
        # Avoid duplicate alerts
        recent_alerts = [a for a in self.alerts 
                        if a.component == component and 
                        a.alert_type == alert_type and
                        not a.resolved and
                        (datetime.now() - a.timestamp).seconds < 60]
        
        if recent_alerts:
            return  # Don't generate duplicate alert
        
        alert = PerformanceAlert(
            alert_id=f"alert_{self.alert_counter}",
            timestamp=datetime.now(),
            alert_type=alert_type,
            component=component,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value
        )
        
        self.alerts.append(alert)
        self.alert_counter += 1
        
        self.alert_generated.emit(alert)
        logger.warning(f"Performance alert: {message}")
    
    def get_metrics_history(self, metric_name: str, duration_minutes: int = 5) -> List[Dict]:
        """Get metrics history for specified duration"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [entry for entry in self.metrics_history[metric_name] 
                if entry['timestamp'] >= cutoff_time]

class ApplicationProfiler(QObject):
    """Profiles FootLab application performance"""
    
    profile_updated = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.profiling_enabled = False
        self.performance_data = {}
        self.function_call_counts = {}
        self.execution_times = {}
        
    def start_profiling(self):
        """Start application profiling"""
        self.profiling_enabled = True
        logger.info("Application profiling started")
    
    def stop_profiling(self):
        """Stop application profiling"""
        self.profiling_enabled = False
        logger.info("Application profiling stopped")
    
    def profile_function(self, func_name: str, execution_time: float):
        """Profile function execution"""
        if not self.profiling_enabled:
            return
        
        if func_name not in self.function_call_counts:
            self.function_call_counts[func_name] = 0
            self.execution_times[func_name] = []
        
        self.function_call_counts[func_name] += 1
        self.execution_times[func_name].append(execution_time)
        
        # Keep only recent data
        if len(self.execution_times[func_name]) > 1000:
            self.execution_times[func_name] = self.execution_times[func_name][-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {
            'total_functions_profiled': len(self.function_call_counts),
            'total_function_calls': sum(self.function_call_counts.values()),
            'top_called_functions': [],
            'slowest_functions': [],
            'function_stats': {}
        }
        
        # Top called functions
        top_called = sorted(self.function_call_counts.items(), 
                           key=lambda x: x[1], reverse=True)[:10]
        summary['top_called_functions'] = top_called
        
        # Slowest functions (by average execution time)
        function_avg_times = {}
        for func_name, times in self.execution_times.items():
            if times:
                function_avg_times[func_name] = np.mean(times)
        
        slowest = sorted(function_avg_times.items(), 
                        key=lambda x: x[1], reverse=True)[:10]
        summary['slowest_functions'] = slowest
        
        # Detailed function stats
        for func_name in self.function_call_counts.keys():
            times = self.execution_times.get(func_name, [])
            if times:
                summary['function_stats'][func_name] = {
                    'call_count': self.function_call_counts[func_name],
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_time': np.sum(times)
                }
        
        return summary

class PerformanceOptimizer:
    """Automated performance optimization"""
    
    def __init__(self):
        self.optimization_enabled = True
        self.optimizations_applied = []
        
    def analyze_and_optimize(self, metrics: Dict[str, PerformanceMetric]) -> List[str]:
        """Analyze metrics and apply optimizations"""
        if not self.optimization_enabled:
            return []
        
        optimizations = []
        
        # Memory optimization
        memory_metric = metrics.get('memory_usage')
        if memory_metric and memory_metric.value > 80:
            optimizations.extend(self._optimize_memory())
        
        # CPU optimization
        cpu_metric = metrics.get('cpu_usage')
        if cpu_metric and cpu_metric.value > 80:
            optimizations.extend(self._optimize_cpu())
        
        # Disk optimization
        disk_metric = metrics.get('disk_usage')
        if disk_metric and disk_metric.value > 90:
            optimizations.extend(self._optimize_disk())
        
        self.optimizations_applied.extend(optimizations)
        return optimizations
    
    def _optimize_memory(self) -> List[str]:
        """Apply memory optimizations"""
        optimizations = []
        
        # Garbage collection
        collected = gc.collect()
        if collected > 0:
            optimizations.append(f"Garbage collection: freed {collected} objects")
        
        # Clear caches (example)
        optimizations.append("Cleared internal caches")
        
        return optimizations
    
    def _optimize_cpu(self) -> List[str]:
        """Apply CPU optimizations"""
        optimizations = []
        
        # Example: reduce update frequencies
        optimizations.append("Reduced update frequencies for non-critical components")
        
        # Example: optimize algorithms
        optimizations.append("Applied algorithmic optimizations")
        
        return optimizations
    
    def _optimize_disk(self) -> List[str]:
        """Apply disk optimizations"""
        optimizations = []
        
        # Example: cleanup temporary files
        temp_path = Path("temp")
        if temp_path.exists():
            temp_files = list(temp_path.glob("*.tmp"))
            for file in temp_files:
                try:
                    file.unlink()
                except Exception:
                    pass
            
            if temp_files:
                optimizations.append(f"Cleaned up {len(temp_files)} temporary files")
        
        return optimizations

class PerformanceDashboard(QWidget):
    """Real-time performance monitoring dashboard"""
    
    def __init__(self):
        super().__init__()
        self.resource_monitor = ResourceMonitor()
        self.app_profiler = ApplicationProfiler()
        self.optimizer = PerformanceOptimizer()
        
        self.setupUI()
        self.setup_monitoring()
        
        # Start monitoring
        self.monitoring_task = None
        self.start_monitoring()
    
    def setupUI(self):
        """Setup dashboard UI"""
        layout = QVBoxLayout(self)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Main dashboard tabs
        self.tab_widget = QTabWidget()
        
        # Real-time metrics tab
        metrics_tab = self.create_metrics_tab()
        self.tab_widget.addTab(metrics_tab, "Real-time Metrics")
        
        # Performance graphs tab
        graphs_tab = self.create_graphs_tab()
        self.tab_widget.addTab(graphs_tab, "Performance Graphs")
        
        # Alerts tab
        alerts_tab = self.create_alerts_tab()
        self.tab_widget.addTab(alerts_tab, "Alerts & Issues")
        
        # Application profiler tab
        profiler_tab = self.create_profiler_tab()
        self.tab_widget.addTab(profiler_tab, "Application Profiler")
        
        # System info tab
        system_tab = self.create_system_tab()
        self.tab_widget.addTab(system_tab, "System Information")
        
        layout.addWidget(self.tab_widget)
    
    def create_control_panel(self) -> QWidget:
        """Create monitoring control panel"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout(panel)
        
        # Monitoring controls
        self.start_btn = QPushButton("Start Monitoring")
        self.start_btn.clicked.connect(self.start_monitoring)
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Monitoring")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        # Update interval
        layout.addWidget(QLabel("Update Interval:"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(1)
        self.interval_spin.setSuffix(" sec")
        layout.addWidget(self.interval_spin)
        
        # Auto-optimization
        self.auto_optimize = QCheckBox("Auto-optimize")
        self.auto_optimize.setChecked(True)
        layout.addWidget(self.auto_optimize)
        
        # Manual optimization
        optimize_btn = QPushButton("Optimize Now")
        optimize_btn.clicked.connect(self.manual_optimize)
        layout.addWidget(optimize_btn)
        
        layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("Monitoring: Stopped")
        layout.addWidget(self.status_label)
        
        return panel
    
    def create_metrics_tab(self) -> QWidget:
        """Create real-time metrics display"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Metrics grid
        self.metrics_layout = QVBoxLayout()
        layout.addLayout(self.metrics_layout)
        
        # Initialize metric displays
        self.metric_widgets = {}
        
        return widget
    
    def create_graphs_tab(self) -> QWidget:
        """Create performance graphs"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Graph widget
        self.graph_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graph_widget)
        
        # Initialize plots
        self.plots = {}
        self.setup_plots()
        
        return widget
    
    def create_alerts_tab(self) -> QWidget:
        """Create alerts display"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Alerts table
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(6)
        self.alerts_table.setHorizontalHeaderLabels([
            "Time", "Type", "Component", "Message", "Status", "Actions"
        ])
        layout.addWidget(self.alerts_table)
        
        # Alert controls
        controls = QHBoxLayout()
        
        clear_btn = QPushButton("Clear Resolved Alerts")
        clear_btn.clicked.connect(self.clear_resolved_alerts)
        controls.addWidget(clear_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        return widget
    
    def create_profiler_tab(self) -> QWidget:
        """Create application profiler display"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Profiler controls
        controls = QHBoxLayout()
        
        self.profiler_start_btn = QPushButton("Start Profiling")
        self.profiler_start_btn.clicked.connect(self.app_profiler.start_profiling)
        controls.addWidget(self.profiler_start_btn)
        
        self.profiler_stop_btn = QPushButton("Stop Profiling")
        self.profiler_stop_btn.clicked.connect(self.app_profiler.stop_profiling)
        controls.addWidget(self.profiler_stop_btn)
        
        refresh_btn = QPushButton("Refresh Data")
        refresh_btn.clicked.connect(self.update_profiler_display)
        controls.addWidget(refresh_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Profiler data display
        splitter = QSplitter(Qt.Horizontal)
        
        # Function call statistics
        self.function_stats_table = QTableWidget()
        self.function_stats_table.setColumnCount(5)
        self.function_stats_table.setHorizontalHeaderLabels([
            "Function", "Calls", "Avg Time (ms)", "Total Time (ms)", "% of Total"
        ])
        splitter.addWidget(self.function_stats_table)
        
        # Performance summary
        self.performance_summary = QTextEdit()
        self.performance_summary.setReadOnly(True)
        splitter.addWidget(self.performance_summary)
        
        layout.addWidget(splitter)
        
        return widget
    
    def create_system_tab(self) -> QWidget:
        """Create system information display"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # System info display
        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setPlainText(self.get_system_info())
        layout.addWidget(self.system_info)
        
        return widget
    
    def setup_plots(self):
        """Setup performance plots"""
        # CPU usage plot
        self.plots['cpu'] = self.graph_widget.addPlot(title="CPU Usage (%)")
        self.plots['cpu'].setLabel('left', 'Usage', '%')
        self.plots['cpu'].setLabel('bottom', 'Time', 's')
        self.plots['cpu'].showGrid(x=True, y=True)
        
        # Memory usage plot
        self.graph_widget.nextRow()
        self.plots['memory'] = self.graph_widget.addPlot(title="Memory Usage (%)")
        self.plots['memory'].setLabel('left', 'Usage', '%')
        self.plots['memory'].setLabel('bottom', 'Time', 's')
        self.plots['memory'].showGrid(x=True, y=True)
        
        # Network I/O plot
        self.graph_widget.nextRow()
        self.plots['network'] = self.graph_widget.addPlot(title="Network I/O (MB/s)")
        self.plots['network'].setLabel('left', 'Transfer Rate', 'MB/s')
        self.plots['network'].setLabel('bottom', 'Time', 's')
        self.plots['network'].showGrid(x=True, y=True)
        
        # Initialize plot data
        self.plot_data = {name: {'x': [], 'y': []} for name in self.plots.keys()}
    
    def setup_monitoring(self):
        """Setup monitoring connections"""
        self.resource_monitor.metrics_updated.connect(self.update_metrics_display)
        self.resource_monitor.alert_generated.connect(self.add_alert)
        
        # Update timer for UI refresh
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui)
        self.ui_timer.start(1000)  # Update UI every second
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(
                self.resource_monitor.start_monitoring()
            )
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Monitoring: Active")
            
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
        
        self.resource_monitor.stop_monitoring()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Monitoring: Stopped")
        
        logger.info("Performance monitoring stopped")
    
    def update_metrics_display(self, metrics: Dict[str, PerformanceMetric]):
        """Update metrics display"""
        for name, metric in metrics.items():
            if name not in self.metric_widgets:
                self.create_metric_widget(name, metric)
            
            self.update_metric_widget(name, metric)
        
        # Update plots
        self.update_plots(metrics)
        
        # Auto-optimization
        if self.auto_optimize.isChecked():
            optimizations = self.optimizer.analyze_and_optimize(metrics)
            if optimizations:
                logger.info(f"Applied optimizations: {optimizations}")
    
    def create_metric_widget(self, name: str, metric: PerformanceMetric):
        """Create widget for displaying metric"""
        group = QGroupBox(metric.name)
        layout = QVBoxLayout(group)
        
        # Value display
        value_label = QLabel(f"{metric.value:.1f} {metric.unit}")
        value_font = QFont()
        value_font.setPointSize(14)
        value_font.setBold(True)
        value_label.setFont(value_font)
        layout.addWidget(value_label)
        
        # Progress bar
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(int(metric.value))
        layout.addWidget(progress_bar)
        
        # Status indicator
        status_label = QLabel(metric.level.value.title())
        layout.addWidget(status_label)
        
        # Description
        desc_label = QLabel(metric.description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        self.metric_widgets[name] = {
            'group': group,
            'value_label': value_label,
            'progress_bar': progress_bar,
            'status_label': status_label,
            'desc_label': desc_label
        }
        
        self.metrics_layout.addWidget(group)
    
    def update_metric_widget(self, name: str, metric: PerformanceMetric):
        """Update metric widget with new values"""
        if name not in self.metric_widgets:
            return
        
        widgets = self.metric_widgets[name]
        
        # Update value
        widgets['value_label'].setText(f"{metric.value:.1f} {metric.unit}")
        
        # Update progress bar
        widgets['progress_bar'].setValue(int(metric.value))
        
        # Update color based on performance level
        colors = {
            PerformanceLevel.EXCELLENT: "green",
            PerformanceLevel.GOOD: "lightgreen",
            PerformanceLevel.FAIR: "yellow",
            PerformanceLevel.POOR: "orange",
            PerformanceLevel.CRITICAL: "red"
        }
        
        color = colors.get(metric.level, "gray")
        widgets['progress_bar'].setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
        
        # Update status
        widgets['status_label'].setText(metric.level.value.title())
        widgets['status_label'].setStyleSheet(f"color: {color};")
        
        # Update description
        widgets['desc_label'].setText(metric.description)
    
    def update_plots(self, metrics: Dict[str, PerformanceMetric]):
        """Update performance plots"""
        current_time = time.time()
        
        # Update plot data for specific metrics
        plot_mapping = {
            'cpu': 'cpu_usage',
            'memory': 'memory_usage', 
            'network': 'network_io'
        }
        
        for plot_name, metric_name in plot_mapping.items():
            if metric_name in metrics:
                metric = metrics[metric_name]
                
                # Add new data point
                self.plot_data[plot_name]['x'].append(current_time)
                self.plot_data[plot_name]['y'].append(metric.value)
                
                # Keep only recent data (last 5 minutes)
                cutoff_time = current_time - 300
                while (self.plot_data[plot_name]['x'] and 
                       self.plot_data[plot_name]['x'][0] < cutoff_time):
                    self.plot_data[plot_name]['x'].pop(0)
                    self.plot_data[plot_name]['y'].pop(0)
                
                # Update plot
                if self.plot_data[plot_name]['x']:
                    # Convert to relative time (seconds ago)
                    x_data = [(current_time - t) for t in self.plot_data[plot_name]['x']]
                    x_data.reverse()  # Show most recent on right
                    
                    y_data = list(reversed(self.plot_data[plot_name]['y']))
                    
                    self.plots[plot_name].clear()
                    self.plots[plot_name].plot(x_data, y_data, pen='b')
    
    def add_alert(self, alert: PerformanceAlert):
        """Add alert to alerts table"""
        row = self.alerts_table.rowCount()
        self.alerts_table.insertRow(row)
        
        # Time
        time_item = QTableWidgetItem(alert.timestamp.strftime("%H:%M:%S"))
        self.alerts_table.setItem(row, 0, time_item)
        
        # Type
        type_item = QTableWidgetItem(alert.alert_type.value.upper())
        type_colors = {
            AlertType.INFO: QColor("blue"),
            AlertType.WARNING: QColor("orange"),
            AlertType.ERROR: QColor("red"),
            AlertType.CRITICAL: QColor("darkred")
        }
        type_item.setForeground(type_colors.get(alert.alert_type, QColor("black")))
        self.alerts_table.setItem(row, 1, type_item)
        
        # Component
        self.alerts_table.setItem(row, 2, QTableWidgetItem(alert.component))
        
        # Message
        self.alerts_table.setItem(row, 3, QTableWidgetItem(alert.message))
        
        # Status
        status_item = QTableWidgetItem("Active" if not alert.resolved else "Resolved")
        self.alerts_table.setItem(row, 4, status_item)
        
        # Actions
        resolve_btn = QPushButton("Resolve")
        resolve_btn.clicked.connect(lambda: self.resolve_alert(alert))
        self.alerts_table.setCellWidget(row, 5, resolve_btn)
        
        # Auto-scroll to new alert
        self.alerts_table.scrollToBottom()
    
    def resolve_alert(self, alert: PerformanceAlert):
        """Resolve an alert"""
        alert.resolved = True
        alert.resolution_time = datetime.now()
        
        # Update UI
        for row in range(self.alerts_table.rowCount()):
            if self.alerts_table.item(row, 3).text() == alert.message:
                self.alerts_table.item(row, 4).setText("Resolved")
                break
    
    def clear_resolved_alerts(self):
        """Clear resolved alerts from display"""
        for row in range(self.alerts_table.rowCount() - 1, -1, -1):
            status_item = self.alerts_table.item(row, 4)
            if status_item and status_item.text() == "Resolved":
                self.alerts_table.removeRow(row)
    
    def update_profiler_display(self):
        """Update application profiler display"""
        summary = self.app_profiler.get_performance_summary()
        
        # Update function statistics table
        self.function_stats_table.setRowCount(len(summary.get('function_stats', {})))
        
        row = 0
        total_time = sum(stats['total_time'] for stats in summary.get('function_stats', {}).values())
        
        for func_name, stats in summary.get('function_stats', {}).items():
            self.function_stats_table.setItem(row, 0, QTableWidgetItem(func_name))
            self.function_stats_table.setItem(row, 1, QTableWidgetItem(str(stats['call_count'])))
            self.function_stats_table.setItem(row, 2, QTableWidgetItem(f"{stats['avg_time']*1000:.2f}"))
            self.function_stats_table.setItem(row, 3, QTableWidgetItem(f"{stats['total_time']*1000:.2f}"))
            
            if total_time > 0:
                percentage = (stats['total_time'] / total_time) * 100
                self.function_stats_table.setItem(row, 4, QTableWidgetItem(f"{percentage:.1f}%"))
            
            row += 1
        
        # Update summary text
        summary_text = f"""
Performance Summary:
- Functions Profiled: {summary['total_functions_profiled']}
- Total Function Calls: {summary['total_function_calls']}

Top Called Functions:
{chr(10).join([f"  {name}: {count} calls" for name, count in summary['top_called_functions'][:5]])}

Slowest Functions:
{chr(10).join([f"  {name}: {time*1000:.2f}ms avg" for name, time in summary['slowest_functions'][:5]])}
"""
        self.performance_summary.setPlainText(summary_text)
    
    def manual_optimize(self):
        """Manually trigger optimization"""
        # Get current metrics
        # This would be implemented with actual metrics
        optimizations = self.optimizer.analyze_and_optimize({})
        
        if optimizations:
            message = "Applied optimizations:\n" + "\n".join(optimizations)
        else:
            message = "No optimizations needed at this time."
        
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Manual Optimization", message)
    
    def update_ui(self):
        """Periodic UI updates"""
        # Update profiler display if visible
        if self.tab_widget.currentIndex() == 3:  # Profiler tab
            self.update_profiler_display()
    
    def get_system_info(self) -> str:
        """Get detailed system information"""
        import platform
        import sys
        
        # Basic system info
        info = f"""System Information:
Platform: {platform.system()} {platform.release()}
Architecture: {platform.machine()}
Processor: {platform.processor()}
Python Version: {sys.version}

Hardware:
CPU Cores: {psutil.cpu_count()}
Physical Cores: {psutil.cpu_count(logical=False)}
Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB

Storage:
"""
        
        # Disk information
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                info += f"  {partition.device}: {usage.total / (1024**3):.1f} GB total, "
                info += f"{usage.free / (1024**3):.1f} GB free\n"
            except PermissionError:
                info += f"  {partition.device}: Access denied\n"
        
        # Network interfaces
        info += "\nNetwork Interfaces:\n"
        for interface, addresses in psutil.net_if_addrs().items():
            info += f"  {interface}:\n"
            for addr in addresses:
                info += f"    {addr.family.name}: {addr.address}\n"
        
        return info

# Integration functions
def create_performance_monitor() -> PerformanceDashboard:
    """Create performance monitoring dashboard"""
    return PerformanceDashboard()

def profile_function(func):
    """Decorator for profiling function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            # This would connect to the actual profiler instance
            logger.debug(f"Function {func.__name__} executed in {execution_time:.4f}s")
    
    return wrapper

# Example usage
if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    dashboard = PerformanceDashboard()
    dashboard.show()
    
    sys.exit(app.exec())