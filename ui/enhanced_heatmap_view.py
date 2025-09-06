# ui/enhanced_heatmap_view.py
import os
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QFrame, QGridLayout, QSlider, QComboBox, 
                               QCheckBox, QPushButton, QGroupBox)
from PySide6.QtCore import QRectF, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from scipy.interpolate import Rbf, griddata
from scipy.ndimage import gaussian_filter, uniform_filter
from core.enhanced_foot_shape import (
    foot_outline_points_enhanced, create_pressure_zones_mask,
    adaptive_sensor_placement, polygon_to_mask
)

class AdvancedColorMaps:
    """State-of-the-art color maps for clinical analysis"""
    
    @staticmethod
    def clinical_pressure():
        """Clinical pressure colormap: blue->green->yellow->red"""
        colors = [
            (0.0, [0, 0, 64]),      # Dark blue (no pressure)
            (0.1, [0, 64, 128]),    # Blue
            (0.3, [0, 128, 64]),    # Teal
            (0.5, [128, 128, 0]),   # Yellow
            (0.7, [255, 128, 0]),   # Orange
            (1.0, [255, 0, 0])      # Red (high pressure)
        ]
        return pg.ColorMap([c[0] for c in colors], [c[1] for c in colors])
    
    @staticmethod
    def thermal_enhanced():
        """Enhanced thermal colormap with better contrast"""
        colors = [
            (0.0, [0, 0, 0]),       # Black
            (0.2, [64, 0, 128]),    # Purple
            (0.4, [128, 0, 255]),   # Magenta
            (0.6, [255, 64, 64]),   # Red
            (0.8, [255, 255, 0]),   # Yellow
            (1.0, [255, 255, 255])  # White
        ]
        return pg.ColorMap([c[0] for c in colors], [c[1] for c in colors])
    
    @staticmethod
    def rainbow_medical():
        """Medical rainbow with enhanced visibility"""
        colors = [
            (0.0, [0, 0, 128]),     # Navy
            (0.2, [0, 128, 255]),   # Blue
            (0.4, [0, 255, 128]),   # Green
            (0.6, [255, 255, 0]),   # Yellow
            (0.8, [255, 128, 0]),   # Orange
            (1.0, [255, 0, 0])      # Red
        ]
        return pg.ColorMap([c[0] for c in colors], [c[1] for c in colors])

class AdvancedInterpolation:
    """Advanced interpolation methods for pressure data"""
    
    @staticmethod
    def radial_basis_function(sensor_coords, pressures, grid_coords, 
                            function='thin_plate', smooth=0.1):
        """RBF interpolation for smooth, realistic pressure distribution"""
        if len(pressures) == 0 or np.sum(pressures) == 0:
            return np.zeros(grid_coords.shape[0])
        
        # Remove zero pressure sensors for better interpolation
        active_mask = pressures > 0.01
        if np.sum(active_mask) < 3:  # Need at least 3 points
            # Fallback to simple inverse distance weighting
            return AdvancedInterpolation.inverse_distance_weighting(
                sensor_coords, pressures, grid_coords)
        
        active_coords = sensor_coords[active_mask]
        active_pressures = pressures[active_mask]
        
        try:
            rbf = Rbf(active_coords[:, 0], active_coords[:, 1], active_pressures, 
                     function=function, smooth=smooth)
            interpolated = rbf(grid_coords[:, 0], grid_coords[:, 1])
            return np.maximum(interpolated, 0)  # Ensure non-negative
        except Exception:
            # Fallback to IDW if RBF fails
            return AdvancedInterpolation.inverse_distance_weighting(
                sensor_coords, pressures, grid_coords)
    
    @staticmethod
    def inverse_distance_weighting(sensor_coords, pressures, grid_coords, 
                                 power=2.0, radius=None):
        """Enhanced IDW with adaptive radius"""
        if len(pressures) == 0:
            return np.zeros(grid_coords.shape[0])
        
        # Calculate distances
        distances = np.sqrt(((grid_coords[:, None, :] - sensor_coords[None, :, :]) ** 2).sum(axis=2))
        
        # Apply radius cutoff if specified
        if radius is not None:
            distances = np.where(distances > radius, np.inf, distances)
        
        # Handle zero distances (exact sensor positions)
        zero_mask = distances < 1e-12
        
        # Calculate weights
        weights = 1.0 / (distances ** power + 1e-12)
        weights = np.where(distances == np.inf, 0, weights)
        
        # Normalize weights
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0, 1, weight_sum)
        weights = weights / weight_sum
        
        # Apply exact values where sensors are located
        interpolated = (weights * pressures[None, :]).sum(axis=1)
        
        # Handle exact positions
        if np.any(zero_mask):
            exact_indices = np.where(zero_mask.any(axis=1))[0]
            for idx in exact_indices:
                sensor_idx = np.where(zero_mask[idx])[0][0]
                interpolated[idx] = pressures[sensor_idx]
        
        return interpolated
    
    @staticmethod
    def kriging_interpolation(sensor_coords, pressures, grid_coords):
        """Kriging interpolation for advanced geostatistical analysis"""
        try:
            # Simple kriging using scipy griddata with cubic interpolation
            interpolated = griddata(sensor_coords, pressures, grid_coords, 
                                  method='cubic', fill_value=0.0)
            return np.maximum(interpolated, 0)
        except Exception:
            # Fallback to linear interpolation
            interpolated = griddata(sensor_coords, pressures, grid_coords, 
                                  method='linear', fill_value=0.0)
            return np.maximum(interpolated, 0)

class RealTimeProcessor:
    """Real-time signal processing for smooth visualization"""
    
    def __init__(self, alpha=0.3, noise_threshold=0.5):
        self.alpha = alpha
        self.noise_threshold = noise_threshold
        self.previous_frame = None
        self.noise_buffer = []
        self.peak_detector = PeakDetector()
    
    def process_frame(self, current_frame):
        """Process incoming pressure frame with smoothing and noise reduction"""
        if self.previous_frame is None:
            self.previous_frame = current_frame.copy()
            return current_frame
        
        # Temporal smoothing (EMA)
        smoothed = (1 - self.alpha) * self.previous_frame + self.alpha * current_frame
        
        # Noise reduction
        denoised = self.denoise_frame(smoothed)
        
        # Update previous frame
        self.previous_frame = smoothed.copy()
        
        return denoised
    
    def denoise_frame(self, frame):
        """Remove noise while preserving pressure patterns"""
        # Median filter for spike removal
        from scipy.signal import medfilt
        
        # Apply only to non-zero values to preserve actual pressure patterns
        mask = frame > self.noise_threshold
        if np.any(mask):
            frame_filtered = frame.copy()
            frame_filtered[mask] = medfilt(frame[mask], kernel_size=3)
            return frame_filtered
        
        return frame

class PeakDetector:
    """Detect pressure peaks and their evolution"""
    
    def __init__(self, min_prominence=10.0):
        self.min_prominence = min_prominence
        self.peak_history = []
    
    def detect_peaks(self, pressure_grid, foot_mask):
        """Detect local pressure maxima"""
        from scipy.signal import find_peaks
        
        # Flatten grid for peak detection
        masked_grid = pressure_grid.copy()
        masked_grid[~foot_mask] = 0
        
        flat_pressures = masked_grid.ravel()
        peaks, properties = find_peaks(flat_pressures, 
                                     prominence=self.min_prominence,
                                     distance=5)
        
        if len(peaks) > 0:
            # Convert back to 2D coordinates
            peak_coords = []
            for peak_idx in peaks:
                y, x = divmod(peak_idx, pressure_grid.shape[1])
                peak_coords.append((x, y, flat_pressures[peak_idx]))
            
            self.peak_history.append(peak_coords)
            if len(self.peak_history) > 30:  # Keep last 30 frames
                self.peak_history.pop(0)
            
            return peak_coords
        
        return []

class EnhancedHeatmapCanvas(QWidget):
    """State-of-the-art heatmap canvas with advanced visualization"""
    
    def __init__(self, title="Enhanced Left", grid_w=128, grid_h=160, 
                 n_sensors=16, is_left=True, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.grid_w, self.grid_h = grid_w, grid_h
        self.n_sensors = n_sensors
        self.is_left = is_left
        self.title = title
        
        # Advanced features
        self.interpolation_method = "rbf"  # "rbf", "idw", "kriging"
        self.colormap_name = "clinical_pressure"
        self.smoothing_enabled = True
        self.show_sensors = True
        self.show_zones = False
        self.show_peaks = True
        self.animation_enabled = True
        
        # Processing
        self.processor = RealTimeProcessor(alpha=0.2)
        self.peak_detector = PeakDetector()
        
        # Generate enhanced foot shape and sensor layout
        self._setup_foot_geometry()
        self._setup_ui()
        self._setup_colormaps()
        
        # Animation timer for smooth updates
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animate_frame)
        self.animation_timer.start(16)  # ~60 FPS
    
    def _setup_foot_geometry(self):
        """Setup enhanced foot geometry with anatomical accuracy"""
        # Enhanced foot outline
        self.foot_outline = foot_outline_points_enhanced(
            left=self.is_left, foot_type="normal", samples_per_seg=300)
        
        # Create high-resolution mask
        outline_px = self.foot_outline.copy()
        outline_px[:, 0] *= self.grid_w
        outline_px[:, 1] *= self.grid_h
        self.foot_mask = polygon_to_mask(outline_px, self.grid_w, self.grid_h)
        
        # Generate adaptive sensor placement
        self.sensor_coords = adaptive_sensor_placement(
            self.foot_outline, n_sensors=self.n_sensors, anatomical_priority=True)
        
        # Create pressure zones
        self.pressure_zones = create_pressure_zones_mask(
            self.foot_outline, self.grid_w, self.grid_h)
        
        # Setup interpolation grid
        y_grid, x_grid = np.mgrid[0:self.grid_h, 0:self.grid_w]
        self.grid_coords = np.column_stack([
            ((x_grid + 0.5) / self.grid_w).ravel(),
            ((y_grid + 0.5) / self.grid_h).ravel()
        ])
    
    def _setup_ui(self):
        """Setup the enhanced UI with modern styling"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title with modern styling
        title_label = QLabel(self.title)
        title_label.setObjectName("enhancedPanelTitle")
        title_label.setStyleSheet("""
            QLabel#enhancedPanelTitle { 
                font-size: 16px; 
                font-weight: 700; 
                color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                           stop:0 #2d3748, stop:1 #4a5568);
                padding: 8px;
                border-radius: 6px;
                margin-bottom: 4px;
            }
        """)
        layout.addWidget(title_label)
        
        # Main graphics widget with enhanced styling
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("#1a202c")
        self.glw.setStyleSheet("""
            QWidget {
                border: 2px solid #4a5568;
                border-radius: 8px;
                background: #1a202c;
            }
        """)
        layout.addWidget(self.glw)
        
        # Setup view with enhanced settings
        self.view = self.glw.addViewBox(lockAspect=True, enableMouse=False)
        self.view.setMouseEnabled(x=False, y=False)
        self.view.invertY(True)  # Toes at top
        self.view.setMenuEnabled(False)
        
        # Enhanced image item with smooth scaling
        self.image = pg.ImageItem()
        self.image.setOpts(axisOrder='row-major')
        self.view.addItem(self.image)
        
        # Foot outline with modern styling
        outline_px = self.foot_outline.copy()
        outline_px[:, 0] *= self.grid_w
        outline_px[:, 1] *= self.grid_h
        
        self.outline_item = pg.PlotDataItem(
            x=outline_px[:, 0], 
            y=outline_px[:, 1],
            pen=pg.mkPen(color=(255, 255, 255, 200), width=3,
                        style=pg.QtCore.Qt.SolidLine),
            connect='all'
        )
        self.view.addItem(self.outline_item)
        
        # Sensor positions
        self.sensor_items = []
        if self.show_sensors:
            self._add_sensor_markers()
        
        # Pressure zone overlays
        self.zone_items = []
        if self.show_zones:
            self._add_zone_overlays()
        
        # Peak markers
        self.peak_items = []
        
        # Enhanced colorbar
        self.colorbar = pg.ColorBarItem(
            values=(0, 100),
            width=15,
            interactive=False,
            label="Pressure (kPa)"
        )
        self.colorbar.setLabels(left="Pressure (kPa)")
        self.glw.addItem(self.colorbar)
        
        # Set initial view
        self.view.setRange(
            xRange=(-5, self.grid_w + 5),
            yRange=(-5, self.grid_h + 5),
            padding=0
        )
        
        # Initialize with empty data
        self._set_empty_display()
    
    def _setup_colormaps(self):
        """Setup enhanced colormaps"""
        self.colormaps = {
            "clinical_pressure": AdvancedColorMaps.clinical_pressure(),
            "thermal_enhanced": AdvancedColorMaps.thermal_enhanced(),
            "rainbow_medical": AdvancedColorMaps.rainbow_medical(),
            "viridis": pg.colormap.get("viridis"),
            "plasma": pg.colormap.get("plasma"),
            "inferno": pg.colormap.get("inferno")
        }
        
        # Set initial colormap
        self.set_colormap(self.colormap_name)
    
    def _add_sensor_markers(self):
        """Add visual markers for sensor positions"""
        for i, coord in enumerate(self.sensor_coords):
            x_px = coord[0] * self.grid_w
            y_px = coord[1] * self.grid_h
            
            # Create sensor marker
            marker = pg.ScatterPlotItem(
                pos=[(x_px, y_px)],
                size=8,
                pen=pg.mkPen(color=(255, 255, 255, 180), width=2),
                brush=pg.mkBrush(color=(0, 255, 255, 120)),
                symbol='o'
            )
            self.sensor_items.append(marker)
            self.view.addItem(marker)
    
    def _add_zone_overlays(self):
        """Add anatomical zone overlays"""
        zone_colors = {
            "heel": (255, 0, 0, 60),
            "midfoot_medial": (0, 255, 0, 40),
            "midfoot_lateral": (0, 0, 255, 40),
            "forefoot_medial": (255, 255, 0, 50),
            "forefoot_central": (255, 0, 255, 50),
            "forefoot_lateral": (0, 255, 255, 50),
            "hallux": (255, 128, 0, 70),
            "lesser_toes": (128, 255, 128, 50)
        }
        
        for zone_name, zone_mask in self.pressure_zones.items():
            if zone_name in zone_colors:
                # Create zone boundary
                y_coords, x_coords = np.where(zone_mask)
                if len(x_coords) > 0:
                    # Find boundary points (simplified)
                    boundary_points = []
                    # This is a simplified boundary extraction
                    # In practice, you'd want a more sophisticated contour detection
                    
                    color = zone_colors[zone_name]
                    zone_item = pg.PlotDataItem(
                        pen=pg.mkPen(color=color, width=1, style=pg.QtCore.Qt.DashLine)
                    )
                    self.zone_items.append(zone_item)
                    # self.view.addItem(zone_item)  # Uncomment to show zones
    
    def _set_empty_display(self):
        """Set empty display state"""
        empty_grid = np.zeros((self.grid_h, self.grid_w), dtype=float)
        empty_grid[~self.foot_mask] = np.nan
        
        self.image.setImage(
            empty_grid,
            autoLevels=False,
            levels=(0, 100)
        )
        self.image.setRect(QRectF(0, 0, self.grid_w, self.grid_h))
    
    def set_colormap(self, name: str):
        """Set enhanced colormap"""
        if name in self.colormaps:
            self.colormap_name = name
            colormap = self.colormaps[name]
            lut = colormap.getLookupTable(nPts=256)
            
            self.image.setLookupTable(lut)
            self.colorbar.setColorMap(colormap)
    
    def set_interpolation_method(self, method: str):
        """Set interpolation method"""
        if method in ["rbf", "idw", "kriging"]:
            self.interpolation_method = method
    
    def update_with_pressures(self, pressures: np.ndarray, 
                            show_real_time_effects: bool = True):
        """Update display with new pressure data using advanced processing"""
        if pressures is None or len(pressures) != self.n_sensors:
            return
        
        # Ensure non-negative pressures
        pressures = np.maximum(pressures, 0)
        
        # Real-time processing
        if show_real_time_effects:
            processed_pressures = self.processor.process_frame(pressures)
        else:
            processed_pressures = pressures
        
        # Advanced interpolation
        if self.interpolation_method == "rbf":
            interpolated = AdvancedInterpolation.radial_basis_function(
                self.sensor_coords, processed_pressures, self.grid_coords)
        elif self.interpolation_method == "kriging":
            interpolated = AdvancedInterpolation.kriging_interpolation(
                self.sensor_coords, processed_pressures, self.grid_coords)
        else:  # IDW
            interpolated = AdvancedInterpolation.inverse_distance_weighting(
                self.sensor_coords, processed_pressures, self.grid_coords, 
                power=2.5, radius=0.15)
        
        # Reshape to grid
        pressure_grid = interpolated.reshape(self.grid_h, self.grid_w)
        
        # Apply foot mask
        pressure_grid[~self.foot_mask] = np.nan
        
        # Optional smoothing
        if self.smoothing_enabled:
            # Apply Gaussian smoothing only to valid (non-nan) areas
            valid_mask = ~np.isnan(pressure_grid)
            if np.any(valid_mask):
                smoothed = gaussian_filter(
                    np.where(valid_mask, pressure_grid, 0), 
                    sigma=1.2, 
                    mode='constant', 
                    cval=0
                )
                pressure_grid[valid_mask] = smoothed[valid_mask]
        
        # Update display
        self._update_display(pressure_grid, processed_pressures)
        
        # Detect and show peaks
        if self.show_peaks:
            peaks = self.peak_detector.detect_peaks(pressure_grid, self.foot_mask)
            self._update_peak_markers(peaks)
    
    def _update_display(self, pressure_grid: np.ndarray, sensor_pressures: np.ndarray):
        """Update the visual display with enhanced effects"""
        # Adaptive scaling
        valid_pressures = pressure_grid[~np.isnan(pressure_grid)]
        if len(valid_pressures) > 0:
            p95 = np.percentile(valid_pressures, 95)
            max_pressure = max(10.0, p95 * 1.1)
        else:
            max_pressure = 100.0
        
        # Update image with enhanced contrast
        self.image.setImage(
            pressure_grid,
            autoLevels=False,
            levels=(0, max_pressure)
        )
        
        # Update colorbar
        self.colorbar.setLevels((0, max_pressure))
        
        # Update sensor markers with real-time pressure indication
        if self.show_sensors and hasattr(self, 'sensor_items'):
            for i, (sensor_item, pressure) in enumerate(zip(self.sensor_items, sensor_pressures)):
                if hasattr(sensor_item, 'setData'):
                    # Scale marker size with pressure
                    size = 6 + min(pressure / 10.0, 10)
                    alpha = min(120 + pressure * 2, 255)
                    
                    sensor_item.setData(
                        size=[size],
                        brush=[pg.mkBrush(color=(255, 255, 0, int(alpha)))]
                    )
    
    def _update_peak_markers(self, peaks):
        """Update pressure peak markers"""
        # Remove old peak markers
        for item in self.peak_items:
            self.view.removeItem(item)
        self.peak_items.clear()
        
        # Add new peak markers
        if peaks and len(peaks) > 0:
            peak_data = []
            for x, y, pressure in peaks:
                peak_data.append([x, y])
            
            if peak_data:
                peak_marker = pg.ScatterPlotItem(
                    pos=peak_data,
                    size=15,
                    pen=pg.mkPen(color=(255, 255, 255, 255), width=2),
                    brush=pg.mkBrush(color=(255, 0, 0, 150)),
                    symbol='star'
                )
                self.peak_items.append(peak_marker)
                self.view.addItem(peak_marker)
    
    def _animate_frame(self):
        """Handle smooth animations"""
        if self.animation_enabled:
            # Add subtle pulsing effect to active sensors
            pass  # Implement animation effects as needed

class StateOfTheArtHeatmapView(QWidget):
    """Complete state-of-the-art heatmap visualization system"""
    
    def __init__(self, grid_w=128, grid_h=160, n_sensors=16, 
                 title="State-of-the-Art Plantar Pressure Analysis"):
        super().__init__()
        
        self.setObjectName("StateOfTheArtHeatmapPanel")
        self.setStyleSheet("""
            QWidget#StateOfTheArtHeatmapPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                           stop:0 #2d3748, stop:1 #1a202c);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        # Configuration
        self.grid_w, self.grid_h = grid_w, grid_h
        self.n_sensors = n_sensors
        
        self._setup_ui(title)
        self._setup_controls()
    
    def _setup_ui(self, title):
        """Setup the main UI layout"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        
        # Enhanced title
        title_label = QLabel(title)
        title_label.setObjectName("stateArtTitle")
        title_label.setStyleSheet("""
            QLabel#stateArtTitle {
                font-size: 20px;
                font-weight: 800;
                color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                           stop:0 #667eea, stop:1 #764ba2);
                padding: 12px;
                border-radius: 8px;
                text-align: center;
            }
        """)
        main_layout.addWidget(title_label)
        
        # Heatmap display area
        display_layout = QHBoxLayout()
        display_layout.setSpacing(15)
        
        # Left foot canvas
        self.left_canvas = EnhancedHeatmapCanvas(
            "Left Foot Analysis", self.grid_w, self.grid_h, 
            self.n_sensors, is_left=True, parent=self
        )
        
        # Right foot canvas
        self.right_canvas = EnhancedHeatmapCanvas(
            "Right Foot Analysis", self.grid_w, self.grid_h, 
            self.n_sensors, is_left=False, parent=self
        )
        
        # Add canvases to layout with equal weight
        display_layout.addWidget(self.left_canvas, 1)
        display_layout.addWidget(self.right_canvas, 1)
        
        main_layout.addLayout(display_layout, 1)
    
    def _setup_controls(self):
        """Setup advanced control panel"""
        controls_frame = QFrame()
        controls_frame.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setSpacing(20)
        
        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_group.setStyleSheet("QGroupBox { font-weight: bold; color: white; }")
        viz_layout = QHBoxLayout(viz_group)
        
        # Colormap selection
        colormap_combo = QComboBox()
        colormap_combo.addItems([
            "Clinical Pressure", "Thermal Enhanced", "Rainbow Medical",
            "Viridis", "Plasma", "Inferno"
        ])
        colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        viz_layout.addWidget(QLabel("Colormap:"))
        viz_layout.addWidget(colormap_combo)
        
        # Interpolation method
        interp_combo = QComboBox()
        interp_combo.addItems(["RBF (Recommended)", "IDW", "Kriging"])
        interp_combo.currentTextChanged.connect(self._on_interpolation_changed)
        viz_layout.addWidget(QLabel("Interpolation:"))
        viz_layout.addWidget(interp_combo)
        
        controls_layout.addWidget(viz_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_group.setStyleSheet("QGroupBox { font-weight: bold; color: white; }")
        display_layout = QHBoxLayout(display_group)
        
        # Checkboxes for various display options
        self.smoothing_cb = QCheckBox("Smoothing")
        self.smoothing_cb.setChecked(True)
        self.smoothing_cb.toggled.connect(self._on_smoothing_toggled)
        
        self.sensors_cb = QCheckBox("Show Sensors")
        self.sensors_cb.setChecked(True)
        self.sensors_cb.toggled.connect(self._on_sensors_toggled)
        
        self.peaks_cb = QCheckBox("Show Peaks")
        self.peaks_cb.setChecked(True)
        self.peaks_cb.toggled.connect(self._on_peaks_toggled)
        
        self.zones_cb = QCheckBox("Anatomical Zones")
        self.zones_cb.toggled.connect(self._on_zones_toggled)
        
        display_layout.addWidget(self.smoothing_cb)
        display_layout.addWidget(self.sensors_cb)
        display_layout.addWidget(self.peaks_cb)
        display_layout.addWidget(self.zones_cb)
        
        controls_layout.addWidget(display_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced")
        advanced_group.setStyleSheet("QGroupBox { font-weight: bold; color: white; }")
        advanced_layout = QHBoxLayout(advanced_group)
        
        # Real-time processing toggle
        self.realtime_cb = QCheckBox("Real-time Processing")
        self.realtime_cb.setChecked(True)
        advanced_layout.addWidget(self.realtime_cb)
        
        # Animation toggle
        self.animation_cb = QCheckBox("Animations")
        self.animation_cb.setChecked(True)
        self.animation_cb.toggled.connect(self._on_animation_toggled)
        advanced_layout.addWidget(self.animation_cb)
        
        controls_layout.addWidget(advanced_group)
        
        # Add controls to main layout
        self.layout().addWidget(controls_frame)
    
    def _on_colormap_changed(self, colormap_name):
        """Handle colormap change"""
        colormap_map = {
            "Clinical Pressure": "clinical_pressure",
            "Thermal Enhanced": "thermal_enhanced", 
            "Rainbow Medical": "rainbow_medical",
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno"
        }
        
        if colormap_name in colormap_map:
            colormap_key = colormap_map[colormap_name]
            self.left_canvas.set_colormap(colormap_key)
            self.right_canvas.set_colormap(colormap_key)
    
    def _on_interpolation_changed(self, interp_name):
        """Handle interpolation method change"""
        interp_map = {
            "RBF (Recommended)": "rbf",
            "IDW": "idw",
            "Kriging": "kriging"
        }
        
        if interp_name in interp_map:
            interp_method = interp_map[interp_name]
            self.left_canvas.set_interpolation_method(interp_method)
            self.right_canvas.set_interpolation_method(interp_method)
    
    def _on_smoothing_toggled(self, enabled):
        """Handle smoothing toggle"""
        self.left_canvas.smoothing_enabled = enabled
        self.right_canvas.smoothing_enabled = enabled
    
    def _on_sensors_toggled(self, enabled):
        """Handle sensor display toggle"""
        self.left_canvas.show_sensors = enabled
        self.right_canvas.show_sensors = enabled
        
        # Show/hide sensor markers
        for canvas in [self.left_canvas, self.right_canvas]:
            for sensor_item in canvas.sensor_items:
                sensor_item.setVisible(enabled)
    
    def _on_peaks_toggled(self, enabled):
        """Handle peak detection toggle"""
        self.left_canvas.show_peaks = enabled
        self.right_canvas.show_peaks = enabled
    
    def _on_zones_toggled(self, enabled):
        """Handle anatomical zones toggle"""
        self.left_canvas.show_zones = enabled
        self.right_canvas.show_zones = enabled
    
    def _on_animation_toggled(self, enabled):
        """Handle animation toggle"""
        self.left_canvas.animation_enabled = enabled
        self.right_canvas.animation_enabled = enabled
    
    # Public API methods
    def update_with_sample(self, sample: dict, copL=None, copR=None):
        """Update both feet with new pressure sample"""
        left_pressures = np.array(sample.get("left", [0] * self.n_sensors), dtype=float)
        right_pressures = np.array(sample.get("right", [0] * self.n_sensors), dtype=float)
        
        # Ensure non-negative pressures
        left_pressures = np.maximum(left_pressures, 0)
        right_pressures = np.maximum(right_pressures, 0)
        
        # Update canvases
        use_realtime = self.realtime_cb.isChecked()
        self.left_canvas.update_with_pressures(left_pressures, use_realtime)
        self.right_canvas.update_with_pressures(right_pressures, use_realtime)
    
    def set_foot_type(self, foot_type: str):
        """Set foot type for both feet (normal, pes_planus, pes_cavus, etc.)"""
        # This would require regenerating the foot geometry
        # Implementation depends on whether you want dynamic foot type changes
        pass
    
    def export_current_frame(self, filepath: str):
        """Export current heatmap frame as high-resolution image"""
        # Implementation for exporting current visualization
        pass