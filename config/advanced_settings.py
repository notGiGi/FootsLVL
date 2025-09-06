# config/advanced_settings.py
"""
Advanced configuration system for FootLab with validation, 
profiles, and clinical customization
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime
from PySide6.QtCore import QSettings
from pydantic import BaseModel, Field, validator
import jsonschema

logger = logging.getLogger(__name__)

class ConfigCategory(Enum):
    SYSTEM = "system"
    SENSORS = "sensors"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    CLINICAL = "clinical"
    ADVANCED = "advanced"
    USER_INTERFACE = "user_interface"
    EXPORT = "export"

@dataclass
class SensorConfig:
    """Sensor configuration settings"""
    type: str = "simulator"  # "simulator", "nurvv_ble", "novel", "tekscan"
    sampling_rate: float = 100.0
    n_sensors_per_foot: int = 16
    calibration_enabled: bool = True
    auto_connect: bool = True
    connection_timeout: float = 10.0
    max_reconnect_attempts: int = 5
    sync_window_ms: float = 50.0
    
    # Sensor-specific settings
    nurvv_settings: Dict[str, Any] = field(default_factory=lambda: {
        "sync_calibration_duration": 5.0,
        "battery_warning_level": 20,
        "rssi_threshold": -80
    })
    
    simulator_settings: Dict[str, Any] = field(default_factory=lambda: {
        "base_amplitude": 60.0,
        "noise_level": 0.4,
        "gait_pattern": "normal"  # "normal", "pathological", "elderly"
    })

@dataclass 
class AnalysisConfig:
    """Analysis configuration settings"""
    mode: str = "clinical"  # "real_time", "clinical", "research", "rehabilitation"
    enable_advanced_biomechanics: bool = True
    enable_pathology_detection: bool = True
    enable_real_time_processing: bool = True
    
    # Gait analysis parameters
    min_contact_pressure: float = 5.0
    heel_strike_threshold: float = 0.15
    toe_off_threshold: float = 0.10
    step_detection_sensitivity: float = 0.8
    
    # Temporal parameters
    stance_phase_normal_range: Tuple[float, float] = (60.0, 65.0)  # percentage
    cadence_normal_range: Tuple[float, float] = (100.0, 120.0)  # steps/min
    asymmetry_threshold: float = 15.0  # percentage
    
    # Pathology detection
    ml_confidence_threshold: float = 0.7
    enable_anomaly_detection: bool = True
    pathology_models_path: str = "models/pathology"
    
    # Advanced features
    enable_frequency_analysis: bool = True
    enable_stability_metrics: bool = True
    enable_cop_analysis: bool = True

@dataclass
class VisualizationConfig:
    """Visualization configuration settings"""
    # Heatmap settings
    grid_resolution: Tuple[int, int] = (128, 160)
    interpolation_method: str = "rbf"  # "rbf", "idw", "kriging"
    smoothing_enabled: bool = True
    smoothing_sigma: float = 1.2
    
    # Color settings
    colormap: str = "clinical_pressure"
    auto_scale_intensity: bool = True
    intensity_scale: float = 1.0
    show_pressure_contours: bool = False
    
    # Anatomical features
    show_foot_outline: bool = True
    show_sensor_positions: bool = False
    show_anatomical_zones: bool = False
    show_pressure_peaks: bool = True
    
    # Animation and updates
    update_rate_hz: float = 30.0
    enable_animations: bool = True
    trail_length: int = 50  # CoP trail length
    
    # Clinical display
    show_reference_values: bool = True
    show_asymmetry_indicators: bool = True
    highlight_abnormal_values: bool = True

@dataclass
class ClinicalConfig:
    """Clinical-specific configuration"""
    # Units and scales
    pressure_units: str = "kPa"  # "kPa", "psi", "N/cm2"
    length_units: str = "mm"     # "mm", "cm", "inches"
    weight_units: str = "kg"     # "kg", "lbs"
    
    # Reference values (can be customized per population)
    reference_values: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "peak_pressure": (50.0, 250.0),  # kPa
        "contact_time": (0.5, 0.8),      # seconds
        "cop_velocity": (50.0, 200.0),   # mm/s
        "stance_percentage": (60.0, 65.0) # percentage
    })
    
    # Clinical flags and alerts
    enable_clinical_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high_pressure_alert": 300.0,      # kPa
        "asymmetry_alert": 20.0,           # percentage
        "temporal_irregularity": 15.0,     # percentage CV
        "balance_deficit": 25.0            # percentage
    })
    
    # Population-specific settings
    population_norms: str = "adult_healthy"  # "adult_healthy", "elderly", "pediatric", "diabetic"
    
    # Report settings
    include_normative_comparison: bool = True
    include_trend_analysis: bool = True
    generate_clinical_recommendations: bool = True

@dataclass
class UIConfig:
    """User interface configuration"""
    theme: str = "dark_modern"  # "dark_modern", "light_clinical", "high_contrast"
    font_family: str = "Segoe UI"
    font_size: int = 11
    
    # Layout preferences
    default_layout: str = "clinical"  # "clinical", "research", "compact"
    show_status_bar: bool = True
    show_tool_bar: bool = True
    
    # Window settings
    remember_window_geometry: bool = True
    auto_save_interval: int = 300  # seconds
    
    # Accessibility
    high_contrast_mode: bool = False
    large_text_mode: bool = False
    screen_reader_support: bool = False

@dataclass
class ExportConfig:
    """Export and reporting configuration"""
    # Default export paths
    reports_directory: str = "reports"
    data_directory: str = "data"
    temp_directory: str = "temp"
    
    # Report settings
    default_report_format: str = "pdf"  # "pdf", "docx", "html"
    include_raw_data: bool = False
    include_analysis_plots: bool = True
    include_statistical_summary: bool = True
    
    # PDF settings
    pdf_dpi: int = 300
    pdf_page_size: str = "A4"  # "A4", "Letter", "A3"
    
    # Data export settings
    csv_delimiter: str = ","
    include_timestamps: bool = True
    export_coordinate_system: str = "normalized"  # "normalized", "pixels", "millimeters"

class AdvancedFootLabConfig:
    """Advanced configuration management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config")
        self.config_path.mkdir(exist_ok=True)
        
        # Configuration components
        self.sensor = SensorConfig()
        self.analysis = AnalysisConfig()
        self.visualization = VisualizationConfig()
        self.clinical = ClinicalConfig()
        self.ui = UIConfig()
        self.export = ExportConfig()
        
        # Configuration metadata
        self.version = "2.0"
        self.created = datetime.now()
        self.modified = datetime.now()
        self.profile_name = "default"
        
        # Load configuration
        self.load_config()
    
    def load_config(self, profile: str = "default"):
        """Load configuration from file"""
        config_file = self.config_path / f"{profile}.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                self._update_from_dict(data)
                self.profile_name = profile
                
                logger.info(f"Configuration loaded from {config_file}")
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                self._load_defaults()
        else:
            logger.info("Using default configuration")
            self._load_defaults()
    
    def save_config(self, profile: str = None):
        """Save configuration to file"""
        profile = profile or self.profile_name
        config_file = self.config_path / f"{profile}.json"
        
        try:
            data = self._to_dict()
            data['profile_name'] = profile
            data['modified'] = datetime.now().isoformat()
            
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.profile_name = profile
            self.modified = datetime.now()
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def _load_defaults(self):
        """Load default configuration values"""
        self.sensor = SensorConfig()
        self.analysis = AnalysisConfig()
        self.visualization = VisualizationConfig()
        self.clinical = ClinicalConfig()
        self.ui = UIConfig()
        self.export = ExportConfig()
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "version": self.version,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "sensor": asdict(self.sensor),
            "analysis": asdict(self.analysis),
            "visualization": asdict(self.visualization),
            "clinical": asdict(self.clinical),
            "ui": asdict(self.ui),
            "export": asdict(self.export)
        }
    
    def _update_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary"""
        if "sensor" in data:
            self.sensor = SensorConfig(**data["sensor"])
        
        if "analysis" in data:
            self.analysis = AnalysisConfig(**data["analysis"])
        
        if "visualization" in data:
            self.visualization = VisualizationConfig(**data["visualization"])
        
        if "clinical" in data:
            self.clinical = ClinicalConfig(**data["clinical"])
        
        if "ui" in data:
            self.ui = UIConfig(**data["ui"])
        
        if "export" in data:
            self.export = ExportConfig(**data["export"])
        
        # Update metadata
        if "version" in data:
            self.version = data["version"]
        
        if "created" in data:
            self.created = datetime.fromisoformat(data["created"])
        
        if "modified" in data:
            self.modified = datetime.fromisoformat(data["modified"])
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate configuration settings"""
        errors = []
        
        # Validate sensor configuration
        if self.sensor.sampling_rate < 10 or self.sensor.sampling_rate > 1000:
            errors.append("Sampling rate must be between 10-1000 Hz")
        
        if self.sensor.n_sensors_per_foot < 1 or self.sensor.n_sensors_per_foot > 64:
            errors.append("Number of sensors per foot must be between 1-64")
        
        # Validate analysis configuration
        if self.analysis.min_contact_pressure < 0:
            errors.append("Minimum contact pressure cannot be negative")
        
        if not (0 < self.analysis.heel_strike_threshold < 1):
            errors.append("Heel strike threshold must be between 0-1")
        
        # Validate visualization configuration
        grid_w, grid_h = self.visualization.grid_resolution
        if grid_w < 32 or grid_h < 32 or grid_w > 512 or grid_h > 512:
            errors.append("Grid resolution must be between 32x32 and 512x512")
        
        # Validate clinical configuration
        valid_units = {"kPa", "psi", "N/cm2"}
        if self.clinical.pressure_units not in valid_units:
            errors.append(f"Pressure units must be one of: {valid_units}")
        
        # Validate paths
        try:
            Path(self.export.reports_directory).mkdir(parents=True, exist_ok=True)
            Path(self.export.data_directory).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create export directories: {e}")
        
        return len(errors) == 0, errors
    
    def get_profile_list(self) -> List[str]:
        """Get list of available configuration profiles"""
        profiles = []
        
        for config_file in self.config_path.glob("*.json"):
            if config_file.stem != "schema":  # Skip schema file
                profiles.append(config_file.stem)
        
        return sorted(profiles)
    
    def create_profile(self, name: str, description: str = ""):
        """Create a new configuration profile"""
        profile_data = self._to_dict()
        profile_data['description'] = description
        profile_data['created'] = datetime.now().isoformat()
        
        profile_file = self.config_path / f"{name}.json"
        
        with open(profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2, default=str)
        
        logger.info(f"Created configuration profile: {name}")
    
    def delete_profile(self, name: str):
        """Delete a configuration profile"""
        if name == "default":
            raise ValueError("Cannot delete default profile")
        
        profile_file = self.config_path / f"{name}.json"
        
        if profile_file.exists():
            profile_file.unlink()
            logger.info(f"Deleted configuration profile: {name}")
        else:
            raise FileNotFoundError(f"Profile {name} not found")
    
    def export_config(self, file_path: str, format: str = "json"):
        """Export configuration to file"""
        path = Path(file_path)
        data = self._to_dict()
        
        if format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format.lower() == "yaml":
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Configuration exported to {path}")
    
    def import_config(self, file_path: str):
        """Import configuration from file"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix.lower() == '.json':
                data = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Validate before importing
        try:
            temp_config = AdvancedFootLabConfig()
            temp_config._update_from_dict(data)
            valid, errors = temp_config.validate_config()
            
            if not valid:
                raise ValueError(f"Invalid configuration: {errors}")
            
            # Import successful
            self._update_from_dict(data)
            logger.info(f"Configuration imported from {path}")
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            raise
    
    def reset_to_defaults(self, category: Optional[ConfigCategory] = None):
        """Reset configuration to defaults"""
        if category is None:
            self._load_defaults()
            logger.info("All configuration reset to defaults")
        else:
            if category == ConfigCategory.SENSORS:
                self.sensor = SensorConfig()
            elif category == ConfigCategory.ANALYSIS:
                self.analysis = AnalysisConfig()
            elif category == ConfigCategory.VISUALIZATION:
                self.visualization = VisualizationConfig()
            elif category == ConfigCategory.CLINICAL:
                self.clinical = ClinicalConfig()
            elif category == ConfigCategory.USER_INTERFACE:
                self.ui = UIConfig()
            elif category == ConfigCategory.EXPORT:
                self.export = ExportConfig()
            
            logger.info(f"Configuration category {category.value} reset to defaults")
        
        self.modified = datetime.now()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        return {
            "Profile": self.profile_name,
            "Version": self.version,
            "Last Modified": self.modified.strftime("%Y-%m-%d %H:%M:%S"),
            "Sensor Type": self.sensor.type,
            "Sampling Rate": f"{self.sensor.sampling_rate} Hz",
            "Analysis Mode": self.analysis.mode,
            "Pathology Detection": "Enabled" if self.analysis.enable_pathology_detection else "Disabled",
            "Visualization": f"{self.visualization.grid_resolution[0]}x{self.visualization.grid_resolution[1]}",
            "Theme": self.ui.theme
        }

# Predefined configuration profiles
CLINICAL_PROFILE = {
    "sensor": {
        "type": "nurvv_ble",
        "sampling_rate": 100.0,
        "auto_connect": True,
        "calibration_enabled": True
    },
    "analysis": {
        "mode": "clinical",
        "enable_pathology_detection": True,
        "enable_advanced_biomechanics": True,
        "asymmetry_threshold": 15.0
    },
    "visualization": {
        "colormap": "clinical_pressure",
        "show_anatomical_zones": True,
        "show_pressure_peaks": True
    },
    "clinical": {
        "enable_clinical_alerts": True,
        "include_normative_comparison": True,
        "generate_clinical_recommendations": True
    }
}

RESEARCH_PROFILE = {
    "sensor": {
        "type": "nurvv_ble", 
        "sampling_rate": 200.0,
        "calibration_enabled": True
    },
    "analysis": {
        "mode": "research",
        "enable_advanced_biomechanics": True,
        "enable_frequency_analysis": True,
        "enable_stability_metrics": True
    },
    "visualization": {
        "grid_resolution": (256, 320),
        "interpolation_method": "rbf",
        "show_sensor_positions": True
    },
    "export": {
        "include_raw_data": True,
        "pdf_dpi": 600,
        "include_statistical_summary": True
    }
}

REHABILITATION_PROFILE = {
    "sensor": {
        "type": "simulator",  # For training/demo
        "sampling_rate": 100.0
    },
    "analysis": {
        "mode": "rehabilitation",
        "enable_real_time_processing": True,
        "asymmetry_threshold": 10.0
    },
    "visualization": {
        "show_asymmetry_indicators": True,
        "highlight_abnormal_values": True,
        "enable_animations": True
    },
    "clinical": {
        "enable_clinical_alerts": True,
        "population_norms": "elderly"
    }
}

def create_default_profiles(config_path: str = "config"):
    """Create default configuration profiles"""
    config_dir = Path(config_path)
    config_dir.mkdir(exist_ok=True)
    
    profiles = {
        "clinical": CLINICAL_PROFILE,
        "research": RESEARCH_PROFILE,
        "rehabilitation": REHABILITATION_PROFILE
    }
    
    for name, profile_data in profiles.items():
        profile_file = config_dir / f"{name}.json"
        
        if not profile_file.exists():
            with open(profile_file, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            logger.info(f"Created default profile: {name}")

# Configuration validation schema (JSON Schema)
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "sensor": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["simulator", "nurvv_ble", "novel", "tekscan"]},
                "sampling_rate": {"type": "number", "minimum": 10, "maximum": 1000},
                "n_sensors_per_foot": {"type": "integer", "minimum": 1, "maximum": 64}
            },
            "required": ["type", "sampling_rate"]
        },
        "analysis": {
            "type": "object", 
            "properties": {
                "mode": {"type": "string", "enum": ["real_time", "clinical", "research", "rehabilitation"]},
                "asymmetry_threshold": {"type": "number", "minimum": 0, "maximum": 100}
            }
        }
    },
    "required": ["sensor", "analysis"]
}

def validate_config_with_schema(config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate configuration against JSON schema"""
    try:
        jsonschema.validate(config_data, CONFIG_SCHEMA)
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e)]

# Factory function
def create_config(profile: str = "default", config_path: str = None) -> AdvancedFootLabConfig:
    """Create configuration instance with specified profile"""
    config = AdvancedFootLabConfig(config_path)
    
    if profile != "default":
        try:
            config.load_config(profile)
        except Exception as e:
            logger.warning(f"Failed to load profile {profile}, using default: {e}")
    
    return config

# Integration with Qt Settings for UI preferences
class QtSettingsManager:
    """Manages Qt-specific UI settings"""
    
    def __init__(self, organization: str = "FootLab", application: str = "FootLab"):
        self.settings = QSettings(organization, application)
    
    def save_window_geometry(self, window):
        """Save window geometry"""
        self.settings.setValue("geometry", window.saveGeometry())
        self.settings.setValue("windowState", window.saveState())
    
    def restore_window_geometry(self, window):
        """Restore window geometry"""
        geometry = self.settings.value("geometry")
        if geometry:
            window.restoreGeometry(geometry)
        
        state = self.settings.value("windowState") 
        if state:
            window.restoreState(state)
    
    def save_user_preferences(self, preferences: Dict[str, Any]):
        """Save user preferences"""
        for key, value in preferences.items():
            self.settings.setValue(f"user_prefs/{key}", value)
    
    def load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences"""
        preferences = {}
        
        self.settings.beginGroup("user_prefs")
        for key in self.settings.childKeys():
            preferences[key] = self.settings.value(key)
        self.settings.endGroup()
        
        return preferences
    
    def clear_settings(self):
        """Clear all settings"""
        self.settings.clear()

# Example usage
if __name__ == "__main__":
    # Create default profiles
    create_default_profiles()
    
    # Create configuration
    config = create_config("clinical")
    
    # Display configuration summary
    summary = config.get_config_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Validate configuration
    valid, errors = config.validate_config()
    if valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Save configuration
    config.save_config("test_profile")
    print("Configuration saved successfully")