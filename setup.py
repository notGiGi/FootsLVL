# setup.py
"""
FootLab State-of-the-Art Baropodometry System
Professional installation and deployment script
"""

import sys
import os
import subprocess
import platform
from pathlib import Path
from setuptools import setup, find_packages, Command
import pkg_resources
from distutils.command.build import build
from distutils.command.install import install
import zipfile
import requests
from typing import List, Dict, Any

# Package metadata
PACKAGE_NAME = "footlab"
VERSION = "2.0.0"
DESCRIPTION = "State-of-the-Art Clinical Baropodometry System"
AUTHOR = "FootLab Development Team"
AUTHOR_EMAIL = "dev@footlab.com"
URL = "https://github.com/footlab/footlab-pro"

# System requirements
PYTHON_REQUIRES = ">=3.9"
PLATFORMS = ["Windows", "macOS", "Linux"]

# Core dependencies
CORE_REQUIREMENTS = [
    "PySide6>=6.4.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0", 
    "pandas>=1.3.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "scikit-learn>=1.0.0",
    "pyqtgraph>=0.13.0",
    "asyncio-mqtt>=0.11.0",
    "bleak>=0.19.0",
    "pydantic>=1.9.0",
    "jsonschema>=4.0.0",
    "PyYAML>=6.0",
    "reportlab>=3.6.0",
    "Pillow>=9.0.0",
    "psutil>=5.8.0",
    "python-dotenv>=0.19.0",
]

# Optional dependencies for advanced features
ADVANCED_REQUIREMENTS = [
    "tensorflow>=2.8.0",
    "torch>=1.11.0",
    "opencv-python>=4.5.0",
    "plotly>=5.6.0",
    "dash>=2.3.0",
    "jupyter>=1.0.0",
    "ipywidgets>=7.6.0",
]

# Development dependencies
DEV_REQUIREMENTS = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.18.0",
    "pytest-qt>=4.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.931",
    "pre-commit>=2.17.0",
    "sphinx>=4.4.0",
    "sphinx-rtd-theme>=1.0.0",
]

# Platform-specific requirements
PLATFORM_REQUIREMENTS = {
    "Windows": [
        "pywin32>=227",
        "wmi>=1.5.1",
    ],
    "Darwin": [  # macOS
        "pyobjc-core>=8.0",
        "pyobjc-framework-Cocoa>=8.0",
    ],
    "Linux": [
        "python3-dev",
        "libgl1-mesa-glx",
    ]
}

class PreInstallCommand(Command):
    """Pre-installation system checks and setup"""
    description = "Perform pre-installation checks"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("🔍 Running pre-installation checks...")
        
        # Check Python version
        if sys.version_info < (3, 9):
            raise RuntimeError("Python 3.9 or higher is required")
        print(f"✓ Python version: {sys.version}")
        
        # Check platform
        system = platform.system()
        if system not in PLATFORMS:
            print(f"⚠️  Warning: {system} is not officially supported")
        else:
            print(f"✓ Platform: {system}")
        
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 4:
                print(f"⚠️  Warning: Low memory detected ({memory_gb:.1f} GB). 8GB+ recommended")
            else:
                print(f"✓ Memory: {memory_gb:.1f} GB")
        except ImportError:
            print("ℹ️  Memory check skipped (psutil not available)")
        
        # Check disk space
        try:
            import shutil
            free_space_gb = shutil.disk_usage(".").free / (1024**3)
            if free_space_gb < 2:
                raise RuntimeError(f"Insufficient disk space: {free_space_gb:.1f} GB available, 2GB+ required")
            print(f"✓ Disk space: {free_space_gb:.1f} GB available")
        except Exception as e:
            print(f"⚠️  Disk space check failed: {e}")
        
        # Check for existing installations
        self.check_existing_installations()
        
        print("✅ Pre-installation checks completed")

    def check_existing_installations(self):
        """Check for existing FootLab installations"""
        try:
            import footlab
            print(f"⚠️  Existing FootLab installation detected: {footlab.__version__}")
            response = input("Continue with installation? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                raise RuntimeError("Installation cancelled by user")
        except ImportError:
            print("✓ No existing installation detected")

class PostInstallCommand(Command):
    """Post-installation setup and configuration"""
    description = "Perform post-installation setup"
    user_options = [
        ('create-desktop-shortcut', None, 'Create desktop shortcut'),
        ('setup-sample-data', None, 'Install sample data'),
        ('configure-system', None, 'Run system configuration'),
    ]

    def initialize_options(self):
        self.create_desktop_shortcut = None
        self.setup_sample_data = None
        self.configure_system = None

    def finalize_options(self):
        pass

    def run(self):
        print("🔧 Running post-installation setup...")
        
        # Create application directories
        self.create_app_directories()
        
        # Install default configurations
        self.install_default_configs()
        
        # Download and install sample data if requested
        if self.setup_sample_data:
            self.install_sample_data()
        
        # Create desktop shortcut if requested
        if self.create_desktop_shortcut:
            self.create_desktop_shortcut()
        
        # Run initial configuration if requested
        if self.configure_system:
            self.run_initial_configuration()
        
        print("✅ Post-installation setup completed")

    def create_app_directories(self):
        """Create application directories"""
        app_dirs = [
            "config",
            "data",
            "data/sessions", 
            "data/patients",
            "reports",
            "logs",
            "temp",
            "models",
            "assets"
        ]
        
        for dir_name in app_dirs:
            dir_path = Path.home() / ".footlab" / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {dir_path}")

    def install_default_configs(self):
        """Install default configuration files"""
        from config.advanced_settings import create_default_profiles
        
        config_path = Path.home() / ".footlab" / "config"
        create_default_profiles(str(config_path))
        print("✓ Default configuration profiles installed")

    def install_sample_data(self):
        """Download and install sample data"""
        print("📥 Installing sample data...")
        
        sample_data_url = "https://github.com/footlab/sample-data/archive/main.zip"
        sample_data_path = Path.home() / ".footlab" / "data" / "samples"
        
        try:
            # Download sample data
            response = requests.get(sample_data_url)
            response.raise_for_status()
            
            # Extract to samples directory
            sample_data_path.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_file.extractall(sample_data_path)
            
            print("✓ Sample data installed successfully")
            
        except Exception as e:
            print(f"⚠️  Failed to install sample data: {e}")

    def create_desktop_shortcut(self):
        """Create desktop shortcut"""
        try:
            if platform.system() == "Windows":
                self.create_windows_shortcut()
            elif platform.system() == "Darwin":
                self.create_macos_shortcut()
            elif platform.system() == "Linux":
                self.create_linux_shortcut()
            
            print("✓ Desktop shortcut created")
            
        except Exception as e:
            print(f"⚠️  Failed to create desktop shortcut: {e}")

    def create_windows_shortcut(self):
        """Create Windows desktop shortcut"""
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "FootLab.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = "-m footlab"
        shortcut.WorkingDirectory = str(Path.home() / ".footlab")
        shortcut.IconLocation = sys.executable
        shortcut.save()

    def create_macos_shortcut(self):
        """Create macOS application bundle"""
        # This would create a proper .app bundle
        # Implementation depends on specific macOS deployment requirements
        pass

    def create_linux_shortcut(self):
        """Create Linux desktop entry"""
        desktop_entry = f"""[Desktop Entry]
Name=FootLab
Comment=State-of-the-Art Baropodometry System
Exec={sys.executable} -m footlab
Icon=footlab
Terminal=false
Type=Application
Categories=Science;Medical;
"""
        
        desktop_path = Path.home() / "Desktop" / "FootLab.desktop"
        with open(desktop_path, 'w') as f:
            f.write(desktop_entry)
        
        # Make executable
        desktop_path.chmod(0o755)

    def run_initial_configuration(self):
        """Run initial system configuration"""
        print("🔧 Running initial configuration...")
        
        # This would launch a configuration wizard
        # For now, just create basic config
        try:
            from config.advanced_settings import create_config
            config = create_config()
            config.save_config("user_default")
            print("✓ Initial configuration completed")
        except Exception as e:
            print(f"⚠️  Configuration setup failed: {e}")

class TestCommand(Command):
    """Run comprehensive test suite"""
    description = "Run test suite"
    user_options = [
        ('test-type=', 't', 'Type of tests to run (unit, integration, ui, all)'),
        ('coverage', 'c', 'Generate coverage report'),
        ('verbose', 'v', 'Verbose output'),
    ]

    def initialize_options(self):
        self.test_type = 'all'
        self.coverage = None
        self.verbose = None

    def finalize_options(self):
        pass

    def run(self):
        print(f"🧪 Running {self.test_type} tests...")
        
        import pytest
        
        # Build pytest arguments
        args = []
        
        if self.test_type == 'unit':
            args.append('tests/unit')
        elif self.test_type == 'integration':
            args.append('tests/integration')
        elif self.test_type == 'ui':
            args.append('tests/ui')
        else:
            args.append('tests')
        
        if self.coverage:
            args.extend(['--cov=footlab', '--cov-report=html', '--cov-report=term'])
        
        if self.verbose:
            args.append('-v')
        
        # Run tests
        exit_code = pytest.main(args)
        
        if exit_code == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Tests failed with exit code {exit_code}")
            sys.exit(exit_code)

class BuildExecutableCommand(Command):
    """Build standalone executable"""
    description = "Build standalone executable using PyInstaller"
    user_options = [
        ('platform=', 'p', 'Target platform (windows, macos, linux)'),
        ('onefile', None, 'Create single file executable'),
    ]

    def initialize_options(self):
        self.platform = platform.system().lower()
        self.onefile = None

    def finalize_options(self):
        pass

    def run(self):
        print(f"🏗️  Building executable for {self.platform}...")
        
        try:
            import PyInstaller.__main__
            
            # PyInstaller arguments
            args = [
                'main_application.py',
                '--name=FootLab',
                '--windowed',
                '--add-data=assets;assets',
                '--add-data=config;config',
                '--add-data=models;models',
                '--hidden-import=scipy.special.cython_special',
                '--hidden-import=sklearn.utils._cython_blas',
                '--hidden-import=sklearn.neighbors.typedefs',
            ]
            
            if self.onefile:
                args.append('--onefile')
            
            # Platform-specific options
            if self.platform == 'windows':
                args.extend([
                    '--icon=assets/footlab.ico',
                    '--version-file=version_info.txt'
                ])
            elif self.platform == 'macos':
                args.extend([
                    '--icon=assets/footlab.icns',
                    '--osx-bundle-identifier=com.footlab.app'
                ])
            
            PyInstaller.__main__.run(args)
            
            print("✅ Executable built successfully!")
            print(f"📦 Output location: dist/FootLab")
            
        except ImportError:
            print("❌ PyInstaller not found. Install with: pip install pyinstaller")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Build failed: {e}")
            sys.exit(1)

class CleanCommand(Command):
    """Clean build artifacts"""
    description = "Clean build artifacts and temporary files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("🧹 Cleaning build artifacts...")
        
        # Directories to clean
        clean_dirs = [
            "build",
            "dist", 
            "*.egg-info",
            "__pycache__",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
        ]
        
        import shutil
        import glob
        
        for pattern in clean_dirs:
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"✓ Removed directory: {path}")
                elif os.path.isfile(path):
                    os.remove(path)
                    print(f"✓ Removed file: {path}")
        
        print("✅ Cleanup completed")

def get_platform_requirements():
    """Get platform-specific requirements"""
    system = platform.system()
    return PLATFORM_REQUIREMENTS.get(system, [])

def read_readme():
    """Read README file for long description"""
    readme_path = Path("README.md")
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return DESCRIPTION

# Setup configuration
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    
    # Package discovery
    packages=find_packages(),
    include_package_data=True,
    
    # Entry points
    entry_points={
        "console_scripts": [
            "footlab=main_application:main",
            "footlab-config=config.advanced_settings:main",
            "footlab-test=tests.run_tests:main",
        ],
        "gui_scripts": [
            "footlab-gui=main_application:main",
        ]
    },
    
    # Dependencies
    python_requires=PYTHON_REQUIRES,
    install_requires=CORE_REQUIREMENTS + get_platform_requirements(),
    
    extras_require={
        "advanced": ADVANCED_REQUIREMENTS,
        "dev": DEV_REQUIREMENTS,
        "all": ADVANCED_REQUIREMENTS + DEV_REQUIREMENTS,
    },
    
    # Custom commands
    cmdclass={
        "preinstall": PreInstallCommand,
        "postinstall": PostInstallCommand,
        "test": TestCommand,
        "build_exe": BuildExecutableCommand,
        "clean": CleanCommand,
    },
    
    # Package data
    package_data={
        "footlab": [
            "assets/*",
            "config/*.json",
            "models/*.joblib",
            "models/*.json",
        ]
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    
    keywords="baropodometry gait-analysis clinical biomechanics pressure-mapping",
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Documentation": f"{URL}/docs",
    },
    
    # Platform requirements
    platforms=PLATFORMS,
)

# tests/conftest.py
"""
Pytest configuration and fixtures for FootLab testing
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock
import tempfile
import json

# Import FootLab components for testing
from core.modern_architecture import ModernFootLabCore, SystemConfig
from core.simulator import SimulatorSource
from config.advanced_settings import AdvancedFootLabConfig
from core.advanced_biomechanics import AdvancedGaitAnalyzer

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_directory():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_pressure_data():
    """Generate sample pressure data for testing"""
    n_samples = 1000
    n_sensors = 16
    
    # Generate realistic pressure patterns
    time_steps = np.linspace(0, 10, n_samples)  # 10 seconds
    
    left_pressures = []
    right_pressures = []
    
    for t in time_steps:
        # Simulate gait cycle
        gait_phase = (t * 1.2) % 2.0  # 1.2 Hz cadence
        
        if gait_phase < 0.6:  # Stance phase
            # Heel to toe pressure progression
            heel_pressure = max(0, 50 * (1 - gait_phase/0.6))
            toe_pressure = max(0, 40 * (gait_phase/0.6))
        else:  # Swing phase
            heel_pressure = 0
            toe_pressure = 0
        
        # Distribute pressure across sensors
        left_frame = np.zeros(n_sensors)
        right_frame = np.zeros(n_sensors)
        
        # Heel sensors (last 4 sensors)
        left_frame[-4:] = heel_pressure + np.random.normal(0, 5, 4)
        right_frame[-4:] = heel_pressure + np.random.normal(0, 5, 4)
        
        # Toe sensors (first 4 sensors)
        left_frame[:4] = toe_pressure + np.random.normal(0, 3, 4)
        right_frame[:4] = toe_pressure + np.random.normal(0, 3, 4)
        
        # Mid-foot sensors (middle sensors)
        left_frame[4:-4] = np.random.normal(0, 2, n_sensors-8)
        right_frame[4:-4] = np.random.normal(0, 2, n_sensors-8)
        
        # Ensure non-negative
        left_frame = np.maximum(left_frame, 0)
        right_frame = np.maximum(right_frame, 0)
        
        left_pressures.append(left_frame)
        right_pressures.append(right_frame)
    
    return {
        "timestamps": time_steps,
        "left_pressures": np.array(left_pressures),
        "right_pressures": np.array(right_pressures),
        "sampling_rate": 100.0
    }

@pytest.fixture
def mock_sensor_system():
    """Create mock sensor system for testing"""
    mock_system = Mock()
    mock_system.is_connected.return_value = True
    mock_system.get_statistics.return_value = {
        "packets_received": 1000,
        "packets_dropped": 5,
        "connection_state": "streaming"
    }
    return mock_system

@pytest.fixture
def test_config():
    """Create test configuration"""
    return SystemConfig(
        sampling_rate=100.0,
        n_sensors_per_foot=16,
        enable_advanced_analytics=True,
        enable_machine_learning=False  # Disable for faster tests
    )

@pytest.fixture
def footlab_system(test_config):
    """Create FootLab system for testing"""
    from core.modern_architecture import create_footlab_system
    return create_footlab_system(test_config)

@pytest.fixture
def gait_analyzer():
    """Create gait analyzer for testing"""
    return AdvancedGaitAnalyzer(sampling_rate=100.0)

@pytest.fixture
def advanced_config(temp_directory):
    """Create advanced configuration for testing"""
    return AdvancedFootLabConfig(str(temp_directory))

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        "patient_id": "TEST_001",
        "name": "Test Patient",
        "age": 35,
        "height": 175,
        "weight": 70,
        "gender": "Female",
        "diagnosis": "Healthy Control"
    }

@pytest.fixture
def sample_session_data(sample_patient_data, sample_pressure_data):
    """Sample session data for testing"""
    return {
        "session_id": "TEST_SESSION_001",
        "patient_info": sample_patient_data,
        "timestamp": "2024-01-01T10:00:00",
        "duration": 10.0,
        "sampling_rate": 100.0,
        "pressure_data_left": sample_pressure_data["left_pressures"].tolist(),
        "pressure_data_right": sample_pressure_data["right_pressures"].tolist(),
        "timestamps": sample_pressure_data["timestamps"].tolist()
    }

# tests/unit/test_core_system.py
"""
Unit tests for core system components
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from core.modern_architecture import ModernFootLabCore, SystemConfig, DataQuality

class TestModernFootLabCore:
    """Test the modern FootLab core system"""
    
    def test_system_initialization(self, test_config):
        """Test system initialization"""
        system = ModernFootLabCore(test_config)
        
        assert system.config == test_config
        assert system.event_bus is not None
        assert system.state_manager is not None
        assert system._running is False
    
    @pytest.mark.asyncio
    async def test_system_start_stop(self, footlab_system, mock_sensor_system):
        """Test system start and stop"""
        # Register mock data source
        footlab_system.register_data_source(mock_sensor_system)
        
        # Start system
        await footlab_system.start_system()
        assert footlab_system.state_manager.get_state('system_status') == 'running'
        
        # Stop system
        await footlab_system.stop_system()
        assert footlab_system.state_manager.get_state('system_status') == 'stopped'
    
    def test_component_registration(self, footlab_system, gait_analyzer):
        """Test component registration"""
        # Test analysis engine registration
        footlab_system.register_analysis_engine(gait_analyzer)
        assert footlab_system.analysis_engine == gait_analyzer
    
    def test_event_system(self, footlab_system):
        """Test event bus functionality"""
        event_received = []
        
        def event_handler(event_data):
            event_received.append(event_data)
        
        # Subscribe to event
        footlab_system.event_bus.subscribe('test_event', event_handler)
        
        # Publish event
        test_data = {'message': 'test'}
        footlab_system.event_bus.publish('test_event', test_data)
        
        assert len(event_received) == 1
        assert event_received[0] == test_data
    
    def test_state_management(self, footlab_system):
        """Test state manager"""
        state_manager = footlab_system.state_manager
        
        # Test state setting and getting
        state_manager.set_state('test_key', 'test_value')
        assert state_manager.get_state('test_key') == 'test_value'
        
        # Test state change notifications
        changes_received = []
        
        def state_change_handler(key, old_value, new_value):
            changes_received.append((key, old_value, new_value))
        
        state_manager.subscribe(state_change_handler)
        state_manager.set_state('test_key', 'new_value')
        
        assert len(changes_received) == 1
        assert changes_received[0] == ('test_key', 'test_value', 'new_value')

# tests/unit/test_gait_analysis.py
"""
Unit tests for gait analysis components
"""

import pytest
import numpy as np
from core.advanced_biomechanics import AdvancedGaitAnalyzer, StepMetrics

class TestGaitAnalyzer:
    """Test advanced gait analyzer"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = AdvancedGaitAnalyzer(sampling_rate=100.0)
        
        assert analyzer.sampling_rate == 100.0
        assert analyzer.dt == 0.01
        assert len(analyzer.pressure_history_L) == 0
        assert len(analyzer.pressure_history_R) == 0
    
    def test_sample_processing(self, gait_analyzer, sample_pressure_data):
        """Test sample processing"""
        timestamps = sample_pressure_data["timestamps"]
        left_pressures = sample_pressure_data["left_pressures"]
        right_pressures = sample_pressure_data["right_pressures"]
        
        # Process samples
        for i in range(min(100, len(timestamps))):  # Process first 100 samples
            gait_analyzer.add_sample(
                timestamp=timestamps[i],
                pressures_L=left_pressures[i],
                pressures_R=right_pressures[i],
                cop_L=(0.5, 0.3),  # Mock COP
                cop_R=(0.5, 0.3)
            )
        
        # Check that data was stored
        assert len(gait_analyzer.pressure_history_L) == 100
        assert len(gait_analyzer.pressure_history_R) == 100
        assert len(gait_analyzer.timestamps) == 100
    
    def test_step_detection(self, gait_analyzer, sample_pressure_data):
        """Test step detection algorithm"""
        timestamps = sample_pressure_data["timestamps"][:500]  # First 5 seconds
        left_pressures = sample_pressure_data["left_pressures"][:500]
        right_pressures = sample_pressure_data["right_pressures"][:500]
        
        # Process samples to trigger step detection
        for i in range(len(timestamps)):
            gait_analyzer.add_sample(
                timestamp=timestamps[i],
                pressures_L=left_pressures[i],
                pressures_R=right_pressures[i],
                cop_L=(0.5, 0.3),
                cop_R=(0.5, 0.3)
            )
        
        # Should detect some gait events
        assert len(gait_analyzer.gait_events) > 0
    
    def test_asymmetry_calculation(self, gait_analyzer):
        """Test asymmetry index calculation"""
        # Create mock steps with known asymmetry
        from core.advanced_biomechanics import StepMetrics
        
        # Perfect symmetry
        step_L = StepMetrics(
            stance_time=0.6, swing_time=0.4, step_time=1.0, cadence=60.0,
            peak_pressure=100.0, mean_pressure=50.0, pressure_time_integral=30.0,
            contact_area=15.0, cop_path_length=50.0, cop_velocity_mean=25.0,
            cop_velocity_max=40.0, cop_range_ml=10.0, cop_range_ap=30.0,
            heel_load_percent=40.0, forefoot_load_percent=60.0,
            medial_load_percent=45.0, lateral_load_percent=55.0,
            symmetry_index=0.0, impulse=30.0, load_rate=500.0, stability_margin=0.8
        )
        
        step_R = StepMetrics(
            stance_time=0.6, swing_time=0.4, step_time=1.0, cadence=60.0,
            peak_pressure=120.0, mean_pressure=60.0, pressure_time_integral=36.0,  # 20% higher
            contact_area=15.0, cop_path_length=50.0, cop_velocity_mean=25.0,
            cop_velocity_max=40.0, cop_range_ml=10.0, cop_range_ap=30.0,
            heel_load_percent=40.0, forefoot_load_percent=60.0,
            medial_load_percent=45.0, lateral_load_percent=55.0,
            symmetry_index=0.0, impulse=36.0, load_rate=600.0, stability_margin=0.8
        )
        
        gait_analyzer.steps_L = [step_L]
        gait_analyzer.steps_R = [step_R]
        
        asymmetry = gait_analyzer.calculate_asymmetry_indices()
        
        # Should detect asymmetry in peak pressure (20% difference)
        assert 'peak_pressure_asymmetry' in asymmetry
        assert asymmetry['peak_pressure_asymmetry'] > 15.0  # Should be ~18.2%

# tests/integration/test_system_integration.py
"""
Integration tests for complete system functionality
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

class TestSystemIntegration:
    """Test complete system integration"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, footlab_system, sample_session_data, temp_directory):
        """Test complete data acquisition and analysis workflow"""
        
        # Create simulator source
        from core.simulator import SimulatorSource
        simulator = SimulatorSource(n_sensors=16, freq=50, base_amp=60.0)  # Lower frequency for testing
        
        # Register components
        footlab_system.register_data_source(simulator)
        
        from core.advanced_biomechanics import create_gait_analyzer
        gait_analyzer = create_gait_analyzer(sampling_rate=50.0)
        footlab_system.register_analysis_engine(gait_analyzer)
        
        # Start system
        await footlab_system.start_system()
        
        # Start session
        session_id = footlab_system.start_session(sample_session_data)
        assert session_id is not None
        
        # Let system run for a short time
        await asyncio.sleep(2.0)
        
        # Check system status
        status = footlab_system.get_system_status()
        assert status['system_status'] == 'running'
        assert status['session_active'] is True
        
        # End session
        session_file = footlab_system.end_session()
        assert session_file is not None
        
        # Stop system
        await footlab_system.stop_system()
        
        assert footlab_system.state_manager.get_state('system_status') == 'stopped'
    
    def test_configuration_system(self, temp_directory):
        """Test configuration system integration"""
        from config.advanced_settings import AdvancedFootLabConfig
        
        config = AdvancedFootLabConfig(str(temp_directory))
        
        # Test configuration modification
        config.sensor.sampling_rate = 200.0
        config.analysis.enable_pathology_detection = False
        
        # Save configuration
        config.save_config("test_profile")
        
        # Load configuration
        config2 = AdvancedFootLabConfig(str(temp_directory))
        config2.load_config("test_profile")
        
        assert config2.sensor.sampling_rate == 200.0
        assert config2.analysis.enable_pathology_detection is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self, footlab_system):
        """Test system error handling"""
        
        # Test starting system without data source
        with pytest.raises(ValueError):
            await footlab_system.start_system()
        
        # Test invalid configuration
        invalid_config = SystemConfig(sampling_rate=-1)  # Invalid
        
        # System should handle invalid config gracefully
        system2 = ModernFootLabCore(invalid_config)
        assert system2.config.sampling_rate == -1  # Config stored as-is, validation separate

# tests/ui/test_user_interface.py
"""
UI tests using pytest-qt
"""

import pytest
from unittest.mock import Mock, patch
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt

# Import UI components
from main_application import StateOfTheArtFootLabApp
from config.advanced_settings import AdvancedFootLabConfig

@pytest.fixture
def app(qtbot):
    """Create application instance for testing"""
    test_app = StateOfTheArtFootLabApp()
    qtbot.addWidget(test_app)
    return test_app

class TestMainApplication:
    """Test main application UI"""
    
    def test_application_startup(self, app):
        """Test application starts correctly"""
        assert app.windowTitle() == "FootLab - State-of-the-Art Baropodometry System"
        assert app.system is not None
        assert app.control_panel is not None
        assert app.heatmap_view is not None
    
    def test_menu_actions(self, app, qtbot):
        """Test menu actions"""
        # Test File menu
        file_menu = None
        for action in app.menuBar().actions():
            if action.text() == "&File":
                file_menu = action.menu()
                break
        
        assert file_menu is not None
        
        # Find New Session action
        new_session_action = None
        for action in file_menu.actions():
            if "New Session" in action.text():
                new_session_action = action
                break
        
        assert new_session_action is not None
    
    @patch('main_application.PatientDialog')
    def test_patient_dialog(self, mock_dialog, app, qtbot):
        """Test patient selection dialog"""
        # Mock dialog response
        mock_dialog.return_value.exec.return_value = 1  # Accepted
        mock_dialog.return_value.get_patient_data.return_value = {
            "patient_id": "TEST_001",
            "name": "Test Patient"
        }
        
        # Trigger patient selection
        app.control_panel.select_patient()
        
        # Verify dialog was called
        mock_dialog.assert_called_once()
    
    def test_status_updates(self, app, qtbot):
        """Test status widget updates"""
        status_widget = app.status_widget
        
        # Test initial state
        assert status_widget.connection_status.text() == "Disconnected"
        
        # Simulate status update
        if app.system:
            app.system.state_manager.set_state('system_status', 'running')
            status_widget.update_status()
            
            # Status should update
            # Note: Actual text depends on update logic

class TestConfigurationUI:
    """Test configuration UI components"""
    
    def test_configuration_validation(self, temp_directory):
        """Test configuration validation"""
        config = AdvancedFootLabConfig(str(temp_directory))
        
        # Test valid configuration
        valid, errors = config.validate_config()
        assert valid is True
        assert len(errors) == 0
        
        # Test invalid configuration
        config.sensor.sampling_rate = -1
        valid, errors = config.validate_config()
        assert valid is False
        assert len(errors) > 0

# Performance tests
class TestPerformance:
    """Test system performance"""
    
    @pytest.mark.asyncio
    async def test_data_processing_performance(self, gait_analyzer, sample_pressure_data):
        """Test data processing performance"""
        import time
        
        timestamps = sample_pressure_data["timestamps"][:1000]  # 10 seconds of data
        left_pressures = sample_pressure_data["left_pressures"][:1000]
        right_pressures = sample_pressure_data["right_pressures"][:1000]
        
        start_time = time.time()
        
        # Process data
        for i in range(len(timestamps)):
            gait_analyzer.add_sample(
                timestamp=timestamps[i],
                pressures_L=left_pressures[i],
                pressures_R=right_pressures[i],
                cop_L=(0.5, 0.3),
                cop_R=(0.5, 0.3)
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 1000 samples in reasonable time
        assert processing_time < 5.0  # Less than 5 seconds
        
        # Calculate processing rate
        samples_per_second = len(timestamps) / processing_time
        assert samples_per_second > 100  # Should handle at least 100 samples/second

if __name__ == "__main__":
    pytest.main([__file__, "-v"])