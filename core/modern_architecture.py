# core/modern_architecture.py
"""
Modern architecture system for FootLab with dependency injection,
event-driven design, and plugin-based sensor support.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Protocol, TypeVar, Generic
from enum import Enum
import threading
import time
from contextlib import asynccontextmanager
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
SampleCallback = Callable[[Dict[str, Any]], None]
EventCallback = Callable[[Dict[str, Any]], None]

class AnalysisMode(Enum):
    """Analysis modes for different use cases"""
    REAL_TIME = "real_time"
    CLINICAL = "clinical" 
    RESEARCH = "research"
    REHABILITATION = "rehabilitation"

class DataQuality(Enum):
    """Data quality indicators"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"

@dataclass
class SystemConfig:
    """Central system configuration"""
    analysis_mode: AnalysisMode = AnalysisMode.REAL_TIME
    sampling_rate: float = 100.0
    buffer_size: int = 1000
    enable_real_time_processing: bool = True
    enable_advanced_analytics: bool = True
    enable_machine_learning: bool = False
    data_storage_path: Path = field(default_factory=lambda: Path("data"))
    log_level: str = "INFO"
    
    # Sensor configuration
    n_sensors_per_foot: int = 16
    sensor_type: str = "nurvv"  # "nurvv", "novel", "tekscan"
    
    # Processing configuration
    interpolation_method: str = "rbf"
    smoothing_enabled: bool = True
    peak_detection_enabled: bool = True
    
    # Clinical configuration
    pressure_units: str = "kPa"  # "kPa", "psi", "N/cm2"
    show_anatomical_zones: bool = False
    enable_asymmetry_analysis: bool = True

class EventBus:
    """Event bus for decoupled component communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[EventCallback]] = {}
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: str, callback: EventCallback):
        """Subscribe to an event type"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: EventCallback):
        """Unsubscribe from an event type"""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                except ValueError:
                    pass
    
    def publish(self, event_type: str, event_data: Dict[str, Any]):
        """Publish an event to all subscribers"""
        subscribers = []
        with self._lock:
            subscribers = self._subscribers.get(event_type, []).copy()
        
        for callback in subscribers:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type}: {e}")

class DataSource(Protocol):
    """Protocol for data sources"""
    
    def start(self, callback: SampleCallback) -> None:
        """Start data acquisition"""
        ...
    
    def stop(self) -> None:
        """Stop data acquisition"""
        ...
    
    def is_connected(self) -> bool:
        """Check if source is connected"""
        ...
    
    def get_info(self) -> Dict[str, Any]:
        """Get source information"""
        ...

class AnalysisEngine(Protocol):
    """Protocol for analysis engines"""
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample"""
        ...
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current analysis metrics"""
        ...
    
    def reset(self) -> None:
        """Reset analysis state"""
        ...

class Visualizer(Protocol):
    """Protocol for visualizers"""
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update visualization with new data"""
        ...
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure visualization settings"""
        ...

class DataLogger(Protocol):
    """Protocol for data logging"""
    
    def start_session(self, session_info: Dict[str, Any]) -> str:
        """Start a new logging session"""
        ...
    
    def log_sample(self, sample: Dict[str, Any]) -> None:
        """Log a data sample"""
        ...
    
    def end_session(self) -> Path:
        """End current session and return file path"""
        ...

class ServiceContainer:
    """Dependency injection container"""
    
    def __init__(self):
        self._services: Dict[type, Any] = {}
        self._singletons: Dict[type, Any] = {}
        self._factories: Dict[type, Callable] = {}
    
    def register_singleton(self, service_type: type, instance: Any):
        """Register a singleton service"""
        self._singletons[service_type] = instance
    
    def register_factory(self, service_type: type, factory: Callable):
        """Register a service factory"""
        self._factories[service_type] = factory
    
    def get(self, service_type: type) -> Any:
        """Get a service instance"""
        # Check singletons first
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check factories
        if service_type in self._factories:
            instance = self._factories[service_type]()
            return instance
        
        # Try direct instantiation
        try:
            return service_type()
        except Exception as e:
            raise ValueError(f"Cannot create instance of {service_type}: {e}")

class PluginManager:
    """Plugin manager for extensible sensor support"""
    
    def __init__(self):
        self._plugins: Dict[str, type] = {}
        self._plugin_configs: Dict[str, Dict] = {}
    
    def register_plugin(self, name: str, plugin_class: type, config: Dict = None):
        """Register a plugin"""
        self._plugins[name] = plugin_class
        self._plugin_configs[name] = config or {}
    
    def create_plugin(self, name: str, **kwargs) -> Any:
        """Create plugin instance"""
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} not found")
        
        plugin_class = self._plugins[name]
        plugin_config = self._plugin_configs[name].copy()
        plugin_config.update(kwargs)
        
        return plugin_class(**plugin_config)
    
    def list_plugins(self) -> List[str]:
        """List available plugins"""
        return list(self._plugins.keys())

class StateManager:
    """Centralized state management"""
    
    def __init__(self):
        self._state: Dict[str, Any] = {
            'system_status': 'idle',
            'data_quality': DataQuality.GOOD,
            'session_active': False,
            'connected_sensors': [],
            'current_metrics': {},
            'alerts': []
        }
        self._subscribers: List[Callable] = []
        self._lock = threading.RLock()
    
    def get_state(self, key: str = None) -> Any:
        """Get state value(s)"""
        with self._lock:
            if key is None:
                return self._state.copy()
            return self._state.get(key)
    
    def set_state(self, key: str, value: Any):
        """Set state value and notify subscribers"""
        with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            
            if old_value != value:
                self._notify_subscribers(key, old_value, value)
    
    def update_state(self, updates: Dict[str, Any]):
        """Update multiple state values"""
        with self._lock:
            for key, value in updates.items():
                self.set_state(key, value)
    
    def subscribe(self, callback: Callable[[str, Any, Any], None]):
        """Subscribe to state changes"""
        with self._lock:
            self._subscribers.append(callback)
    
    def _notify_subscribers(self, key: str, old_value: Any, new_value: Any):
        """Notify all subscribers of state changes"""
        for callback in self._subscribers:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Error notifying state subscriber: {e}")

class ModernFootLabCore:
    """Modern FootLab core system with advanced architecture"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        
        # Core components
        self.event_bus = EventBus()
        self.service_container = ServiceContainer()
        self.plugin_manager = PluginManager()
        self.state_manager = StateManager()
        
        # Service instances
        self.data_source: Optional[DataSource] = None
        self.analysis_engine: Optional[AnalysisEngine] = None
        self.visualizers: List[Visualizer] = []
        self.data_logger: Optional[DataLogger] = None
        
        # Processing state
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        
        self._setup_core_services()
        self._setup_event_handlers()
    
    def _setup_core_services(self):
        """Setup core services in dependency container"""
        
        # Register singleton services
        self.service_container.register_singleton(EventBus, self.event_bus)
        self.service_container.register_singleton(StateManager, self.state_manager)
        self.service_container.register_singleton(SystemConfig, self.config)
        
        # Register service factories
        self.service_container.register_factory(
            'enhanced_heatmap_view',
            lambda: self._create_enhanced_heatmap()
        )
        
        self.service_container.register_factory(
            'gait_analyzer',
            lambda: self._create_gait_analyzer()
        )
    
    def _create_enhanced_heatmap(self):
        """Factory for enhanced heatmap view"""
        from ui.enhanced_heatmap_view import StateOfTheArtHeatmapView
        return StateOfTheArtHeatmapView(
            grid_w=128, 
            grid_h=160, 
            n_sensors=self.config.n_sensors_per_foot
        )
    
    def _create_gait_analyzer(self):
        """Factory for gait analyzer"""
        from core.advanced_biomechanics import create_gait_analyzer
        return create_gait_analyzer(sampling_rate=self.config.sampling_rate)
    
    def _setup_event_handlers(self):
        """Setup core event handlers"""
        
        self.event_bus.subscribe('sample_received', self._handle_sample)
        self.event_bus.subscribe('system_error', self._handle_error)
        self.event_bus.subscribe('data_quality_changed', self._handle_quality_change)
    
    def _handle_sample(self, event_data: Dict[str, Any]):
        """Handle incoming sample data"""
        sample = event_data.get('sample', {})
        
        # Update state
        self.state_manager.set_state('last_sample_time', time.time())
        
        # Process with analysis engine
        if self.analysis_engine:
            try:
                analysis_result = self.analysis_engine.process_sample(sample)
                self.event_bus.publish('analysis_complete', {
                    'sample': sample,
                    'analysis': analysis_result
                })
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                self.event_bus.publish('system_error', {
                    'component': 'analysis_engine',
                    'error': str(e)
                })
        
        # Update visualizers
        for visualizer in self.visualizers:
            try:
                visualizer.update(sample)
            except Exception as e:
                logger.error(f"Visualization error: {e}")
        
        # Log data
        if self.data_logger and self.state_manager.get_state('session_active'):
            try:
                self.data_logger.log_sample(sample)
            except Exception as e:
                logger.error(f"Logging error: {e}")
    
    def _handle_error(self, event_data: Dict[str, Any]):
        """Handle system errors"""
        error_info = {
            'timestamp': time.time(),
            'component': event_data.get('component', 'unknown'),
            'error': event_data.get('error', 'Unknown error'),
            'severity': event_data.get('severity', 'error')
        }
        
        # Add to alerts
        current_alerts = self.state_manager.get_state('alerts') or []
        current_alerts.append(error_info)
        
        # Keep only last 10 alerts
        if len(current_alerts) > 10:
            current_alerts = current_alerts[-10:]
        
        self.state_manager.set_state('alerts', current_alerts)
        
        logger.error(f"System error in {error_info['component']}: {error_info['error']}")
    
    def _handle_quality_change(self, event_data: Dict[str, Any]):
        """Handle data quality changes"""
        new_quality = event_data.get('quality', DataQuality.GOOD)
        self.state_manager.set_state('data_quality', new_quality)
        
        # Emit warning for poor quality
        if new_quality in [DataQuality.POOR, DataQuality.UNUSABLE]:
            self.event_bus.publish('system_warning', {
                'message': f'Data quality degraded: {new_quality.value}',
                'severity': 'warning'
            })
    
    def register_data_source(self, source: DataSource):
        """Register data source"""
        self.data_source = source
        self.state_manager.set_state('data_source_connected', True)
        logger.info(f"Data source registered: {type(source).__name__}")
    
    def register_analysis_engine(self, engine: AnalysisEngine):
        """Register analysis engine"""
        self.analysis_engine = engine
        logger.info(f"Analysis engine registered: {type(engine).__name__}")
    
    def register_visualizer(self, visualizer: Visualizer):
        """Register visualizer"""
        self.visualizers.append(visualizer)
        logger.info(f"Visualizer registered: {type(visualizer).__name__}")
    
    def register_data_logger(self, logger_instance: DataLogger):
        """Register data logger"""
        self.data_logger = logger_instance
        logger.info(f"Data logger registered: {type(logger_instance).__name__}")
    
    async def start_system(self):
        """Start the complete system"""
        if self._running:
            logger.warning("System already running")
            return
        
        logger.info("Starting FootLab system...")
        
        # Validate configuration
        if not self.data_source:
            raise ValueError("No data source registered")
        
        # Update state
        self.state_manager.set_state('system_status', 'starting')
        
        try:
            # Start data source
            if hasattr(self.data_source, 'start_async'):
                await self.data_source.start_async(self._handle_raw_sample)
            else:
                self.data_source.start(self._handle_raw_sample)
            
            # Start processing task
            self._processing_task = asyncio.create_task(self._processing_loop())
            
            self._running = True
            self.state_manager.set_state('system_status', 'running')
            
            logger.info("FootLab system started successfully")
            
        except Exception as e:
            self.state_manager.set_state('system_status', 'error')
            self.event_bus.publish('system_error', {
                'component': 'system_startup',
                'error': str(e),
                'severity': 'critical'
            })
            raise
    
    async def stop_system(self):
        """Stop the complete system"""
        if not self._running:
            logger.warning("System not running")
            return
        
        logger.info("Stopping FootLab system...")
        
        self._running = False
        self.state_manager.set_state('system_status', 'stopping')
        
        # Stop data source
        if self.data_source:
            try:
                if hasattr(self.data_source, 'stop_async'):
                    await self.data_source.stop_async()
                else:
                    self.data_source.stop()
            except Exception as e:
                logger.error(f"Error stopping data source: {e}")
        
        # Cancel processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        # End logging session
        if self.data_logger and self.state_manager.get_state('session_active'):
            try:
                session_file = self.data_logger.end_session()
                logger.info(f"Session saved to: {session_file}")
            except Exception as e:
                logger.error(f"Error ending session: {e}")
        
        self.state_manager.set_state('system_status', 'stopped')
        self.state_manager.set_state('session_active', False)
        
        logger.info("FootLab system stopped")
    
    def _handle_raw_sample(self, raw_sample: Dict[str, Any]):
        """Handle raw sample from data source"""
        # Add timestamp if not present
        if 't_ms' not in raw_sample:
            raw_sample['t_ms'] = int(time.time() * 1000)
        
        # Validate sample
        if self._validate_sample(raw_sample):
            self.event_bus.publish('sample_received', {'sample': raw_sample})
        else:
            self.event_bus.publish('sample_invalid', {'sample': raw_sample})
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate incoming sample"""
        required_keys = ['t_ms', 'left', 'right']
        
        for key in required_keys:
            if key not in sample:
                return False
        
        # Validate pressure arrays
        left_pressures = sample.get('left', [])
        right_pressures = sample.get('right', [])
        
        expected_sensors = self.config.n_sensors_per_foot
        
        if (len(left_pressures) != expected_sensors or 
            len(right_pressures) != expected_sensors):
            return False
        
        return True
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self._running:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Check data quality
                await self._check_data_quality()
                
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.event_bus.publish('system_error', {
                    'component': 'processing_loop',
                    'error': str(e)
                })
                await asyncio.sleep(1.0)  # Back off on error
    
    async def _update_system_metrics(self):
        """Update system-wide metrics"""
        current_time = time.time()
        last_sample_time = self.state_manager.get_state('last_sample_time')
        
        # Check if we're receiving data
        if last_sample_time:
            time_since_last = current_time - last_sample_time
            receiving_data = time_since_last < 2.0  # 2 second timeout
            self.state_manager.set_state('receiving_data', receiving_data)
        
        # Update analysis metrics if available
        if self.analysis_engine:
            try:
                current_metrics = self.analysis_engine.get_current_metrics()
                self.state_manager.set_state('current_metrics', current_metrics)
            except Exception as e:
                logger.error(f"Error getting analysis metrics: {e}")
    
    async def _check_data_quality(self):
        """Check and update data quality"""
        # This would implement sophisticated data quality assessment
        # For now, just check if we're receiving data
        receiving_data = self.state_manager.get_state('receiving_data', False)
        
        if receiving_data:
            current_quality = DataQuality.GOOD
        else:
            current_quality = DataQuality.POOR
        
        # Only emit event if quality changed
        old_quality = self.state_manager.get_state('data_quality')
        if old_quality != current_quality:
            self.event_bus.publish('data_quality_changed', {
                'quality': current_quality
            })
    
    def start_session(self, session_info: Dict[str, Any] = None):
        """Start a data collection session"""
        if not self.data_logger:
            raise ValueError("No data logger registered")
        
        session_info = session_info or {}
        session_id = self.data_logger.start_session(session_info)
        
        self.state_manager.set_state('session_active', True)
        self.state_manager.set_state('current_session_id', session_id)
        
        self.event_bus.publish('session_started', {
            'session_id': session_id,
            'session_info': session_info
        })
        
        logger.info(f"Session started: {session_id}")
        return session_id
    
    def end_session(self) -> Optional[Path]:
        """End current session"""
        if not self.state_manager.get_state('session_active'):
            logger.warning("No active session to end")
            return None
        
        if not self.data_logger:
            logger.error("No data logger available")
            return None
        
        try:
            session_file = self.data_logger.end_session()
            self.state_manager.set_state('session_active', False)
            self.state_manager.set_state('current_session_id', None)
            
            self.event_bus.publish('session_ended', {
                'session_file': str(session_file)
            })
            
            logger.info(f"Session ended: {session_file}")
            return session_file
            
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        state = self.state_manager.get_state()
        
        # Add additional computed status
        status = {
            **state,
            'uptime': time.time() - getattr(self, '_start_time', time.time()),
            'plugins_available': self.plugin_manager.list_plugins(),
            'visualizers_count': len(self.visualizers),
            'components_registered': {
                'data_source': self.data_source is not None,
                'analysis_engine': self.analysis_engine is not None,
                'data_logger': self.data_logger is not None,
                'visualizers': len(self.visualizers)
            }
        }
        
        return status
    
    @asynccontextmanager
    async def session_context(self, session_info: Dict[str, Any] = None):
        """Context manager for sessions"""
        session_id = None
        try:
            session_id = self.start_session(session_info)
            yield session_id
        finally:
            if session_id:
                self.end_session()

# Convenience factory function
def create_footlab_system(config: SystemConfig = None) -> ModernFootLabCore:
    """Create a fully configured FootLab system"""
    
    config = config or SystemConfig()
    system = ModernFootLabCore(config)
    
    # Register default plugins
    from core.simulator import SimulatorSource
    system.plugin_manager.register_plugin('simulator', SimulatorSource)
    
    # You would register other plugins here:
    # system.plugin_manager.register_plugin('nurvv_ble', NurvvBleSource)
    # system.plugin_manager.register_plugin('novel_bluetooth', NovelBluetoothSource)
    
    return system

# Example usage and integration classes
class ModernMainWindow:
    """Example integration with modern architecture"""
    
    def __init__(self, config: SystemConfig = None):
        self.system = create_footlab_system(config)
        self._setup_components()
        self._setup_ui_event_handlers()
    
    def _setup_components(self):
        """Setup system components"""
        
        # Create and register components
        heatmap_view = self.system.service_container.get('enhanced_heatmap_view')
        gait_analyzer = self.system.service_container.get('gait_analyzer')
        
        # Register with system
        self.system.register_visualizer(heatmap_view)
        self.system.register_analysis_engine(gait_analyzer)
        
        # Create data source (would be selected by user)
        simulator = self.system.plugin_manager.create_plugin('simulator', 
                                                           n_sensors=16, 
                                                           freq=100)
        self.system.register_data_source(simulator)
    
    def _setup_ui_event_handlers(self):
        """Setup UI-specific event handlers"""
        
        # Subscribe to system events
        self.system.event_bus.subscribe('system_error', self._handle_ui_error)
        self.system.event_bus.subscribe('session_started', self._handle_session_started)
        self.system.event_bus.subscribe('analysis_complete', self._handle_analysis_update)
        
        # Subscribe to state changes
        self.system.state_manager.subscribe(self._handle_state_change)
    
    def _handle_ui_error(self, event_data: Dict[str, Any]):
        """Handle errors in UI"""
        # Show error dialog or status update
        pass
    
    def _handle_session_started(self, event_data: Dict[str, Any]):
        """Handle session start in UI"""
        # Update UI to show session is active
        pass
    
    def _handle_analysis_update(self, event_data: Dict[str, Any]):
        """Handle analysis updates"""
        # Update metrics display
        pass
    
    def _handle_state_change(self, key: str, old_value: Any, new_value: Any):
        """Handle state changes"""
        if key == 'system_status':
            # Update status indicator
            pass
        elif key == 'data_quality':
            # Update quality indicator
            pass
    
    async def start(self):
        """Start the application"""
        await self.system.start_system()
    
    async def stop(self):
        """Stop the application"""
        await self.system.stop_system()

# Usage example
async def example_usage():
    """Example of how to use the modern architecture"""
    
    # Create system with custom config
    config = SystemConfig(
        analysis_mode=AnalysisMode.CLINICAL,
        sampling_rate=120.0,
        enable_advanced_analytics=True,
        n_sensors_per_foot=16
    )
    
    system = create_footlab_system(config)
    
    # Setup components
    simulator = system.plugin_manager.create_plugin('simulator')
    system.register_data_source(simulator)
    
    # Start system
    await system.start_system()
    
    # Start a session
    async with system.session_context({'patient_id': 'test_001'}):
        # System is collecting data
        await asyncio.sleep(10)  # Collect for 10 seconds
        
        # Get current status
        status = system.get_system_status()
        print(f"System status: {status['system_status']}")
        print(f"Data quality: {status['data_quality']}")
    
    # Stop system
    await system.stop_system()

if __name__ == "__main__":
    asyncio.run(example_usage())