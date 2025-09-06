# calibration/advanced_calibration.py
"""
Advanced multi-modal calibration system for FootLab
Includes force plate validation, temporal synchronization, and spatial alignment
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import json
from scipy import optimize, signal, spatial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CalibrationType(Enum):
    """Types of calibration procedures"""
    BASELINE_ZERO = "baseline_zero"
    FORCE_SCALING = "force_scaling"
    SPATIAL_ALIGNMENT = "spatial_alignment"
    TEMPORAL_SYNC = "temporal_sync"
    CROSS_VALIDATION = "cross_validation"
    FACTORY_RESET = "factory_reset"

class CalibrationStatus(Enum):
    """Calibration procedure status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class CalibrationPoint:
    """Single calibration data point"""
    timestamp: float
    reference_force: np.ndarray  # From force plate or known weights
    sensor_readings: np.ndarray  # Raw sensor values
    temperature: float = 25.0
    humidity: float = 50.0
    pressure: float = 1013.25  # atmospheric pressure
    notes: str = ""

@dataclass
class CalibrationSession:
    """Complete calibration session data"""
    session_id: str
    calibration_type: CalibrationType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: CalibrationStatus = CalibrationStatus.NOT_STARTED
    
    # Data points
    calibration_points: List[CalibrationPoint] = field(default_factory=list)
    
    # Results
    calibration_matrix: Optional[np.ndarray] = None
    offset_vector: Optional[np.ndarray] = None
    scale_factors: Optional[np.ndarray] = None
    correlation_coefficient: float = 0.0
    rms_error: float = 0.0
    max_error: float = 0.0
    
    # Metadata
    sensor_configuration: Dict[str, Any] = field(default_factory=dict)
    environmental_conditions: Dict[str, float] = field(default_factory=dict)
    operator: str = ""
    notes: str = ""

class ForceReference:
    """Reference force measurement system interface"""
    
    def __init__(self, reference_type: str = "known_weights"):
        self.reference_type = reference_type
        self.is_connected = False
        self.calibration_factor = 1.0
    
    async def connect(self) -> bool:
        """Connect to reference system"""
        if self.reference_type == "known_weights":
            # Manual calibration with known weights
            self.is_connected = True
            return True
        elif self.reference_type == "force_plate":
            # Connect to external force plate
            return await self._connect_force_plate()
        else:
            logger.error(f"Unknown reference type: {self.reference_type}")
            return False
    
    async def _connect_force_plate(self) -> bool:
        """Connect to external force plate system"""
        try:
            # This would implement actual force plate connection
            # For now, simulate connection
            await asyncio.sleep(1)
            self.is_connected = True
            logger.info("Connected to force plate reference system")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to force plate: {e}")
            return False
    
    async def get_reference_force(self) -> np.ndarray:
        """Get current reference force measurement"""
        if not self.is_connected:
            raise RuntimeError("Reference system not connected")
        
        if self.reference_type == "known_weights":
            # Return zero for manual calibration
            return np.array([0.0, 0.0, 0.0])  # Fx, Fy, Fz
        elif self.reference_type == "force_plate":
            return await self._read_force_plate()
    
    async def _read_force_plate(self) -> np.ndarray:
        """Read force plate data"""
        # Simulate force plate reading
        await asyncio.sleep(0.01)
        # Return simulated 3D force vector
        return np.array([
            np.random.normal(0, 5),    # Fx
            np.random.normal(0, 5),    # Fy  
            np.random.normal(500, 50)  # Fz (vertical)
        ])

class SpatialCalibrator:
    """Spatial alignment and coordinate system calibration"""
    
    def __init__(self, n_sensors: int = 16):
        self.n_sensors = n_sensors
        self.anatomical_landmarks = {}
        self.sensor_coordinates = np.zeros((n_sensors, 2))
        
    def define_anatomical_landmarks(self, landmarks: Dict[str, Tuple[float, float]]):
        """Define anatomical landmark positions"""
        self.anatomical_landmarks = landmarks
        logger.info(f"Defined {len(landmarks)} anatomical landmarks")
    
    def calibrate_sensor_positions(self, 
                                 reference_points: List[Tuple[str, np.ndarray]], 
                                 sensor_responses: List[np.ndarray]) -> Dict[str, Any]:
        """
        Calibrate sensor positions using known reference points
        
        Args:
            reference_points: List of (landmark_name, position) pairs
            sensor_responses: Corresponding sensor responses for each reference point
        """
        
        if len(reference_points) != len(sensor_responses):
            raise ValueError("Number of reference points must match sensor responses")
        
        # Extract landmark positions and sensor data
        reference_positions = np.array([pos for _, pos in reference_points])
        responses = np.array(sensor_responses)
        
        # Find optimal sensor positions using least squares
        calibrated_positions = []
        calibration_errors = []
        
        for sensor_idx in range(self.n_sensors):
            sensor_data = responses[:, sensor_idx]
            
            # Use inverse distance weighting to estimate sensor position
            weights = sensor_data / np.sum(sensor_data) if np.sum(sensor_data) > 0 else np.ones(len(sensor_data))
            
            # Weighted centroid
            estimated_position = np.average(reference_positions, weights=weights, axis=0)
            calibrated_positions.append(estimated_position)
            
            # Calculate calibration error
            predicted_responses = self._calculate_response_model(estimated_position, reference_positions)
            error = np.mean(np.abs(predicted_responses - sensor_data))
            calibration_errors.append(error)
        
        self.sensor_coordinates = np.array(calibrated_positions)
        
        return {
            "sensor_positions": self.sensor_coordinates,
            "calibration_errors": calibration_errors,
            "mean_error": np.mean(calibration_errors),
            "max_error": np.max(calibration_errors),
            "success": np.max(calibration_errors) < 10.0  # mm threshold
        }
    
    def _calculate_response_model(self, sensor_pos: np.ndarray, reference_positions: np.ndarray) -> np.ndarray:
        """Calculate expected sensor response using inverse distance model"""
        distances = np.linalg.norm(reference_positions - sensor_pos[None, :], axis=1)
        # Avoid division by zero
        distances = np.maximum(distances, 1e-6)
        # Simple inverse distance model
        responses = 1.0 / (distances ** 2)
        return responses / np.sum(responses) * 100  # Normalize to 0-100 range
    
    def validate_spatial_calibration(self, test_points: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Validate spatial calibration using test points"""
        errors = []
        
        for true_position, sensor_response in test_points:
            # Estimate position from sensor response
            estimated_position = self._estimate_position_from_response(sensor_response)
            error = np.linalg.norm(estimated_position - true_position)
            errors.append(error)
        
        return {
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
            "max_error": np.max(errors),
            "rmse": np.sqrt(np.mean(np.array(errors)**2))
        }
    
    def _estimate_position_from_response(self, sensor_response: np.ndarray) -> np.ndarray:
        """Estimate load position from sensor response using calibrated positions"""
        total_response = np.sum(sensor_response)
        if total_response < 1e-6:
            return np.array([0.0, 0.0])
        
        weights = sensor_response / total_response
        estimated_position = np.sum(self.sensor_coordinates * weights[:, None], axis=0)
        return estimated_position

class TemporalSynchronizer:
    """Temporal synchronization and timing calibration"""
    
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        self.time_offsets = {}
        self.sync_quality_history = []
    
    async def calibrate_synchronization(self, 
                                      sensor_systems: Dict[str, Any],
                                      duration: float = 10.0) -> Dict[str, Any]:
        """
        Calibrate temporal synchronization between multiple sensor systems
        
        Args:
            sensor_systems: Dictionary of sensor system identifiers and objects
            duration: Calibration duration in seconds
        """
        
        logger.info(f"Starting temporal synchronization calibration for {duration}s")
        
        # Collect synchronized data from all systems
        sync_data = {}
        start_time = time.time()
        
        # Start data collection from all systems
        collection_tasks = []
        for system_id, system in sensor_systems.items():
            task = asyncio.create_task(
                self._collect_sync_data(system_id, system, duration)
            )
            collection_tasks.append(task)
        
        # Wait for all data collection to complete
        results = await asyncio.gather(*collection_tasks)
        
        # Organize collected data
        for system_id, data in zip(sensor_systems.keys(), results):
            sync_data[system_id] = data
        
        # Calculate cross-correlation and time offsets
        reference_system = list(sensor_systems.keys())[0]  # Use first system as reference
        
        for system_id in sensor_systems.keys():
            if system_id != reference_system:
                offset = self._calculate_time_offset(
                    sync_data[reference_system],
                    sync_data[system_id]
                )
                self.time_offsets[system_id] = offset
        
        # Validate synchronization quality
        sync_quality = self._validate_synchronization(sync_data)
        
        return {
            "time_offsets": self.time_offsets.copy(),
            "sync_quality": sync_quality,
            "calibration_duration": duration,
            "reference_system": reference_system,
            "success": sync_quality > 0.8
        }
    
    async def _collect_sync_data(self, system_id: str, system: Any, duration: float) -> Dict[str, Any]:
        """Collect synchronization data from a single system"""
        
        timestamps = []
        signals = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get current timestamp and signal
            current_time = time.time()
            
            # Get signal data (this would be actual sensor reading)
            if hasattr(system, 'get_current_signal'):
                signal_data = await system.get_current_signal()
            else:
                # Simulate signal data
                signal_data = np.random.normal(0, 1) + np.sin(2 * np.pi * current_time)
            
            timestamps.append(current_time)
            signals.append(signal_data)
            
            await asyncio.sleep(1.0 / self.sampling_rate)
        
        return {
            "system_id": system_id,
            "timestamps": np.array(timestamps),
            "signals": np.array(signals)
        }
    
    def _calculate_time_offset(self, reference_data: Dict, target_data: Dict) -> float:
        """Calculate time offset using cross-correlation"""
        
        ref_signal = reference_data["signals"]
        target_signal = target_data["signals"]
        
        # Ensure signals have same length
        min_length = min(len(ref_signal), len(target_signal))
        ref_signal = ref_signal[:min_length]
        target_signal = target_signal[:min_length]
        
        # Calculate cross-correlation
        correlation = signal.correlate(ref_signal, target_signal, mode='full')
        
        # Find peak correlation
        peak_idx = np.argmax(np.abs(correlation))
        
        # Convert to time offset
        time_offset = (peak_idx - len(target_signal) + 1) / self.sampling_rate
        
        return time_offset
    
    def _validate_synchronization(self, sync_data: Dict) -> float:
        """Validate synchronization quality"""
        
        if len(sync_data) < 2:
            return 1.0  # Perfect sync if only one system
        
        systems = list(sync_data.keys())
        correlations = []
        
        # Calculate pairwise correlations
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                signal1 = sync_data[systems[i]]["signals"]
                signal2 = sync_data[systems[j]]["signals"]
                
                # Apply time offset correction
                offset = self.time_offsets.get(systems[j], 0.0)
                if offset != 0:
                    # Shift signal2 by offset (simplified)
                    shift_samples = int(offset * self.sampling_rate)
                    if shift_samples > 0:
                        signal2 = signal2[shift_samples:]
                        signal1 = signal1[:-shift_samples]
                    elif shift_samples < 0:
                        signal1 = signal1[-shift_samples:]
                        signal2 = signal2[:shift_samples]
                
                # Calculate correlation
                if len(signal1) > 0 and len(signal2) > 0:
                    min_len = min(len(signal1), len(signal2))
                    corr = np.corrcoef(signal1[:min_len], signal2[:min_len])[0, 1]
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0

class AdvancedCalibrationSystem:
    """Complete advanced calibration system"""
    
    def __init__(self, sensor_config: Dict[str, Any]):
        self.sensor_config = sensor_config
        self.n_sensors = sensor_config.get("n_sensors", 16)
        
        # Calibration components
        self.force_reference = ForceReference()
        self.spatial_calibrator = SpatialCalibrator(self.n_sensors)
        self.temporal_synchronizer = TemporalSynchronizer()
        
        # Calibration sessions
        self.calibration_history: List[CalibrationSession] = []
        self.current_calibration: Optional[CalibrationSession] = None
        
        # Calibration results
        self.current_calibration_matrix = np.eye(self.n_sensors)
        self.current_offset_vector = np.zeros(self.n_sensors)
        self.calibration_valid = False
        self.calibration_timestamp = None
    
    async def run_full_calibration(self, 
                                 calibration_config: Dict[str, Any],
                                 progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run complete multi-modal calibration procedure
        
        Args:
            calibration_config: Configuration for calibration procedure
            progress_callback: Optional callback for progress updates
        """
        
        logger.info("Starting full system calibration")
        
        results = {
            "overall_success": True,
            "calibration_results": {},
            "errors": []
        }
        
        try:
            # Step 1: Baseline zero calibration
            if progress_callback:
                progress_callback("baseline_zero", 0, "Starting baseline calibration")
            
            baseline_result = await self.calibrate_baseline_zero(
                duration=calibration_config.get("baseline_duration", 5.0)
            )
            results["calibration_results"]["baseline_zero"] = baseline_result
            
            if not baseline_result.get("success", False):
                results["overall_success"] = False
                results["errors"].append("Baseline calibration failed")
            
            # Step 2: Force scaling calibration
            if progress_callback:
                progress_callback("force_scaling", 25, "Performing force scaling calibration")
            
            if calibration_config.get("enable_force_calibration", True):
                force_result = await self.calibrate_force_scaling(
                    calibration_weights=calibration_config.get("calibration_weights", [100, 200, 500])
                )
                results["calibration_results"]["force_scaling"] = force_result
                
                if not force_result.get("success", False):
                    results["overall_success"] = False
                    results["errors"].append("Force scaling calibration failed")
            
            # Step 3: Spatial alignment calibration
            if progress_callback:
                progress_callback("spatial_alignment", 50, "Calibrating spatial alignment")
            
            if calibration_config.get("enable_spatial_calibration", True):
                spatial_result = await self.calibrate_spatial_alignment()
                results["calibration_results"]["spatial_alignment"] = spatial_result
                
                if not spatial_result.get("success", False):
                    results["overall_success"] = False
                    results["errors"].append("Spatial calibration failed")
            
            # Step 4: Temporal synchronization
            if progress_callback:
                progress_callback("temporal_sync", 75, "Synchronizing temporal alignment")
            
            if calibration_config.get("enable_temporal_sync", True):
                sync_result = await self.calibrate_temporal_sync(
                    duration=calibration_config.get("sync_duration", 10.0)
                )
                results["calibration_results"]["temporal_sync"] = sync_result
                
                if not sync_result.get("success", False):
                    results["overall_success"] = False
                    results["errors"].append("Temporal synchronization failed")
            
            # Step 5: Cross-validation
            if progress_callback:
                progress_callback("validation", 90, "Running validation tests")
            
            validation_result = await self.validate_calibration()
            results["calibration_results"]["validation"] = validation_result
            
            if not validation_result.get("success", False):
                results["overall_success"] = False
                results["errors"].append("Calibration validation failed")
            
            # Finalize calibration
            if results["overall_success"]:
                self.calibration_valid = True
                self.calibration_timestamp = datetime.now()
                logger.info("Full system calibration completed successfully")
            else:
                logger.error(f"Calibration failed: {results['errors']}")
            
            if progress_callback:
                status = "Calibration completed successfully" if results["overall_success"] else "Calibration failed"
                progress_callback("complete", 100, status)
            
        except Exception as e:
            logger.error(f"Calibration procedure failed: {e}")
            results["overall_success"] = False
            results["errors"].append(f"Calibration procedure exception: {str(e)}")
        
        return results
    
    async def calibrate_baseline_zero(self, duration: float = 5.0) -> Dict[str, Any]:
        """Calibrate baseline zero offset"""
        
        logger.info(f"Starting baseline zero calibration for {duration}s")
        
        session = CalibrationSession(
            session_id=f"baseline_{int(time.time())}",
            calibration_type=CalibrationType.BASELINE_ZERO,
            start_time=datetime.now(),
            status=CalibrationStatus.IN_PROGRESS
        )
        
        self.current_calibration = session
        
        try:
            # Collect baseline data
            baseline_samples = []
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Get sensor readings (simulate for now)
                sensor_reading = np.random.normal(0, 2, self.n_sensors)  # Simulate noise
                baseline_samples.append(sensor_reading)
                
                await asyncio.sleep(0.01)  # 100 Hz sampling
            
            baseline_samples = np.array(baseline_samples)
            
            # Calculate baseline offsets
            baseline_offsets = np.mean(baseline_samples, axis=0)
            baseline_noise = np.std(baseline_samples, axis=0)
            
            # Update calibration
            self.current_offset_vector = baseline_offsets
            
            session.offset_vector = baseline_offsets
            session.status = CalibrationStatus.COMPLETED
            session.end_time = datetime.now()
            session.rms_error = np.mean(baseline_noise)
            
            self.calibration_history.append(session)
            
            logger.info(f"Baseline calibration completed. Mean offset: {np.mean(baseline_offsets):.3f}")
            
            return {
                "success": True,
                "baseline_offsets": baseline_offsets.tolist(),
                "noise_levels": baseline_noise.tolist(),
                "mean_noise": float(np.mean(baseline_noise)),
                "max_noise": float(np.max(baseline_noise)),
                "duration": duration
            }
            
        except Exception as e:
            session.status = CalibrationStatus.FAILED
            session.notes = str(e)
            logger.error(f"Baseline calibration failed: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def calibrate_force_scaling(self, calibration_weights: List[float]) -> Dict[str, Any]:
        """Calibrate force scaling using known weights"""
        
        logger.info(f"Starting force scaling calibration with weights: {calibration_weights}")
        
        session = CalibrationSession(
            session_id=f"force_scaling_{int(time.time())}",
            calibration_type=CalibrationType.FORCE_SCALING,
            start_time=datetime.now(),
            status=CalibrationStatus.IN_PROGRESS
        )
        
        self.current_calibration = session
        
        try:
            calibration_points = []
            
            for weight in calibration_weights:
                logger.info(f"Place {weight}g weight and press Enter...")
                # In real implementation, this would wait for user input
                await asyncio.sleep(2)  # Simulate user placing weight
                
                # Collect data for this weight
                samples = []
                for _ in range(100):  # 1 second at 100 Hz
                    # Simulate sensor reading with weight
                    base_reading = weight * 0.01  # Convert to appropriate units
                    sensor_reading = np.random.normal(base_reading, base_reading * 0.05, self.n_sensors)
                    samples.append(sensor_reading)
                    await asyncio.sleep(0.01)
                
                mean_reading = np.mean(samples, axis=0)
                
                calibration_point = CalibrationPoint(
                    timestamp=time.time(),
                    reference_force=np.array([0, 0, weight]),  # Weight in Z direction
                    sensor_readings=mean_reading
                )
                
                calibration_points.append(calibration_point)
                session.calibration_points.append(calibration_point)
            
            # Calculate scaling factors
            reference_forces = np.array([cp.reference_force[2] for cp in calibration_points])  # Z component
            sensor_totals = np.array([np.sum(cp.sensor_readings) for cp in calibration_points])
            
            # Linear regression to find scaling factor
            if len(reference_forces) > 1:
                slope, intercept, r_value, _, _ = scipy.stats.linregress(sensor_totals, reference_forces)
                scaling_factor = slope
                correlation = r_value**2
            else:
                scaling_factor = reference_forces[0] / sensor_totals[0] if sensor_totals[0] > 0 else 1.0
                correlation = 1.0
            
            # Update calibration matrix
            self.current_calibration_matrix = np.eye(self.n_sensors) * scaling_factor
            
            session.scale_factors = np.full(self.n_sensors, scaling_factor)
            session.correlation_coefficient = correlation
            session.status = CalibrationStatus.COMPLETED
            session.end_time = datetime.now()
            
            self.calibration_history.append(session)
            
            logger.info(f"Force scaling completed. Scaling factor: {scaling_factor:.6f}, R²: {correlation:.3f}")
            
            return {
                "success": True,
                "scaling_factor": float(scaling_factor),
                "correlation": float(correlation),
                "calibration_points": len(calibration_points)
            }
            
        except Exception as e:
            session.status = CalibrationStatus.FAILED
            session.notes = str(e)
            logger.error(f"Force scaling calibration failed: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def calibrate_spatial_alignment(self) -> Dict[str, Any]:
        """Calibrate spatial sensor alignment"""
        
        logger.info("Starting spatial alignment calibration")
        
        # Define standard anatomical landmarks
        landmarks = {
            "heel_center": (0.5, 0.85),
            "mtp1": (0.35, 0.25),
            "mtp2": (0.42, 0.22),
            "mtp3": (0.50, 0.20),
            "mtp5": (0.68, 0.18),
            "hallux": (0.38, 0.12)
        }
        
        self.spatial_calibrator.define_anatomical_landmarks(landmarks)
        
        try:
            # Simulate reference point calibration
            reference_points = []
            sensor_responses = []
            
            for landmark_name, position in landmarks.items():
                # Simulate applying pressure at landmark
                reference_points.append((landmark_name, np.array(position)))
                
                # Simulate sensor response (inverse distance weighting)
                response = np.zeros(self.n_sensors)
                for i in range(self.n_sensors):
                    # Random sensor position for simulation
                    sensor_pos = np.random.uniform(0, 1, 2)
                    distance = np.linalg.norm(sensor_pos - position)
                    response[i] = 100.0 / (1 + distance * 5)  # Inverse distance model
                
                sensor_responses.append(response)
            
            # Perform spatial calibration
            spatial_result = self.spatial_calibrator.calibrate_sensor_positions(
                reference_points, sensor_responses
            )
            
            logger.info(f"Spatial calibration completed. Mean error: {spatial_result['mean_error']:.2f}mm")
            
            return {
                "success": spatial_result["success"],
                "sensor_positions": spatial_result["sensor_positions"].tolist(),
                "mean_error": spatial_result["mean_error"],
                "max_error": spatial_result["max_error"]
            }
            
        except Exception as e:
            logger.error(f"Spatial calibration failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def calibrate_temporal_sync(self, duration: float = 10.0) -> Dict[str, Any]:
        """Calibrate temporal synchronization"""
        
        logger.info(f"Starting temporal synchronization calibration for {duration}s")
        
        try:
            # Simulate multiple sensor systems
            sensor_systems = {
                "left_foot": type('MockSystem', (), {})(),
                "right_foot": type('MockSystem', (), {})()
            }
            
            # Perform synchronization calibration
            sync_result = await self.temporal_synchronizer.calibrate_synchronization(
                sensor_systems, duration
            )
            
            logger.info(f"Temporal sync completed. Quality: {sync_result['sync_quality']:.3f}")
            
            return sync_result
            
        except Exception as e:
            logger.error(f"Temporal synchronization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def validate_calibration(self) -> Dict[str, Any]:
        """Validate complete calibration using test procedures"""
        
        logger.info("Starting calibration validation")
        
        try:
            validation_results = {
                "linearity_test": await self._test_linearity(),
                "repeatability_test": await self._test_repeatability(),
                "cross_talk_test": await self._test_cross_talk(),
                "temperature_stability": await self._test_temperature_stability()
            }
            
            # Overall validation score
            scores = [result.get("score", 0) for result in validation_results.values()]
            overall_score = np.mean(scores)
            
            validation_results["overall_score"] = overall_score
            validation_results["success"] = overall_score > 0.8
            
            logger.info(f"Calibration validation completed. Overall score: {overall_score:.3f}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Calibration validation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_linearity(self) -> Dict[str, Any]:
        """Test sensor linearity"""
        test_loads = [0, 25, 50, 75, 100]  # Percentage of full scale
        actual_responses = []
        expected_responses = []
        
        for load in test_loads:
            # Simulate sensor response
            response = load + np.random.normal(0, 1)  # Add noise
            actual_responses.append(response)
            expected_responses.append(load)
        
        # Calculate linearity error
        actual = np.array(actual_responses)
        expected = np.array(expected_responses)
        
        # Linear fit
        slope, intercept, r_value, _, _ = scipy.stats.linregress(expected, actual)
        predicted = slope * expected + intercept
        
        linearity_error = np.max(np.abs(actual - predicted)) / np.max(expected) * 100
        
        return {
            "score": max(0, 1 - linearity_error / 5.0),  # 5% threshold
            "linearity_error": float(linearity_error),
            "correlation": float(r_value**2),
            "slope": float(slope),
            "intercept": float(intercept)
        }
    
    async def _test_repeatability(self) -> Dict[str, Any]:
        """Test measurement repeatability"""
        n_repeats = 10
        test_load = 50.0  # Fixed test load
        
        measurements = []
        for _ in range(n_repeats):
            # Simulate measurement with repeatability error
            measurement = test_load + np.random.normal(0, 0.5)
            measurements.append(measurement)
            await asyncio.sleep(0.1)
        
        measurements = np.array(measurements)
        
        # Calculate repeatability metrics
        mean_value = np.mean(measurements)
        std_value = np.std(measurements)
        repeatability_error = (std_value / mean_value) * 100 if mean_value > 0 else 100
        
        return {
            "score": max(0, 1 - repeatability_error / 2.0),  # 2% threshold
            "repeatability_error": float(repeatability_error),
            "mean_value": float(mean_value),
            "std_deviation": float(std_value),
            "measurements": measurements.tolist()
        }
    
    async def _test_cross_talk(self) -> Dict[str, Any]:
        """Test sensor cross-talk"""
        # Simulate loading one sensor and measuring others
        loaded_sensor = 0
        load_value = 100.0
        
        sensor_responses = np.zeros(self.n_sensors)
        sensor_responses[loaded_sensor] = load_value
        
        # Add simulated cross-talk
        for i in range(self.n_sensors):
            if i != loaded_sensor:
                distance = abs(i - loaded_sensor)
                crosstalk = load_value * 0.05 / (distance + 1)  # Decreasing with distance
                sensor_responses[i] = crosstalk + np.random.normal(0, 0.1)
        
        # Calculate cross-talk metrics
        cross_talk_total = np.sum(sensor_responses) - sensor_responses[loaded_sensor]
        cross_talk_percentage = (cross_talk_total / load_value) * 100
        
        return {
            "score": max(0, 1 - cross_talk_percentage / 10.0),  # 10% threshold
            "cross_talk_percentage": float(cross_talk_percentage),
            "loaded_sensor": loaded_sensor,
            "sensor_responses": sensor_responses.tolist()
        }
    
    async def _test_temperature_stability(self) -> Dict[str, Any]:
        """Test temperature stability"""
        # Simulate temperature drift test
        temperatures = [20, 25, 30, 35, 40]  # °C
        baseline_reading = 50.0
        
        temperature_responses = []
        
        for temp in temperatures:
            # Simulate temperature coefficient (0.1%/°C)
            temp_coefficient = 0.001
            temp_drift = baseline_reading * temp_coefficient * (temp - 25)
            response = baseline_reading + temp_drift + np.random.normal(0, 0.1)
            temperature_responses.append(response)
        
        # Calculate temperature coefficient
        responses = np.array(temperature_responses)
        temp_coeff = np.polyfit(temperatures, responses, 1)[0]
        temp_coeff_percent = (temp_coeff / baseline_reading) * 100
        
        return {
            "score": max(0, 1 - abs(temp_coeff_percent) / 0.2),  # 0.2%/°C threshold
            "temperature_coefficient": float(temp_coeff_percent),
            "temperatures": temperatures,
            "responses": temperature_responses
        }
    
    def apply_calibration(self, raw_sensor_data: np.ndarray) -> np.ndarray:
        """Apply current calibration to raw sensor data"""
        if not self.calibration_valid:
            logger.warning("Applying calibration but calibration may not be valid")
        
        # Apply offset correction
        corrected_data = raw_sensor_data - self.current_offset_vector
        
        # Apply scaling
        calibrated_data = self.current_calibration_matrix @ corrected_data
        
        return calibrated_data
    
    def save_calibration(self, file_path: str):
        """Save calibration data to file"""
        calibration_data = {
            "timestamp": self.calibration_timestamp.isoformat() if self.calibration_timestamp else None,
            "valid": self.calibration_valid,
            "calibration_matrix": self.current_calibration_matrix.tolist(),
            "offset_vector": self.current_offset_vector.tolist(),
            "sensor_config": self.sensor_config,
            "calibration_history": [
                {
                    "session_id": session.session_id,
                    "type": session.calibration_type.value,
                    "timestamp": session.start_time.isoformat(),
                    "status": session.status.value,
                    "rms_error": session.rms_error,
                    "notes": session.notes
                }
                for session in self.calibration_history
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        logger.info(f"Calibration data saved to {file_path}")
    
    def load_calibration(self, file_path: str) -> bool:
        """Load calibration data from file"""
        try:
            with open(file_path, 'r') as f:
                calibration_data = json.load(f)
            
            self.current_calibration_matrix = np.array(calibration_data["calibration_matrix"])
            self.current_offset_vector = np.array(calibration_data["offset_vector"])
            self.calibration_valid = calibration_data["valid"]
            
            if calibration_data["timestamp"]:
                self.calibration_timestamp = datetime.fromisoformat(calibration_data["timestamp"])
            
            logger.info(f"Calibration data loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
            return False
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status"""
        age_days = 0
        if self.calibration_timestamp:
            age_days = (datetime.now() - self.calibration_timestamp).days
        
        return {
            "valid": self.calibration_valid,
            "timestamp": self.calibration_timestamp.isoformat() if self.calibration_timestamp else None,
            "age_days": age_days,
            "expired": age_days > 30,  # 30-day calibration validity
            "last_calibration_type": self.calibration_history[-1].calibration_type.value if self.calibration_history else None,
            "total_calibrations": len(self.calibration_history),
            "current_session": self.current_calibration.session_id if self.current_calibration else None
        }

# Factory function
def create_calibration_system(sensor_config: Dict[str, Any]) -> AdvancedCalibrationSystem:
    """Create advanced calibration system"""
    return AdvancedCalibrationSystem(sensor_config)

# Example usage
if __name__ == "__main__":
    async def main():
        # Create calibration system
        sensor_config = {
            "n_sensors": 16,
            "sampling_rate": 100.0,
            "sensor_type": "pressure"
        }
        
        calibration_system = create_calibration_system(sensor_config)
        
        # Run full calibration
        calibration_config = {
            "baseline_duration": 5.0,
            "enable_force_calibration": True,
            "calibration_weights": [100, 200, 500],
            "enable_spatial_calibration": True,
            "enable_temporal_sync": True,
            "sync_duration": 10.0
        }
        
        def progress_callback(stage, progress, message):
            print(f"[{progress:3d}%] {stage}: {message}")
        
        results = await calibration_system.run_full_calibration(
            calibration_config, progress_callback
        )
        
        print(f"Calibration completed: {results['overall_success']}")
        if not results['overall_success']:
            print(f"Errors: {results['errors']}")
    
    asyncio.run(main())