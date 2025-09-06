# core/advanced_biomechanics.py
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from scipy import signal
from scipy.stats import entropy
from sklearn.cluster import KMeans
import pandas as pd

class GaitPhase:
    """Gait phase definitions"""
    HEEL_STRIKE = "heel_strike"
    LOADING_RESPONSE = "loading_response"
    MIDSTANCE = "midstance"
    TERMINAL_STANCE = "terminal_stance"
    PRE_SWING = "pre_swing"
    INITIAL_SWING = "initial_swing"
    MID_SWING = "mid_swing"
    TERMINAL_SWING = "terminal_swing"

@dataclass
class GaitEvent:
    """Represents a gait event"""
    timestamp: float
    event_type: str
    foot: str  # "left" or "right"
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class StepMetrics:
    """Comprehensive step metrics"""
    # Temporal metrics
    stance_time: float
    swing_time: float
    step_time: float
    cadence: float
    
    # Pressure metrics
    peak_pressure: float
    mean_pressure: float
    pressure_time_integral: float
    contact_area: float
    
    # Center of pressure metrics
    cop_path_length: float
    cop_velocity_mean: float
    cop_velocity_max: float
    cop_range_ml: float  # Medial-lateral range
    cop_range_ap: float  # Anterior-posterior range
    
    # Load distribution
    heel_load_percent: float
    forefoot_load_percent: float
    medial_load_percent: float
    lateral_load_percent: float
    
    # Advanced metrics
    symmetry_index: float
    impulse: float
    load_rate: float
    stability_margin: float

class AdvancedGaitAnalyzer:
    """State-of-the-art gait analysis system"""
    
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # Buffers for analysis
        self.pressure_history_L = []
        self.pressure_history_R = []
        self.cop_history_L = []
        self.cop_history_R = []
        self.timestamps = []
        
        # Gait events
        self.gait_events = []
        self.steps_L = []
        self.steps_R = []
        
        # Analysis parameters
        self.min_contact_pressure = 5.0  # Minimum pressure for contact
        self.heel_strike_threshold = 0.15  # Relative to peak pressure
        self.toe_off_threshold = 0.10
        
        # Anatomical zone indices (will be set based on sensor layout)
        self.heel_sensors = []
        self.forefoot_sensors = []
        self.medial_sensors = []
        self.lateral_sensors = []
        
        self._setup_filters()
    
    def _setup_filters(self):
        """Setup digital filters for signal processing"""
        # Low-pass filter for pressure signals
        self.filter_cutoff = 20.0  # Hz
        self.filter_order = 4
        
        # Butterworth filter coefficients
        nyquist = self.sampling_rate / 2
        normalized_cutoff = self.filter_cutoff / nyquist
        self.b, self.a = signal.butter(self.filter_order, normalized_cutoff, btype='low')
    
    def set_anatomical_zones(self, sensor_coords: np.ndarray):
        """Set anatomical zone sensor indices based on coordinates"""
        n_sensors = len(sensor_coords)
        
        # Classify sensors by anatomical regions
        for i, coord in enumerate(sensor_coords):
            x, y = coord[0], coord[1]
            
            # Heel region (posterior)
            if y > 0.7:
                self.heel_sensors.append(i)
            
            # Forefoot region (anterior)
            elif y < 0.4:
                self.forefoot_sensors.append(i)
            
            # Medial side
            if x < 0.5:
                self.medial_sensors.append(i)
            
            # Lateral side
            if x > 0.5:
                self.lateral_sensors.append(i)
    
    def add_sample(self, timestamp: float, pressures_L: np.ndarray, 
                   pressures_R: np.ndarray, cop_L: Tuple[float, float], 
                   cop_R: Tuple[float, float]):
        """Add new pressure sample for analysis"""
        
        # Store raw data
        self.timestamps.append(timestamp)
        self.pressure_history_L.append(pressures_L.copy())
        self.pressure_history_R.append(pressures_R.copy())
        self.cop_history_L.append(cop_L)
        self.cop_history_R.append(cop_R)
        
        # Maintain rolling window (last 10 seconds)
        max_samples = int(10 * self.sampling_rate)
        if len(self.timestamps) > max_samples:
            self.timestamps = self.timestamps[-max_samples:]
            self.pressure_history_L = self.pressure_history_L[-max_samples:]
            self.pressure_history_R = self.pressure_history_R[-max_samples:]
            self.cop_history_L = self.cop_history_L[-max_samples:]
            self.cop_history_R = self.cop_history_R[-max_samples:]
        
        # Perform real-time analysis
        if len(self.timestamps) > 2:
            self._detect_gait_events()
            self._update_step_metrics()
    
    def _detect_gait_events(self):
        """Detect gait events in real-time"""
        if len(self.pressure_history_L) < 10:
            return
        
        # Get recent pressure data
        recent_L = np.array(self.pressure_history_L[-10:])
        recent_R = np.array(self.pressure_history_R[-10:])
        recent_timestamps = np.array(self.timestamps[-10:])
        
        # Calculate total force for each foot
        force_L = np.sum(recent_L, axis=1)
        force_R = np.sum(recent_R, axis=1)
        
        # Apply smoothing
        if len(force_L) >= 5:
            force_L_smooth = signal.savgol_filter(force_L, 5, 2)
            force_R_smooth = signal.savgol_filter(force_R, 5, 2)
        else:
            force_L_smooth = force_L
            force_R_smooth = force_R
        
        # Detect events for left foot
        self._detect_foot_events(force_L_smooth, recent_timestamps, "left")
        
        # Detect events for right foot
        self._detect_foot_events(force_R_smooth, recent_timestamps, "right")
    
    def _detect_foot_events(self, force_signal: np.ndarray, 
                           timestamps: np.ndarray, foot: str):
        """Detect gait events for a single foot"""
        
        if len(force_signal) < 5:
            return
        
        # Find peaks and valleys
        peaks, peak_props = signal.find_peaks(force_signal, 
                                            height=self.min_contact_pressure,
                                            distance=int(0.3 * self.sampling_rate))
        
        valleys, valley_props = signal.find_peaks(-force_signal,
                                                height=-self.min_contact_pressure/2)
        
        # Identify heel strikes and toe offs
        for peak_idx in peaks:
            timestamp = timestamps[peak_idx]
            force_value = force_signal[peak_idx]
            
            # Check if this is a new heel strike
            recent_events = [e for e in self.gait_events 
                           if e.foot == foot and 
                           abs(e.timestamp - timestamp) < 0.1 and
                           e.event_type == GaitPhase.HEEL_STRIKE]
            
            if not recent_events:
                event = GaitEvent(
                    timestamp=timestamp,
                    event_type=GaitPhase.HEEL_STRIKE,
                    foot=foot,
                    confidence=min(force_value / 50.0, 1.0),
                    metrics={"peak_force": force_value}
                )
                self.gait_events.append(event)
        
        # Similar logic for toe-off events
        for valley_idx in valleys:
            if valley_idx < len(timestamps) - 1:
                timestamp = timestamps[valley_idx]
                
                # Check if this is a new toe-off
                recent_events = [e for e in self.gait_events 
                               if e.foot == foot and 
                               abs(e.timestamp - timestamp) < 0.1 and
                               e.event_type == "toe_off"]
                
                if not recent_events:
                    event = GaitEvent(
                        timestamp=timestamp,
                        event_type="toe_off",
                        foot=foot,
                        confidence=0.8,
                        metrics={}
                    )
                    self.gait_events.append(event)
    
    def _update_step_metrics(self):
        """Update step metrics based on recent gait events"""
        current_time = self.timestamps[-1] if self.timestamps else 0
        
        # Get recent heel strikes for each foot
        recent_hs_L = [e for e in self.gait_events 
                      if e.foot == "left" and 
                      e.event_type == GaitPhase.HEEL_STRIKE and
                      current_time - e.timestamp < 5.0]
        
        recent_hs_R = [e for e in self.gait_events 
                      if e.foot == "right" and 
                      e.event_type == GaitPhase.HEEL_STRIKE and
                      current_time - e.timestamp < 5.0]
        
        # Calculate step metrics if we have enough data
        if len(recent_hs_L) >= 2:
            self._calculate_step_metrics("left", recent_hs_L)
        
        if len(recent_hs_R) >= 2:
            self._calculate_step_metrics("right", recent_hs_R)
    
    def _calculate_step_metrics(self, foot: str, heel_strikes: List[GaitEvent]):
        """Calculate comprehensive metrics for recent steps"""
        
        if len(heel_strikes) < 2:
            return
        
        # Get the most recent step
        latest_hs = heel_strikes[-1]
        previous_hs = heel_strikes[-2]
        
        # Step time
        step_time = latest_hs.timestamp - previous_hs.timestamp
        
        # Find corresponding data indices
        start_idx = None
        end_idx = None
        
        for i, t in enumerate(self.timestamps):
            if start_idx is None and t >= previous_hs.timestamp:
                start_idx = i
            if end_idx is None and t >= latest_hs.timestamp:
                end_idx = i
                break
        
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            return
        
        # Extract step data
        if foot == "left":
            step_pressures = np.array(self.pressure_history_L[start_idx:end_idx])
            step_cops = self.cop_history_L[start_idx:end_idx]
        else:
            step_pressures = np.array(self.pressure_history_R[start_idx:end_idx])
            step_cops = self.cop_history_R[start_idx:end_idx]
        
        step_timestamps = self.timestamps[start_idx:end_idx]
        
        # Calculate metrics
        metrics = self._compute_comprehensive_metrics(
            step_pressures, step_cops, step_timestamps, step_time)
        
        # Store step metrics
        if foot == "left":
            self.steps_L.append(metrics)
            if len(self.steps_L) > 20:  # Keep last 20 steps
                self.steps_L.pop(0)
        else:
            self.steps_R.append(metrics)
            if len(self.steps_R) > 20:
                self.steps_R.pop(0)
    
    def _compute_comprehensive_metrics(self, pressures: np.ndarray, 
                                     cops: List[Tuple[float, float]], 
                                     timestamps: np.ndarray,
                                     step_time: float) -> StepMetrics:
        """Compute comprehensive step metrics"""
        
        dt = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else self.dt
        
        # Basic pressure metrics
        total_force = np.sum(pressures, axis=1)
        peak_pressure = np.max(total_force)
        mean_pressure = np.mean(total_force)
        contact_time = len(pressures) * dt
        
        # Pressure-time integral (impulse)
        impulse = np.trapz(total_force, dx=dt)
        
        # Contact area (simplified - count active sensors)
        contact_areas = []
        for frame in pressures:
            active_sensors = np.sum(frame > self.min_contact_pressure)
            contact_areas.append(active_sensors)
        contact_area = np.mean(contact_areas)
        
        # Center of pressure analysis
        cop_coords = np.array([(cop[0], cop[1]) for cop in cops 
                              if not (np.isnan(cop[0]) or np.isnan(cop[1]))])
        
        if len(cop_coords) > 1:
            # COP path length
            cop_diffs = np.diff(cop_coords, axis=0)
            cop_distances = np.sqrt(np.sum(cop_diffs**2, axis=1))
            cop_path_length = np.sum(cop_distances)
            
            # COP velocity
            cop_velocities = cop_distances / dt
            cop_velocity_mean = np.mean(cop_velocities)
            cop_velocity_max = np.max(cop_velocities)
            
            # COP range
            cop_range_ml = np.ptp(cop_coords[:, 0])  # Medial-lateral
            cop_range_ap = np.ptp(cop_coords[:, 1])  # Anterior-posterior
        else:
            cop_path_length = 0.0
            cop_velocity_mean = 0.0
            cop_velocity_max = 0.0
            cop_range_ml = 0.0
            cop_range_ap = 0.0
        
        # Load distribution
        heel_pressures = pressures[:, self.heel_sensors] if self.heel_sensors else np.zeros((len(pressures), 1))
        forefoot_pressures = pressures[:, self.forefoot_sensors] if self.forefoot_sensors else np.zeros((len(pressures), 1))
        medial_pressures = pressures[:, self.medial_sensors] if self.medial_sensors else np.zeros((len(pressures), 1))
        lateral_pressures = pressures[:, self.lateral_sensors] if self.lateral_sensors else np.zeros((len(pressures), 1))
        
        total_heel_load = np.sum(heel_pressures)
        total_forefoot_load = np.sum(forefoot_pressures)
        total_medial_load = np.sum(medial_pressures)
        total_lateral_load = np.sum(lateral_pressures)
        total_load = np.sum(pressures)
        
        if total_load > 0:
            heel_load_percent = (total_heel_load / total_load) * 100
            forefoot_load_percent = (total_forefoot_load / total_load) * 100
            medial_load_percent = (total_medial_load / total_load) * 100
            lateral_load_percent = (total_lateral_load / total_load) * 100
        else:
            heel_load_percent = forefoot_load_percent = 0.0
            medial_load_percent = lateral_load_percent = 0.0
        
        # Advanced metrics
        cadence = 60.0 / step_time if step_time > 0 else 0.0
        
        # Load rate (simplified)
        if len(total_force) > 5:
            peak_idx = np.argmax(total_force)
            if peak_idx > 2:
                rise_time = peak_idx * dt
                load_rate = peak_pressure / rise_time if rise_time > 0 else 0.0
            else:
                load_rate = 0.0
        else:
            load_rate = 0.0
        
        # Stability margin (simplified)
        stability_margin = cop_range_ml + cop_range_ap  # Simplified metric
        
        return StepMetrics(
            stance_time=contact_time,
            swing_time=max(0, step_time - contact_time),
            step_time=step_time,
            cadence=cadence,
            peak_pressure=peak_pressure,
            mean_pressure=mean_pressure,
            pressure_time_integral=impulse,
            contact_area=contact_area,
            cop_path_length=cop_path_length,
            cop_velocity_mean=cop_velocity_mean,
            cop_velocity_max=cop_velocity_max,
            cop_range_ml=cop_range_ml,
            cop_range_ap=cop_range_ap,
            heel_load_percent=heel_load_percent,
            forefoot_load_percent=forefoot_load_percent,
            medial_load_percent=medial_load_percent,
            lateral_load_percent=lateral_load_percent,
            symmetry_index=0.0,  # Will be calculated when comparing feet
            impulse=impulse,
            load_rate=load_rate,
            stability_margin=stability_margin
        )
    
    def calculate_asymmetry_indices(self) -> Dict[str, float]:
        """Calculate bilateral asymmetry indices"""
        
        if not self.steps_L or not self.steps_R:
            return {}
        
        # Get recent steps
        recent_L = self.steps_L[-5:] if len(self.steps_L) >= 5 else self.steps_L
        recent_R = self.steps_R[-5:] if len(self.steps_R) >= 5 else self.steps_R
        
        if not recent_L or not recent_R:
            return {}
        
        # Calculate mean values for each metric
        metrics_L = self._average_step_metrics(recent_L)
        metrics_R = self._average_step_metrics(recent_R)
        
        asymmetry_indices = {}
        
        # Calculate symmetry index for each metric
        # SI = |Left - Right| / (0.5 * (Left + Right)) * 100
        metric_names = [
            'step_time', 'stance_time', 'peak_pressure', 'mean_pressure',
            'contact_area', 'cop_path_length', 'heel_load_percent',
            'forefoot_load_percent', 'medial_load_percent', 'lateral_load_percent'
        ]
        
        for metric in metric_names:
            val_L = getattr(metrics_L, metric, 0)
            val_R = getattr(metrics_R, metric, 0)
            
            if val_L + val_R > 0:
                si = abs(val_L - val_R) / (0.5 * (val_L + val_R)) * 100
                asymmetry_indices[f'{metric}_asymmetry'] = si
        
        return asymmetry_indices
    
    def _average_step_metrics(self, steps: List[StepMetrics]) -> StepMetrics:
        """Calculate average of step metrics"""
        if not steps:
            return StepMetrics(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        
        # Average all metrics
        n_steps = len(steps)
        
        return StepMetrics(
            stance_time=sum(s.stance_time for s in steps) / n_steps,
            swing_time=sum(s.swing_time for s in steps) / n_steps,
            step_time=sum(s.step_time for s in steps) / n_steps,
            cadence=sum(s.cadence for s in steps) / n_steps,
            peak_pressure=sum(s.peak_pressure for s in steps) / n_steps,
            mean_pressure=sum(s.mean_pressure for s in steps) / n_steps,
            pressure_time_integral=sum(s.pressure_time_integral for s in steps) / n_steps,
            contact_area=sum(s.contact_area for s in steps) / n_steps,
            cop_path_length=sum(s.cop_path_length for s in steps) / n_steps,
            cop_velocity_mean=sum(s.cop_velocity_mean for s in steps) / n_steps,
            cop_velocity_max=sum(s.cop_velocity_max for s in steps) / n_steps,
            cop_range_ml=sum(s.cop_range_ml for s in steps) / n_steps,
            cop_range_ap=sum(s.cop_range_ap for s in steps) / n_steps,
            heel_load_percent=sum(s.heel_load_percent for s in steps) / n_steps,
            forefoot_load_percent=sum(s.forefoot_load_percent for s in steps) / n_steps,
            medial_load_percent=sum(s.medial_load_percent for s in steps) / n_steps,
            lateral_load_percent=sum(s.lateral_load_percent for s in steps) / n_steps,
            symmetry_index=0.0,
            impulse=sum(s.impulse for s in steps) / n_steps,
            load_rate=sum(s.load_rate for s in steps) / n_steps,
            stability_margin=sum(s.stability_margin for s in steps) / n_steps
        )
    
    def get_gait_analysis_report(self) -> Dict:
        """Generate comprehensive gait analysis report"""
        
        current_time = self.timestamps[-1] if self.timestamps else 0
        
        # Recent gait events
        recent_events = [e for e in self.gait_events 
                        if current_time - e.timestamp < 30.0]
        
        # Step statistics
        steps_L_avg = self._average_step_metrics(self.steps_L) if self.steps_L else None
        steps_R_avg = self._average_step_metrics(self.steps_R) if self.steps_R else None
        
        # Asymmetry analysis
        asymmetry = self.calculate_asymmetry_indices()
        
        # Temporal parameters
        temporal_params = {}
        if steps_L_avg and steps_R_avg:
            temporal_params = {
                'cadence_L': steps_L_avg.cadence,
                'cadence_R': steps_R_avg.cadence,
                'step_time_L': steps_L_avg.step_time,
                'step_time_R': steps_R_avg.step_time,
                'stance_time_L': steps_L_avg.stance_time,
                'stance_time_R': steps_R_avg.stance_time,
                'swing_time_L': steps_L_avg.swing_time,
                'swing_time_R': steps_R_avg.swing_time,
                'stride_time_L': 2 * steps_L_avg.step_time,
                'stride_time_R': 2 * steps_R_avg.step_time,
                'stance_phase_L': (steps_L_avg.stance_time / steps_L_avg.step_time * 100) if steps_L_avg.step_time > 0 else 0,
                'stance_phase_R': (steps_R_avg.stance_time / steps_R_avg.step_time * 100) if steps_R_avg.step_time > 0 else 0,
            }
        
        # Pressure parameters
        pressure_params = {}
        if steps_L_avg and steps_R_avg:
            pressure_params = {
                'peak_pressure_L': steps_L_avg.peak_pressure,
                'peak_pressure_R': steps_R_avg.peak_pressure,
                'mean_pressure_L': steps_L_avg.mean_pressure,
                'mean_pressure_R': steps_R_avg.mean_pressure,
                'pressure_integral_L': steps_L_avg.pressure_time_integral,
                'pressure_integral_R': steps_R_avg.pressure_time_integral,
                'contact_area_L': steps_L_avg.contact_area,
                'contact_area_R': steps_R_avg.contact_area,
                'load_rate_L': steps_L_avg.load_rate,
                'load_rate_R': steps_R_avg.load_rate,
            }
        
        # COP parameters
        cop_params = {}
        if steps_L_avg and steps_R_avg:
            cop_params = {
                'cop_path_length_L': steps_L_avg.cop_path_length,
                'cop_path_length_R': steps_R_avg.cop_path_length,
                'cop_velocity_mean_L': steps_L_avg.cop_velocity_mean,
                'cop_velocity_mean_R': steps_R_avg.cop_velocity_mean,
                'cop_range_ml_L': steps_L_avg.cop_range_ml,
                'cop_range_ml_R': steps_R_avg.cop_range_ml,
                'cop_range_ap_L': steps_L_avg.cop_range_ap,
                'cop_range_ap_R': steps_R_avg.cop_range_ap,
            }
        
        # Load distribution
        load_distribution = {}
        if steps_L_avg and steps_R_avg:
            load_distribution = {
                'heel_load_L': steps_L_avg.heel_load_percent,
                'heel_load_R': steps_R_avg.heel_load_percent,
                'forefoot_load_L': steps_L_avg.forefoot_load_percent,
                'forefoot_load_R': steps_R_avg.forefoot_load_percent,
                'medial_load_L': steps_L_avg.medial_load_percent,
                'medial_load_R': steps_R_avg.medial_load_percent,
                'lateral_load_L': steps_L_avg.lateral_load_percent,
                'lateral_load_R': steps_R_avg.lateral_load_percent,
            }
        
        return {
            'timestamp': current_time,
            'n_steps_L': len(self.steps_L),
            'n_steps_R': len(self.steps_R),
            'n_gait_events': len(recent_events),
            'temporal_parameters': temporal_params,
            'pressure_parameters': pressure_params,
            'cop_parameters': cop_params,
            'load_distribution': load_distribution,
            'asymmetry_indices': asymmetry,
            'data_quality': self._assess_data_quality(),
            'clinical_flags': self._generate_clinical_flags(steps_L_avg, steps_R_avg, asymmetry)
        }
    
    def _assess_data_quality(self) -> Dict[str, float]:
        """Assess the quality of collected data"""
        
        quality_metrics = {
            'data_completeness': 0.0,
            'signal_consistency': 0.0,
            'step_detection_confidence': 0.0,
            'sensor_coverage': 0.0
        }
        
        if not self.timestamps:
            return quality_metrics
        
        # Data completeness (percentage of expected samples)
        duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        expected_samples = duration * self.sampling_rate
        actual_samples = len(self.timestamps)
        
        if expected_samples > 0:
            quality_metrics['data_completeness'] = min(actual_samples / expected_samples, 1.0)
        
        # Signal consistency (coefficient of variation of step times)
        if len(self.steps_L) > 1:
            step_times = [s.step_time for s in self.steps_L]
            cv_L = np.std(step_times) / np.mean(step_times) if np.mean(step_times) > 0 else 1.0
            consistency_L = max(0, 1.0 - cv_L)
        else:
            consistency_L = 0.0
        
        if len(self.steps_R) > 1:
            step_times = [s.step_time for s in self.steps_R]
            cv_R = np.std(step_times) / np.mean(step_times) if np.mean(step_times) > 0 else 1.0
            consistency_R = max(0, 1.0 - cv_R)
        else:
            consistency_R = 0.0
        
        quality_metrics['signal_consistency'] = (consistency_L + consistency_R) / 2
        
        # Step detection confidence (average confidence of recent events)
        recent_events = [e for e in self.gait_events if self.timestamps and 
                        self.timestamps[-1] - e.timestamp < 10.0]
        
        if recent_events:
            avg_confidence = np.mean([e.confidence for e in recent_events])
            quality_metrics['step_detection_confidence'] = avg_confidence
        
        # Sensor coverage (percentage of sensors showing activity)
        if self.pressure_history_L:
            recent_pressures = np.array(self.pressure_history_L[-100:])  # Last 100 samples
            active_sensors = np.sum(np.max(recent_pressures, axis=0) > self.min_contact_pressure)
            total_sensors = recent_pressures.shape[1]
            quality_metrics['sensor_coverage'] = active_sensors / total_sensors if total_sensors > 0 else 0
        
        return quality_metrics
    
    def _generate_clinical_flags(self, steps_L_avg: Optional[StepMetrics], 
                               steps_R_avg: Optional[StepMetrics],
                               asymmetry: Dict[str, float]) -> List[Dict[str, str]]:
        """Generate clinical flags based on analysis"""
        
        flags = []
        
        # Asymmetry flags
        for metric, value in asymmetry.items():
            if value > 15.0:  # >15% asymmetry is clinically significant
                flags.append({
                    'type': 'asymmetry',
                    'severity': 'high' if value > 25.0 else 'moderate',
                    'message': f'High bilateral asymmetry in {metric.replace("_asymmetry", "")}: {value:.1f}%',
                    'metric': metric,
                    'value': value
                })
        
        # Pressure distribution flags
        if steps_L_avg and steps_R_avg:
            # Excessive heel loading
            if steps_L_avg.heel_load_percent > 70 or steps_R_avg.heel_load_percent > 70:
                flags.append({
                    'type': 'pressure_distribution',
                    'severity': 'moderate',
                    'message': 'Excessive heel loading detected',
                    'metric': 'heel_loading',
                    'value': max(steps_L_avg.heel_load_percent, steps_R_avg.heel_load_percent)
                })
            
            # Excessive forefoot loading
            if steps_L_avg.forefoot_load_percent > 70 or steps_R_avg.forefoot_load_percent > 70:
                flags.append({
                    'type': 'pressure_distribution',
                    'severity': 'moderate',
                    'message': 'Excessive forefoot loading detected',
                    'metric': 'forefoot_loading',
                    'value': max(steps_L_avg.forefoot_load_percent, steps_R_avg.forefoot_load_percent)
                })
            
            # High peak pressures
            peak_pressure_threshold = 300  # kPa
            if (steps_L_avg.peak_pressure > peak_pressure_threshold or 
                steps_R_avg.peak_pressure > peak_pressure_threshold):
                flags.append({
                    'type': 'peak_pressure',
                    'severity': 'high',
                    'message': 'Elevated peak pressures detected',
                    'metric': 'peak_pressure',
                    'value': max(steps_L_avg.peak_pressure, steps_R_avg.peak_pressure)
                })
            
            # Temporal asymmetries
            cadence_diff = abs(steps_L_avg.cadence - steps_R_avg.cadence)
            if cadence_diff > 10:  # >10 bpm difference
                flags.append({
                    'type': 'temporal',
                    'severity': 'moderate',
                    'message': f'Significant cadence asymmetry: {cadence_diff:.1f} bpm',
                    'metric': 'cadence_asymmetry',
                    'value': cadence_diff
                })
        
        return flags
    
    def export_analysis_data(self, filepath: str):
        """Export analysis data to CSV for further analysis"""
        
        # Combine all step data
        all_data = []
        
        for i, step in enumerate(self.steps_L):
            data_row = {
                'foot': 'left',
                'step_number': i + 1,
                'timestamp': self.gait_events[i].timestamp if i < len(self.gait_events) else 0,
                **step.__dict__
            }
            all_data.append(data_row)
        
        for i, step in enumerate(self.steps_R):
            data_row = {
                'foot': 'right',
                'step_number': i + 1,
                'timestamp': self.gait_events[i].timestamp if i < len(self.gait_events) else 0,
                **step.__dict__
            }
            all_data.append(data_row)
        
        # Convert to DataFrame and save
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(filepath, index=False)
            return len(all_data)
        
        return 0

# Utility functions for integration with main application
def create_gait_analyzer(sampling_rate: float = 100.0,
                        sensor_coords_L: np.ndarray = None,
                        sensor_coords_R: np.ndarray = None) -> AdvancedGaitAnalyzer:
    """Factory function to create and configure gait analyzer"""
    
    analyzer = AdvancedGaitAnalyzer(sampling_rate=sampling_rate)
    
    # Set anatomical zones for both feet
    if sensor_coords_L is not None:
        analyzer.set_anatomical_zones(sensor_coords_L)
    
    return analyzer

def analyze_session_data(pressure_data_L: List[np.ndarray],
                        pressure_data_R: List[np.ndarray],
                        cop_data_L: List[Tuple[float, float]],
                        cop_data_R: List[Tuple[float, float]],
                        timestamps: List[float],
                        sensor_coords_L: np.ndarray = None) -> Dict:
    """Analyze a complete session of data"""
    
    analyzer = create_gait_analyzer(
        sampling_rate=100.0,  # Assume 100Hz
        sensor_coords_L=sensor_coords_L
    )
    
    # Process all samples
    for i, timestamp in enumerate(timestamps):
        if i < len(pressure_data_L) and i < len(pressure_data_R):
            analyzer.add_sample(
                timestamp=timestamp,
                pressures_L=pressure_data_L[i],
                pressures_R=pressure_data_R[i],
                cop_L=cop_data_L[i] if i < len(cop_data_L) else (0, 0),
                cop_R=cop_data_R[i] if i < len(cop_data_R) else (0, 0)
            )
    
    # Generate final report
    return analyzer.get_gait_analysis_report()