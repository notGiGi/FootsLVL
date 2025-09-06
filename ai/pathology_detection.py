# ai/pathology_detection.py
"""
Advanced Machine Learning system for automatic pathology detection
from plantar pressure and gait patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
import joblib
import logging
from pathlib import Path
import json
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PathologyPrediction:
    """Pathology prediction result"""
    condition: str
    probability: float
    confidence: float
    severity: str  # "mild", "moderate", "severe"
    features_contributing: List[str]
    clinical_notes: str = ""

@dataclass
class GaitFeatureVector:
    """Comprehensive gait feature vector for ML analysis"""
    # Temporal features
    cadence: float
    step_time_variability: float
    stance_phase_percentage: float
    double_support_time: float
    step_length_asymmetry: float
    
    # Pressure features
    peak_pressure_max: float
    peak_pressure_asymmetry: float
    pressure_time_integral: float
    contact_area_mean: float
    loading_rate: float
    
    # COP features  
    cop_path_length: float
    cop_velocity_mean: float
    cop_mediolateral_range: float
    cop_anteroposterior_range: float
    cop_sway_area: float
    
    # Frequency domain features
    step_frequency_dominant: float
    step_frequency_power: float
    pressure_signal_entropy: float
    
    # Symmetry features
    bilateral_symmetry_index: float
    mediolateral_balance: float
    heel_toe_transition_smoothness: float
    
    # Advanced biomechanical features
    propulsion_impulse: float
    braking_impulse: float
    vertical_impulse: float
    center_of_mass_displacement: float
    stability_margin: float

class FeatureExtractor:
    """Extracts comprehensive features from gait data for ML analysis"""
    
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
    
    def extract_features(self, 
                        pressure_data_L: np.ndarray,
                        pressure_data_R: np.ndarray,
                        cop_data_L: List[Tuple[float, float]],
                        cop_data_R: List[Tuple[float, float]], 
                        timestamps: np.ndarray) -> GaitFeatureVector:
        """Extract comprehensive feature vector from gait data"""
        
        # Temporal features
        temporal_features = self._extract_temporal_features(
            pressure_data_L, pressure_data_R, timestamps)
        
        # Pressure features
        pressure_features = self._extract_pressure_features(
            pressure_data_L, pressure_data_R)
        
        # COP features
        cop_features = self._extract_cop_features(cop_data_L, cop_data_R)
        
        # Frequency domain features
        frequency_features = self._extract_frequency_features(
            pressure_data_L, pressure_data_R, timestamps)
        
        # Symmetry features
        symmetry_features = self._extract_symmetry_features(
            pressure_data_L, pressure_data_R, cop_data_L, cop_data_R)
        
        # Advanced biomechanical features
        biomech_features = self._extract_biomechanical_features(
            pressure_data_L, pressure_data_R, cop_data_L, cop_data_R, timestamps)
        
        # Combine all features
        return GaitFeatureVector(
            **temporal_features,
            **pressure_features,
            **cop_features,
            **frequency_features,
            **symmetry_features,
            **biomech_features
        )
    
    def _extract_temporal_features(self, press_L, press_R, timestamps) -> Dict:
        """Extract temporal gait features"""
        
        # Calculate total force signals
        force_L = np.sum(press_L, axis=1)
        force_R = np.sum(press_R, axis=1)
        
        # Detect steps using force thresholds
        threshold = 0.2 * np.max([np.max(force_L), np.max(force_R)])
        
        # Find heel strikes and toe offs
        peaks_L, _ = find_peaks(force_L, height=threshold, distance=int(0.5 * self.sampling_rate))
        peaks_R, _ = find_peaks(force_R, height=threshold, distance=int(0.5 * self.sampling_rate))
        
        # Calculate step times
        if len(peaks_L) > 1:
            step_times_L = np.diff(timestamps[peaks_L])
            cadence_L = 60.0 / np.mean(step_times_L) if len(step_times_L) > 0 else 0
            step_variability_L = np.std(step_times_L) / np.mean(step_times_L) if len(step_times_L) > 0 else 0
        else:
            cadence_L = 0
            step_variability_L = 0
            
        if len(peaks_R) > 1:
            step_times_R = np.diff(timestamps[peaks_R])
            cadence_R = 60.0 / np.mean(step_times_R) if len(step_times_R) > 0 else 0
            step_variability_R = np.std(step_times_R) / np.mean(step_times_R) if len(step_times_R) > 0 else 0
        else:
            cadence_R = 0
            step_variability_R = 0
        
        # Stance phase calculation (simplified)
        stance_L = self._calculate_stance_phase(force_L, threshold)
        stance_R = self._calculate_stance_phase(force_R, threshold)
        
        return {
            'cadence': (cadence_L + cadence_R) / 2,
            'step_time_variability': (step_variability_L + step_variability_R) / 2,
            'stance_phase_percentage': (stance_L + stance_R) / 2,
            'double_support_time': self._calculate_double_support(force_L, force_R, threshold),
            'step_length_asymmetry': abs(cadence_L - cadence_R) / max(cadence_L, cadence_R, 1e-6) * 100
        }
    
    def _extract_pressure_features(self, press_L, press_R) -> Dict:
        """Extract pressure-related features"""
        
        # Peak pressures
        peak_L = np.max(np.sum(press_L, axis=1))
        peak_R = np.max(np.sum(press_R, axis=1))
        peak_asymmetry = abs(peak_L - peak_R) / max(peak_L, peak_R, 1e-6) * 100
        
        # Pressure-time integrals
        pti_L = np.trapz(np.sum(press_L, axis=1)) * self.dt
        pti_R = np.trapz(np.sum(press_R, axis=1)) * self.dt
        
        # Contact area (simplified - count active sensors)
        contact_area_L = np.mean(np.sum(press_L > 5, axis=1))  # 5 units threshold
        contact_area_R = np.mean(np.sum(press_R > 5, axis=1))
        
        # Loading rate
        loading_rate_L = self._calculate_loading_rate(np.sum(press_L, axis=1))
        loading_rate_R = self._calculate_loading_rate(np.sum(press_R, axis=1))
        
        return {
            'peak_pressure_max': max(peak_L, peak_R),
            'peak_pressure_asymmetry': peak_asymmetry,
            'pressure_time_integral': (pti_L + pti_R) / 2,
            'contact_area_mean': (contact_area_L + contact_area_R) / 2,
            'loading_rate': (loading_rate_L + loading_rate_R) / 2
        }
    
    def _extract_cop_features(self, cop_L, cop_R) -> Dict:
        """Extract center of pressure features"""
        
        def analyze_cop_trajectory(cop_data):
            if not cop_data or len(cop_data) < 2:
                return {'path_length': 0, 'velocity_mean': 0, 'ml_range': 0, 'ap_range': 0, 'sway_area': 0}
            
            coords = np.array([(c[0], c[1]) for c in cop_data if not (np.isnan(c[0]) or np.isnan(c[1]))])
            
            if len(coords) < 2:
                return {'path_length': 0, 'velocity_mean': 0, 'ml_range': 0, 'ap_range': 0, 'sway_area': 0}
            
            # Path length
            diffs = np.diff(coords, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            path_length = np.sum(distances)
            
            # Velocity
            velocities = distances / self.dt
            velocity_mean = np.mean(velocities)
            
            # Ranges
            ml_range = np.ptp(coords[:, 0])  # Mediolateral
            ap_range = np.ptp(coords[:, 1])  # Anteroposterior
            
            # Sway area (convex hull area approximation)
            if len(coords) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(coords)
                    sway_area = hull.volume  # 2D volume is area
                except:
                    sway_area = ml_range * ap_range  # Rectangle approximation
            else:
                sway_area = 0
            
            return {
                'path_length': path_length,
                'velocity_mean': velocity_mean,
                'ml_range': ml_range,
                'ap_range': ap_range,
                'sway_area': sway_area
            }
        
        cop_L_features = analyze_cop_trajectory(cop_L)
        cop_R_features = analyze_cop_trajectory(cop_R)
        
        return {
            'cop_path_length': (cop_L_features['path_length'] + cop_R_features['path_length']) / 2,
            'cop_velocity_mean': (cop_L_features['velocity_mean'] + cop_R_features['velocity_mean']) / 2,
            'cop_mediolateral_range': (cop_L_features['ml_range'] + cop_R_features['ml_range']) / 2,
            'cop_anteroposterior_range': (cop_L_features['ap_range'] + cop_R_features['ap_range']) / 2,
            'cop_sway_area': (cop_L_features['sway_area'] + cop_R_features['sway_area']) / 2
        }
    
    def _extract_frequency_features(self, press_L, press_R, timestamps) -> Dict:
        """Extract frequency domain features"""
        
        # Total force signals
        force_L = np.sum(press_L, axis=1)
        force_R = np.sum(press_R, axis=1)
        force_combined = force_L + force_R
        
        # FFT analysis
        if len(force_combined) > 10:
            fft = np.fft.fft(force_combined - np.mean(force_combined))
            freqs = np.fft.fftfreq(len(force_combined), d=self.dt)
            
            # Find dominant frequency
            power_spectrum = np.abs(fft)**2
            positive_freqs = freqs[freqs > 0]
            positive_power = power_spectrum[freqs > 0]
            
            if len(positive_freqs) > 0:
                dominant_freq_idx = np.argmax(positive_power)
                dominant_freq = positive_freqs[dominant_freq_idx]
                dominant_power = positive_power[dominant_freq_idx]
            else:
                dominant_freq = 0
                dominant_power = 0
        else:
            dominant_freq = 0
            dominant_power = 0
        
        # Signal entropy
        entropy = self._calculate_signal_entropy(force_combined)
        
        return {
            'step_frequency_dominant': dominant_freq,
            'step_frequency_power': dominant_power,
            'pressure_signal_entropy': entropy
        }
    
    def _extract_symmetry_features(self, press_L, press_R, cop_L, cop_R) -> Dict:
        """Extract symmetry and balance features"""
        
        # Bilateral symmetry index
        force_L = np.sum(press_L, axis=1)
        force_R = np.sum(press_R, axis=1)
        
        mean_L = np.mean(force_L)
        mean_R = np.mean(force_R)
        
        if mean_L + mean_R > 0:
            bilateral_symmetry = abs(mean_L - mean_R) / (0.5 * (mean_L + mean_R)) * 100
        else:
            bilateral_symmetry = 0
        
        # Mediolateral balance (simplified)
        total_force = force_L + force_R
        if np.sum(total_force) > 0:
            ml_balance = (np.sum(force_L) - np.sum(force_R)) / np.sum(total_force) * 100
        else:
            ml_balance = 0
        
        # Heel-toe transition smoothness
        heel_toe_smoothness = self._calculate_heel_toe_smoothness(press_L, press_R)
        
        return {
            'bilateral_symmetry_index': bilateral_symmetry,
            'mediolateral_balance': abs(ml_balance),
            'heel_toe_transition_smoothness': heel_toe_smoothness
        }
    
    def _extract_biomechanical_features(self, press_L, press_R, cop_L, cop_R, timestamps) -> Dict:
        """Extract advanced biomechanical features"""
        
        # Impulse calculations (simplified)
        force_L = np.sum(press_L, axis=1)
        force_R = np.sum(press_R, axis=1)
        total_force = force_L + force_R
        
        # Vertical impulse
        vertical_impulse = np.trapz(total_force) * self.dt
        
        # Propulsion vs braking (based on COP progression)
        propulsion_impulse = self._calculate_propulsion_impulse(press_L, press_R, cop_L, cop_R)
        braking_impulse = self._calculate_braking_impulse(press_L, press_R, cop_L, cop_R)
        
        # Center of mass displacement approximation
        com_displacement = self._estimate_com_displacement(cop_L, cop_R, total_force)
        
        # Stability margin
        stability_margin = self._calculate_stability_margin(cop_L, cop_R)
        
        return {
            'propulsion_impulse': propulsion_impulse,
            'braking_impulse': braking_impulse,
            'vertical_impulse': vertical_impulse,
            'center_of_mass_displacement': com_displacement,
            'stability_margin': stability_margin
        }
    
    # Helper methods for feature calculations
    def _calculate_stance_phase(self, force_signal, threshold):
        """Calculate stance phase percentage"""
        contact_frames = np.sum(force_signal > threshold)
        total_frames = len(force_signal)
        return (contact_frames / total_frames) * 100 if total_frames > 0 else 0
    
    def _calculate_double_support(self, force_L, force_R, threshold):
        """Calculate double support time"""
        contact_L = force_L > threshold
        contact_R = force_R > threshold
        double_support_frames = np.sum(contact_L & contact_R)
        return (double_support_frames * self.dt)
    
    def _calculate_loading_rate(self, force_signal):
        """Calculate loading rate"""
        if len(force_signal) < 5:
            return 0
        
        # Find peak and calculate slope
        peak_idx = np.argmax(force_signal)
        if peak_idx < 3:
            return 0
        
        # Calculate slope from 20% to 80% of peak
        peak_value = force_signal[peak_idx]
        start_value = 0.2 * peak_value
        end_value = 0.8 * peak_value
        
        # Find indices
        start_idx = None
        end_idx = None
        
        for i in range(peak_idx):
            if start_idx is None and force_signal[i] >= start_value:
                start_idx = i
            if force_signal[i] >= end_value:
                end_idx = i
        
        if start_idx is not None and end_idx is not None and end_idx > start_idx:
            rise_time = (end_idx - start_idx) * self.dt
            return (end_value - start_value) / rise_time if rise_time > 0 else 0
        
        return 0
    
    def _calculate_signal_entropy(self, signal):
        """Calculate signal entropy"""
        if len(signal) == 0:
            return 0
        
        # Normalize and bin the signal
        normalized = (signal - np.min(signal)) / (np.ptp(signal) + 1e-6)
        hist, _ = np.histogram(normalized, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0
        
        return -np.sum(hist * np.log2(hist + 1e-6))
    
    def _calculate_heel_toe_smoothness(self, press_L, press_R):
        """Calculate heel-to-toe transition smoothness"""
        # Simplified: calculate variability in force progression
        force_L = np.sum(press_L, axis=1)
        force_R = np.sum(press_R, axis=1)
        
        if len(force_L) < 5 or len(force_R) < 5:
            return 0
        
        # Apply smoothing and calculate derivatives
        smooth_L = savgol_filter(force_L, min(11, len(force_L)//2*2-1), 3)
        smooth_R = savgol_filter(force_R, min(11, len(force_R)//2*2-1), 3)
        
        # Calculate smoothness as inverse of acceleration variance
        if len(smooth_L) > 2 and len(smooth_R) > 2:
            accel_L = np.diff(smooth_L, 2)
            accel_R = np.diff(smooth_R, 2)
            
            smoothness_L = 1.0 / (1.0 + np.var(accel_L))
            smoothness_R = 1.0 / (1.0 + np.var(accel_R))
            
            return (smoothness_L + smoothness_R) / 2
        
        return 0
    
    def _calculate_propulsion_impulse(self, press_L, press_R, cop_L, cop_R):
        """Calculate propulsion impulse (simplified)"""
        # This would normally require force plate data
        # Here we approximate based on anterior pressure progression
        force_total = np.sum(press_L + press_R, axis=1)
        
        # Simple approximation: impulse in latter half of stance
        mid_point = len(force_total) // 2
        return np.trapz(force_total[mid_point:]) * self.dt if len(force_total) > mid_point else 0
    
    def _calculate_braking_impulse(self, press_L, press_R, cop_L, cop_R):
        """Calculate braking impulse (simplified)"""
        force_total = np.sum(press_L + press_R, axis=1)
        
        # Simple approximation: impulse in first half of stance  
        mid_point = len(force_total) // 2
        return np.trapz(force_total[:mid_point]) * self.dt
    
    def _estimate_com_displacement(self, cop_L, cop_R, force_signal):
        """Estimate center of mass displacement"""
        if not cop_L or not cop_R or len(force_signal) == 0:
            return 0
        
        # Simplified approximation based on COP range
        all_cops = cop_L + cop_R
        valid_cops = [(x, y) for x, y in all_cops if not (np.isnan(x) or np.isnan(y))]
        
        if len(valid_cops) < 2:
            return 0
        
        coords = np.array(valid_cops)
        return np.sqrt(np.var(coords[:, 0]) + np.var(coords[:, 1]))
    
    def _calculate_stability_margin(self, cop_L, cop_R):
        """Calculate stability margin"""
        all_cops = cop_L + cop_R
        valid_cops = [(x, y) for x, y in all_cops if not (np.isnan(x) or np.isnan(y))]
        
        if len(valid_cops) < 2:
            return 0
        
        coords = np.array(valid_cops)
        # Stability approximated as inverse of COP variability
        variability = np.sqrt(np.var(coords[:, 0]) + np.var(coords[:, 1]))
        return 1.0 / (1.0 + variability)

class PathologyClassifier:
    """Machine learning classifier for gait pathology detection"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.model_path = Path(model_path) if model_path else Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Load pre-trained models if available
        if model_path:
            self.load_models()
    
    def _initialize_models(self):
        """Initialize ML models for different pathology types"""
        
        # Multi-class classifier for general pathology detection
        self.models['general'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Specific classifiers for different conditions
        self.models['diabetes'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            random_state=42,
            max_iter=500
        )
        
        self.models['parkinson'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )
        
        self.models['stroke'] = MLPClassifier(
            hidden_layer_sizes=(150, 75, 25),
            activation='tanh',
            solver='adam',
            random_state=42,
            max_iter=800
        )
        
        # Anomaly detection for outlier identification
        self.models['anomaly'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def train_model(self, 
                   features: List[GaitFeatureVector],
                   labels: List[str],
                   model_type: str = 'general',
                   validation_split: float = 0.2) -> Dict[str, float]:
        """Train a specific pathology detection model"""
        
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Convert features to array
        feature_matrix = self._features_to_matrix(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, labels, 
            test_size=validation_split, 
            random_state=42,
            stratify=labels
        )
        
        # Scale features
        scaler = self.scalers[model_type]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = self.models[model_type]
        
        if model_type == 'anomaly':
            # Unsupervised training for anomaly detection
            model.fit(X_train_scaled)
            
            # Evaluate anomaly detection
            train_predictions = model.predict(X_train_scaled)
            test_predictions = model.predict(X_test_scaled)
            
            anomaly_score = np.mean(test_predictions == 1)  # Normal samples percentage
            
            metrics = {
                'anomaly_detection_accuracy': anomaly_score,
                'training_samples': len(X_train)
            }
        else:
            # Supervised training
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Cross validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[model_type] = model.feature_importances_
            
            # Generate predictions for detailed metrics
            y_pred = model.predict(X_test_scaled)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'classification_report': class_report,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            # AUC for binary classification
            if len(set(labels)) == 2:
                try:
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                    auc_score = roc_auc_score(y_test, y_proba)
                    metrics['auc_roc'] = auc_score
                except:
                    pass
        
        logger.info(f"Model {model_type} trained with accuracy: {metrics.get('test_accuracy', 'N/A')}")
        
        return metrics
    
    def predict_pathology(self, features: GaitFeatureVector) -> List[PathologyPrediction]:
        """Predict pathologies from gait features"""
        
        predictions = []
        feature_vector = self._features_to_matrix([features])[0:1]  # Single sample
        
        for model_name, model in self.models.items():
            if model_name == 'anomaly':
                continue  # Handle separately
            
            try:
                scaler = self.scalers[model_name]
                scaled_features = scaler.transform(feature_vector)
                
                # Get prediction and probability
                prediction = model.predict(scaled_features)[0]
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(scaled_features)[0]
                    max_prob_idx = np.argmax(probabilities)
                    confidence = probabilities[max_prob_idx]
                    
                    # Map prediction to condition
                    if hasattr(model, 'classes_'):
                        condition = model.classes_[max_prob_idx]
                    else:
                        condition = prediction
                else:
                    confidence = 0.5  # Default confidence for non-probabilistic models
                    condition = prediction
                
                # Determine severity based on confidence and model type
                severity = self._determine_severity(confidence, model_name)
                
                # Get contributing features
                contributing_features = self._get_contributing_features(
                    features, model_name, scaled_features[0])
                
                # Generate clinical notes
                clinical_notes = self._generate_clinical_notes(
                    condition, confidence, model_name)
                
                pred = PathologyPrediction(
                    condition=str(condition),
                    probability=float(confidence),
                    confidence=float(confidence),
                    severity=severity,
                    features_contributing=contributing_features,
                    clinical_notes=clinical_notes
                )
                
                predictions.append(pred)
                
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                continue
        
        # Check for anomalies
        anomaly_result = self._check_anomaly(features)
        if anomaly_result:
            predictions.append(anomaly_result)
        
        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        
        return predictions
    
    def _check_anomaly(self, features: GaitFeatureVector) -> Optional[PathologyPrediction]:
        """Check for anomalous gait patterns"""
        
        if 'anomaly' not in self.models:
            return None
        
        try:
            feature_vector = self._features_to_matrix([features])[0:1]
            scaler = self.scalers['anomaly']
            scaled_features = scaler.transform(feature_vector)
            
            anomaly_score = self.models['anomaly'].decision_function(scaled_features)[0]
            is_anomaly = self.models['anomaly'].predict(scaled_features)[0] == -1
            
            if is_anomaly:
                # Convert anomaly score to probability-like value
                confidence = min(abs(anomaly_score) / 2.0, 1.0)
                
                return PathologyPrediction(
                    condition="Anomalous Gait Pattern",
                    probability=float(confidence),
                    confidence=float(confidence),
                    severity="moderate" if confidence > 0.7 else "mild",
                    features_contributing=["Overall gait pattern"],
                    clinical_notes="Unusual gait pattern detected. Further clinical evaluation recommended."
                )
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
        
        return None
    
    def _features_to_matrix(self, features: List[GaitFeatureVector]) -> np.ndarray:
        """Convert feature vectors to numpy matrix"""
        
        if not features:
            return np.array([])
        
        # Extract all fields from dataclass
        feature_names = [field.name for field in features[0].__dataclass_fields__.values()]
        
        matrix = []
        for feature_vec in features:
            row = []
            for name in feature_names:
                value = getattr(feature_vec, name, 0)
                # Handle any NaN values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                row.append(float(value))
            matrix.append(row)
        
        return np.array(matrix)
    
    def _determine_severity(self, confidence: float, model_name: str) -> str:
        """Determine severity based on confidence and model type"""
        
        if confidence > 0.8:
            return "severe"
        elif confidence > 0.6:
            return "moderate"
        else:
            return "mild"
    
    def _get_contributing_features(self, features: GaitFeatureVector, 
                                 model_name: str, scaled_features: np.ndarray) -> List[str]:
        """Get features that contribute most to the prediction"""
        
        contributing = []
        
        # Use feature importance if available
        if model_name in self.feature_importances:
            importances = self.feature_importances[model_name]
            feature_names = list(features.__dataclass_fields__.keys())
            
            # Get top 5 most important features
            top_indices = np.argsort(importances)[-5:]
            contributing = [feature_names[i] for i in reversed(top_indices)]
        
        # Fallback to features with extreme values
        if not contributing:
            feature_names = list(features.__dataclass_fields__.keys())
            feature_values = [getattr(features, name, 0) for name in feature_names]
            
            # Standardize and find extreme values
            feature_array = np.array(feature_values)
            z_scores = np.abs(stats.zscore(feature_array))
            extreme_indices = np.where(z_scores > 1.5)[0]
            
            contributing = [feature_names[i] for i in extreme_indices[:5]]
        
        return contributing[:5]  # Limit to top 5
    
    def _generate_clinical_notes(self, condition: str, confidence: float, 
                               model_name: str) -> str:
        """Generate clinical interpretation notes"""
        
        condition_notes = {
            'normal': "Gait patterns within normal parameters.",
            'diabetes': "Gait alterations consistent with diabetic neuropathy. Monitor for foot ulceration risk.",
            'parkinson': "Movement patterns suggestive of parkinsonian features. Consider neurological evaluation.",
            'stroke': "Asymmetric gait patterns consistent with post-stroke hemiparesis.",
            'arthritis': "Gait compensation patterns suggesting joint pathology.",
            'foot_pain': "Altered loading patterns consistent with foot pain or dysfunction."
        }
        
        base_note = condition_notes.get(condition.lower(), f"Detected condition: {condition}")
        
        confidence_qualifier = ""
        if confidence > 0.8:
            confidence_qualifier = " High confidence in assessment."
        elif confidence > 0.6:
            confidence_qualifier = " Moderate confidence - recommend clinical correlation."
        else:
            confidence_qualifier = " Low confidence - findings require clinical interpretation."
        
        return base_note + confidence_qualifier
    
    def save_models(self, path: Optional[str] = None):
        """Save trained models and scalers"""
        
        save_path = Path(path) if path else self.model_path
        save_path.mkdir(exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_file = save_path / f"{name}_model.joblib"
            joblib.dump(model, model_file)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_file = save_path / f"{name}_scaler.joblib"
            joblib.dump(scaler, scaler_file)
        
        # Save feature importances
        if self.feature_importances:
            importance_file = save_path / "feature_importances.json"
            with open(importance_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_importances = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                  for k, v in self.feature_importances.items()}
                json.dump(json_importances, f)
        
        logger.info(f"Models saved to {save_path}")
    
    def load_models(self, path: Optional[str] = None):
        """Load pre-trained models and scalers"""
        
        load_path = Path(path) if path else self.model_path
        
        if not load_path.exists():
            logger.warning(f"Model path {load_path} does not exist")
            return
        
        # Load models
        for name in self.models.keys():
            model_file = load_path / f"{name}_model.joblib"
            if model_file.exists():
                self.models[name] = joblib.load(model_file)
                logger.info(f"Loaded model: {name}")
        
        # Load scalers
        for name in self.scalers.keys():
            scaler_file = load_path / f"{name}_scaler.joblib"
            if scaler_file.exists():
                self.scalers[name] = joblib.load(scaler_file)
        
        # Load feature importances
        importance_file = load_path / "feature_importances.json"
        if importance_file.exists():
            with open(importance_file, 'r') as f:
                self.feature_importances = json.load(f)
                # Convert lists back to numpy arrays
                for k, v in self.feature_importances.items():
                    if isinstance(v, list):
                        self.feature_importances[k] = np.array(v)

# Factory functions and utilities
def create_pathology_classifier(pretrained_models_path: Optional[str] = None) -> PathologyClassifier:
    """Create and optionally load a pathology classifier"""
    return PathologyClassifier(model_path=pretrained_models_path)

def extract_features_from_session(session_data: Dict[str, Any]) -> Optional[GaitFeatureVector]:
    """Extract features from a complete session for pathology analysis"""
    
    try:
        extractor = FeatureExtractor(sampling_rate=session_data.get('sampling_rate', 100.0))
        
        # Extract data arrays
        pressure_L = np.array(session_data['pressure_data_left'])
        pressure_R = np.array(session_data['pressure_data_right'])
        cop_L = session_data['cop_data_left']
        cop_R = session_data['cop_data_right']
        timestamps = np.array(session_data['timestamps'])
        
        # Extract features
        features = extractor.extract_features(
            pressure_L, pressure_R, cop_L, cop_R, timestamps)
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features from session: {e}")
        return None

def batch_analyze_sessions(session_files: List[str], 
                         classifier: PathologyClassifier) -> pd.DataFrame:
    """Batch analyze multiple sessions and return results DataFrame"""
    
    results = []
    
    for session_file in session_files:
        try:
            # Load session data
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Extract features
            features = extract_features_from_session(session_data)
            if features is None:
                continue
            
            # Predict pathologies
            predictions = classifier.predict_pathology(features)
            
            # Add to results
            for pred in predictions:
                result = {
                    'session_file': session_file,
                    'patient_id': session_data.get('patient_id', 'unknown'),
                    'condition': pred.condition,
                    'probability': pred.probability,
                    'confidence': pred.confidence,
                    'severity': pred.severity,
                    'top_features': ', '.join(pred.features_contributing[:3])
                }
                results.append(result)
        
        except Exception as e:
            logger.error(f"Error analyzing {session_file}: {e}")
            continue
    
    return pd.DataFrame(results)

# Example usage and integration
if __name__ == "__main__":
    # Example: Create and train a pathology classifier
    
    # This would normally use real training data
    # Here we show the API structure
    
    classifier = create_pathology_classifier()
    
    # Example feature extraction (would use real session data)
    extractor = FeatureExtractor()
    
    # Simulate some gait data
    dummy_pressure_L = np.random.rand(1000, 16) * 50  # 1000 samples, 16 sensors
    dummy_pressure_R = np.random.rand(1000, 16) * 50
    dummy_cop_L = [(0.5 + 0.1*np.sin(i*0.1), 0.3 + 0.2*i/1000) for i in range(1000)]
    dummy_cop_R = [(0.5 + 0.1*np.cos(i*0.1), 0.3 + 0.2*i/1000) for i in range(1000)]
    dummy_timestamps = np.linspace(0, 10, 1000)  # 10 second session
    
    # Extract features
    features = extractor.extract_features(
        dummy_pressure_L, dummy_pressure_R, 
        dummy_cop_L, dummy_cop_R, dummy_timestamps)
    
    print(f"Extracted features: {features}")
    
    # Predict pathology (would need trained model)
    # predictions = classifier.predict_pathology(features)
    # print(f"Pathology predictions: {predictions}")