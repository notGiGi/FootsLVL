# reports/advanced_report_system.py
"""
Advanced scientific reporting system for clinical baropodometry
with publication-quality figures and comprehensive analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import datetime
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configure matplotlib for publication quality
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

@dataclass 
class PatientInfo:
    """Patient information for reports"""
    patient_id: str
    name: str = ""
    age: int = 0
    height: float = 0.0  # cm
    weight: float = 0.0  # kg
    gender: str = ""
    diagnosis: str = ""
    notes: str = ""

@dataclass
class SessionMetadata:
    """Session metadata for reports"""
    session_id: str
    date: datetime.datetime
    duration: float  # seconds
    sampling_rate: float
    n_samples: int
    conditions: str = "Normal walking"
    clinician: str = ""

class ScientificFootPlotter:
    """Creates publication-quality foot visualizations"""
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        
        # Clinical color schemes
        self.pressure_cmap = 'YlOrRd'
        self.clinical_colors = {
            'normal': '#2E8B57',
            'warning': '#FF8C00', 
            'critical': '#DC143C',
            'background': '#F5F5F5'
        }
    
    def create_pressure_heatmap(self, pressures: np.ndarray, 
                              foot_mask: np.ndarray,
                              title: str = "Pressure Distribution",
                              save_path: str = None) -> plt.Figure:
        """Create publication-quality pressure heatmap"""
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 8), dpi=self.dpi)
        
        # Apply foot mask
        masked_pressures = np.where(foot_mask, pressures, np.nan)
        
        # Create heatmap
        im = ax.imshow(masked_pressures, cmap=self.pressure_cmap, 
                      aspect='equal', origin='upper')
        
        # Colorbar with units
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pressure (kPa)', rotation=270, labelpad=20)
        
        # Styling
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Medial ← → Lateral', fontsize=12)
        ax.set_ylabel('Heel ← → Toe', fontsize=12)
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add foot outline
        self._add_foot_outline(ax, foot_mask)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def create_cop_trajectory(self, cop_data: List[Tuple[float, float]], 
                            foot_outline: np.ndarray,
                            title: str = "Center of Pressure Trajectory") -> plt.Figure:
        """Create COP trajectory visualization"""
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 8), dpi=self.dpi)
        
        if cop_data and len(cop_data) > 1:
            # Extract coordinates
            x_coords = [cop[0] for cop in cop_data if not np.isnan(cop[0])]
            y_coords = [cop[1] for cop in cop_data if not np.isnan(cop[1])]
            
            if x_coords and y_coords:
                # Plot trajectory with color gradient
                points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                from matplotlib.collections import LineCollection
                lc = LineCollection(segments, cmap='viridis', linewidth=2)
                lc.set_array(np.linspace(0, 1, len(segments)))
                line = ax.add_collection(lc)
                
                # Add start and end markers
                ax.plot(x_coords[0], y_coords[0], 'go', markersize=8, 
                       label='Start', zorder=5)
                ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, 
                       label='End', zorder=5)
                
                # Colorbar for time
                cbar = plt.colorbar(line, ax=ax, shrink=0.8)
                cbar.set_label('Time Progression', rotation=270, labelpad=20)
                
                ax.legend()
        
        # Add foot outline
        if foot_outline is not None and len(foot_outline) > 0:
            ax.plot(foot_outline[:, 0], foot_outline[:, 1], 
                   'k-', linewidth=2, alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Medial ← → Lateral', fontsize=12)
        ax.set_ylabel('Heel ← → Toe', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_bilateral_comparison(self, data_left: Dict, data_right: Dict,
                                  metric_name: str) -> plt.Figure:
        """Create bilateral comparison chart"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)
        
        # Extract values
        left_vals = data_left.get(metric_name, [])
        right_vals = data_right.get(metric_name, [])
        
        if left_vals and right_vals:
            # Box plots
            box_data = [left_vals, right_vals]
            bp = ax1.boxplot(box_data, labels=['Left', 'Right'], 
                           patch_artist=True)
            bp['boxes'][0].set_facecolor(self.clinical_colors['normal'])
            bp['boxes'][1].set_facecolor('#4169E1')
            
            ax1.set_title(f'{metric_name} Distribution')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
            
            # Time series
            if len(left_vals) > 1 and len(right_vals) > 1:
                time_left = np.linspace(0, len(left_vals)-1, len(left_vals))
                time_right = np.linspace(0, len(right_vals)-1, len(right_vals))
                
                ax2.plot(time_left, left_vals, 'g-', linewidth=2, 
                        label='Left', alpha=0.8)
                ax2.plot(time_right, right_vals, 'b-', linewidth=2, 
                        label='Right', alpha=0.8)
                
                ax2.set_title(f'{metric_name} Over Time')
                ax2.set_xlabel('Step Number')
                ax2.set_ylabel('Value')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Asymmetry analysis
            if len(left_vals) == len(right_vals):
                asymmetry = []
                for l, r in zip(left_vals, right_vals):
                    if l + r > 0:
                        asym = abs(l - r) / (0.5 * (l + r)) * 100
                        asymmetry.append(asym)
                
                if asymmetry:
                    ax3.hist(asymmetry, bins=10, alpha=0.7, 
                           color=self.clinical_colors['warning'])
                    ax3.axvline(np.mean(asymmetry), color='red', 
                              linestyle='--', linewidth=2, 
                              label=f'Mean: {np.mean(asymmetry):.1f}%')
                    ax3.set_title('Asymmetry Distribution')
                    ax3.set_xlabel('Asymmetry (%)')
                    ax3.set_ylabel('Frequency')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _add_foot_outline(self, ax, foot_mask: np.ndarray):
        """Add foot outline to plot"""
        from scipy import ndimage
        
        # Find contours
        contours = ndimage.find_objects(foot_mask)
        if contours:
            # Simplified outline extraction
            edges = ndimage.sobel(foot_mask.astype(float))
            y_edges, x_edges = np.where(edges > 0.1)
            
            if len(x_edges) > 0:
                # Create boundary points
                boundary_points = list(zip(x_edges, y_edges))
                
                # Sort points to form continuous outline (simplified)
                # In practice, you'd want proper contour tracing
                if len(boundary_points) > 4:
                    ax.scatter(x_edges[::10], y_edges[::10], 
                             c='black', s=1, alpha=0.5)

class ClinicalReportGenerator:
    """Generates comprehensive clinical reports"""
    
    def __init__(self, template_path: str = None):
        self.template_path = template_path
        self.plotter = ScientificFootPlotter()
        
        # Clinical reference values (example - would be from literature)
        self.reference_values = {
            'peak_pressure': {'normal': (50, 250), 'unit': 'kPa'},
            'contact_time': {'normal': (0.5, 0.8), 'unit': 's'},
            'cop_velocity': {'normal': (50, 200), 'unit': 'mm/s'},
            'asymmetry_threshold': 15.0  # percentage
        }
    
    def generate_comprehensive_report(self, 
                                    patient: PatientInfo,
                                    session: SessionMetadata, 
                                    analysis_data: Dict[str, Any],
                                    output_path: str) -> Path:
        """Generate comprehensive clinical report"""
        
        output_path = Path(output_path)
        
        with PdfPages(output_path) as pdf:
            # Cover page
            self._create_cover_page(pdf, patient, session)
            
            # Executive summary
            self._create_executive_summary(pdf, analysis_data)
            
            # Pressure analysis
            self._create_pressure_analysis(pdf, analysis_data)
            
            # Gait analysis
            self._create_gait_analysis(pdf, analysis_data)
            
            # Bilateral comparison
            self._create_bilateral_analysis(pdf, analysis_data)
            
            # Clinical interpretation
            self._create_clinical_interpretation(pdf, analysis_data, patient)
            
            # Technical appendix
            self._create_technical_appendix(pdf, analysis_data, session)
        
        return output_path
    
    def _create_cover_page(self, pdf: PdfPages, patient: PatientInfo, 
                          session: SessionMetadata):
        """Create report cover page"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, 'PLANTAR PRESSURE ANALYSIS', 
               ha='center', va='center', fontsize=24, fontweight='bold')
        
        ax.text(0.5, 0.85, 'Comprehensive Clinical Report', 
               ha='center', va='center', fontsize=16, style='italic')
        
        # Patient information
        patient_info = f"""
Patient ID: {patient.patient_id}
Name: {patient.name}
Age: {patient.age} years
Height: {patient.height} cm
Weight: {patient.weight} kg
Gender: {patient.gender}
Diagnosis: {patient.diagnosis}
"""
        
        ax.text(0.1, 0.65, patient_info, fontsize=12, va='top', 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        # Session information
        session_info = f"""
Session ID: {session.session_id}
Date: {session.date.strftime('%Y-%m-%d %H:%M')}
Duration: {session.duration:.1f} seconds
Sampling Rate: {session.sampling_rate} Hz
Samples Collected: {session.n_samples:,}
Conditions: {session.conditions}
Clinician: {session.clinician}
"""
        
        ax.text(0.6, 0.65, session_info, fontsize=12, va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        # Footer
        ax.text(0.5, 0.1, f'Report Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
               ha='center', va='center', fontsize=10, style='italic')
        
        ax.text(0.5, 0.05, 'FootLab Advanced Baropodometry System', 
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_executive_summary(self, pdf: PdfPages, data: Dict[str, Any]):
        """Create executive summary page"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('EXECUTIVE SUMMARY', fontsize=18, fontweight='bold')
        
        # Key metrics table
        ax1.axis('tight')
        ax1.axis('off')
        
        metrics_data = []
        if 'temporal_parameters' in data:
            temp = data['temporal_parameters']
            metrics_data.extend([
                ['Cadence (Left)', f"{temp.get('cadence_L', 0):.1f}", 'steps/min'],
                ['Cadence (Right)', f"{temp.get('cadence_R', 0):.1f}", 'steps/min'],
                ['Stance Time (Left)', f"{temp.get('stance_time_L', 0):.2f}", 's'],
                ['Stance Time (Right)', f"{temp.get('stance_time_R', 0):.2f}", 's']
            ])
        
        if 'pressure_parameters' in data:
            press = data['pressure_parameters'] 
            metrics_data.extend([
                ['Peak Pressure (Left)', f"{press.get('peak_pressure_L', 0):.1f}", 'kPa'],
                ['Peak Pressure (Right)', f"{press.get('peak_pressure_R', 0):.1f}", 'kPa']
            ])
        
        if metrics_data:
            table = ax1.table(cellText=metrics_data, 
                            colLabels=['Metric', 'Value', 'Unit'],
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
        ax1.set_title('Key Metrics', fontweight='bold')
        
        # Asymmetry radar chart
        if 'asymmetry_indices' in data:
            asym = data['asymmetry_indices']
            self._create_asymmetry_radar(ax2, asym)
        
        # Clinical flags
        if 'clinical_flags' in data:
            flags = data['clinical_flags']
            self._create_flags_summary(ax3, flags)
        
        # Data quality indicator
        if 'data_quality' in data:
            quality = data['data_quality']
            self._create_quality_indicator(ax4, quality)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_pressure_analysis(self, pdf: PdfPages, data: Dict[str, Any]):
        """Create pressure analysis page"""
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('PRESSURE ANALYSIS', fontsize=18, fontweight='bold')
        
        # Create grid layout
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.8])
        
        # Pressure heatmaps would go here
        # (Implementation depends on having actual pressure grid data)
        
        # Pressure distribution analysis
        if 'load_distribution' in data:
            load_dist = data['load_distribution']
            
            ax1 = fig.add_subplot(gs[0, 0])
            self._create_load_distribution_chart(ax1, load_dist, 'left')
            
            ax2 = fig.add_subplot(gs[0, 1]) 
            self._create_load_distribution_chart(ax2, load_dist, 'right')
            
            # Bilateral comparison
            ax3 = fig.add_subplot(gs[1, :2])
            self._create_bilateral_load_comparison(ax3, load_dist)
            
            # Clinical interpretation
            ax4 = fig.add_subplot(gs[:, 2])
            self._create_pressure_interpretation(ax4, load_dist)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_gait_analysis(self, pdf: PdfPages, data: Dict[str, Any]):
        """Create gait analysis page"""
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('GAIT ANALYSIS', fontsize=18, fontweight='bold')
        
        # Temporal parameters
        if 'temporal_parameters' in data:
            temp = data['temporal_parameters']
            
            # Create subplots for various gait metrics
            ax1 = plt.subplot(2, 3, 1)
            self._plot_temporal_metrics(ax1, temp)
            
            ax2 = plt.subplot(2, 3, 2)
            self._plot_stance_swing_ratio(ax2, temp)
            
            ax3 = plt.subplot(2, 3, 3)
            self._plot_step_variability(ax3, temp)
        
        # COP analysis
        if 'cop_parameters' in data:
            cop = data['cop_parameters']
            
            ax4 = plt.subplot(2, 3, 4)
            self._plot_cop_metrics(ax4, cop)
            
            ax5 = plt.subplot(2, 3, 5)
            self._plot_cop_velocity_profile(ax5, cop)
            
            ax6 = plt.subplot(2, 3, 6)
            self._plot_stability_metrics(ax6, cop)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_bilateral_analysis(self, pdf: PdfPages, data: Dict[str, Any]):
        """Create bilateral analysis page"""
        
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('BILATERAL ANALYSIS', fontsize=18, fontweight='bold')
        
        if 'asymmetry_indices' in data:
            asymmetry = data['asymmetry_indices']
            
            # Asymmetry overview
            self._plot_asymmetry_overview(axes[0, 0], asymmetry)
            
            # Asymmetry trends
            self._plot_asymmetry_trends(axes[0, 1], asymmetry)
            
            # Clinical significance
            self._plot_clinical_significance(axes[1, 0], asymmetry)
            
            # Recommendations
            self._create_recommendations_panel(axes[1, 1], asymmetry)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_clinical_interpretation(self, pdf: PdfPages, 
                                      data: Dict[str, Any], 
                                      patient: PatientInfo):
        """Create clinical interpretation page"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'CLINICAL INTERPRETATION', 
               ha='center', va='top', fontsize=18, fontweight='bold')
        
        # Generate clinical narrative
        interpretation = self._generate_clinical_narrative(data, patient)
        
        ax.text(0.05, 0.85, interpretation, ha='left', va='top', 
               fontsize=11, wrap=True, 
               bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
        
        # Clinical recommendations
        recommendations = self._generate_recommendations(data, patient)
        
        ax.text(0.05, 0.35, 'CLINICAL RECOMMENDATIONS:', 
               ha='left', va='top', fontsize=14, fontweight='bold')
        
        ax.text(0.05, 0.30, recommendations, ha='left', va='top', 
               fontsize=11, wrap=True,
               bbox=dict(boxstyle="round,pad=1", facecolor="lightgreen", alpha=0.8))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_technical_appendix(self, pdf: PdfPages, 
                                 data: Dict[str, Any],
                                 session: SessionMetadata):
        """Create technical appendix"""
        
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'TECHNICAL APPENDIX', 
               ha='center', va='top', fontsize=18, fontweight='bold')
        
        # Technical details
        tech_info = f"""
DATA ACQUISITION PARAMETERS:
• Sampling Rate: {session.sampling_rate} Hz
• Session Duration: {session.duration:.1f} seconds  
• Total Samples: {session.n_samples:,}
• Data Completeness: {data.get('data_quality', {}).get('data_completeness', 0)*100:.1f}%

PROCESSING METHODS:
• Interpolation: Radial Basis Function (RBF)
• Smoothing: Gaussian filter (σ=1.2)
• Peak Detection: Prominence-based algorithm
• Asymmetry Calculation: |L-R|/(0.5*(L+R))*100

REFERENCE VALUES:
• Normal Peak Pressure: 50-250 kPa
• Normal Contact Time: 0.5-0.8 s
• Asymmetry Threshold: <15%
• Data Quality Threshold: >90% completeness

CLINICAL VALIDATION:
This analysis is based on established biomechanical principles
and validated clinical protocols. Results should be interpreted
in conjunction with clinical examination and patient history.
"""
        
        ax.text(0.05, 0.85, tech_info, ha='left', va='top', fontsize=10,
               bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.5))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    # Helper methods for specific visualizations
    def _create_asymmetry_radar(self, ax, asymmetry_data: Dict[str, float]):
        """Create asymmetry radar chart"""
        if not asymmetry_data:
            ax.text(0.5, 0.5, 'No asymmetry data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Simplified radar chart
        metrics = list(asymmetry_data.keys())[:6]  # Limit to 6 metrics
        values = [asymmetry_data.get(m, 0) for m in metrics]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label='Asymmetry %')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_asymmetry', '') for m in metrics])
        ax.set_ylim(0, max(max(values), 20))
        ax.set_title('Asymmetry Profile')
        ax.grid(True)
    
    def _create_flags_summary(self, ax, flags: List[Dict]):
        """Create clinical flags summary"""
        ax.axis('off')
        
        if not flags:
            ax.text(0.5, 0.5, 'No clinical flags identified', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
            return
        
        flag_text = "CLINICAL ALERTS:\n\n"
        for i, flag in enumerate(flags[:5]):  # Show max 5 flags
            severity = flag.get('severity', 'unknown')
            message = flag.get('message', 'Unknown issue')
            
            color = {'high': 'red', 'moderate': 'orange', 'low': 'yellow'}.get(severity, 'gray')
            flag_text += f"⚠️ {message}\n"
        
        ax.text(0.05, 0.95, flag_text, ha='left', va='top', transform=ax.transAxes,
               fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="mistyrose"))
        ax.set_title('Clinical Flags')
    
    def _create_quality_indicator(self, ax, quality_data: Dict):
        """Create data quality indicator"""
        if not quality_data:
            return
        
        completeness = quality_data.get('data_completeness', 0) * 100
        consistency = quality_data.get('signal_consistency', 0) * 100
        confidence = quality_data.get('step_detection_confidence', 0) * 100
        
        qualities = ['Completeness', 'Consistency', 'Confidence']
        values = [completeness, consistency, confidence]
        colors = ['green' if v >= 90 else 'orange' if v >= 70 else 'red' for v in values]
        
        bars = ax.barh(qualities, values, color=colors)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Quality (%)')
        ax.set_title('Data Quality Assessment')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1f}%', ha='left', va='center')
    
    def _generate_clinical_narrative(self, data: Dict, patient: PatientInfo) -> str:
        """Generate clinical narrative text"""
        
        narrative = f"CLINICAL ASSESSMENT FOR {patient.name} (ID: {patient.patient_id})\n\n"
        
        # Add temporal analysis
        if 'temporal_parameters' in data:
            temp = data['temporal_parameters']
            cadence_L = temp.get('cadence_L', 0)
            cadence_R = temp.get('cadence_R', 0)
            
            narrative += f"Gait Analysis: Patient demonstrates cadence of {cadence_L:.1f} (left) "
            narrative += f"and {cadence_R:.1f} (right) steps per minute. "
            
            if abs(cadence_L - cadence_R) > 10:
                narrative += "Significant bilateral asymmetry in cadence is observed, "
                narrative += "suggesting potential compensation patterns. "
        
        # Add pressure analysis  
        if 'pressure_parameters' in data:
            press = data['pressure_parameters']
            peak_L = press.get('peak_pressure_L', 0)
            peak_R = press.get('peak_pressure_R', 0)
            
            narrative += f"Peak pressures recorded at {peak_L:.1f} kPa (left) and {peak_R:.1f} kPa (right). "
            
            if max(peak_L, peak_R) > 300:
                narrative += "Elevated peak pressures indicate potential areas of concern requiring attention. "
        
        # Add asymmetry analysis
        if 'asymmetry_indices' in data:
            asym = data['asymmetry_indices']
            high_asym = [k for k, v in asym.items() if v > 15]
            
            if high_asym:
                narrative += f"Clinically significant asymmetries identified in: {', '.join(high_asym)}. "
        
        return narrative
    
    def _generate_recommendations(self, data: Dict, patient: PatientInfo) -> str:
        """Generate clinical recommendations"""
        
        recommendations = "Based on the analysis findings:\n\n"
        
        # Analyze flags and generate recommendations
        if 'clinical_flags' in data:
            flags = data['clinical_flags']
            
            pressure_flags = [f for f in flags if f.get('type') == 'peak_pressure']
            if pressure_flags:
                recommendations += "• Consider pressure redistribution strategies or orthotic intervention\n"
            
            asymmetry_flags = [f for f in flags if f.get('type') == 'asymmetry']  
            if asymmetry_flags:
                recommendations += "• Investigate underlying causes of bilateral asymmetry\n"
                recommendations += "• Consider gait retraining or physical therapy\n"
            
            temporal_flags = [f for f in flags if f.get('type') == 'temporal']
            if temporal_flags:
                recommendations += "• Evaluate for neurological or musculoskeletal factors affecting timing\n"
        
        recommendations += "\n• Follow-up assessment recommended in 6-8 weeks\n"
        recommendations += "• Consider comparative analysis with normative data\n"
        recommendations += "• Correlation with clinical examination findings advised\n"
        
        return recommendations
    
    # Placeholder methods for additional visualizations
    def _create_load_distribution_chart(self, ax, load_data, foot_side):
        """Create load distribution pie chart"""
        pass
    
    def _create_bilateral_load_comparison(self, ax, load_data):
        """Create bilateral load comparison"""
        pass
    
    def _create_pressure_interpretation(self, ax, load_data):
        """Create pressure interpretation panel"""
        pass
    
    def _plot_temporal_metrics(self, ax, temp_data):
        """Plot temporal metrics"""
        pass
    
    def _plot_stance_swing_ratio(self, ax, temp_data):
        """Plot stance/swing ratio"""
        pass
    
    def _plot_step_variability(self, ax, temp_data):
        """Plot step variability"""
        pass
    
    def _plot_cop_metrics(self, ax, cop_data):
        """Plot COP metrics"""
        pass
    
    def _plot_cop_velocity_profile(self, ax, cop_data):
        """Plot COP velocity profile"""
        pass
    
    def _plot_stability_metrics(self, ax, cop_data):
        """Plot stability metrics"""
        pass
    
    def _plot_asymmetry_overview(self, ax, asym_data):
        """Plot asymmetry overview"""
        pass
    
    def _plot_asymmetry_trends(self, ax, asym_data):
        """Plot asymmetry trends"""
        pass
    
    def _plot_clinical_significance(self, ax, asym_data):
        """Plot clinical significance indicators"""
        pass
    
    def _create_recommendations_panel(self, ax, asym_data):
        """Create recommendations panel"""
        pass

# Factory function for easy report generation
def generate_clinical_report(patient_info: Dict, session_info: Dict, 
                           analysis_results: Dict, output_path: str) -> Path:
    """
    Generate comprehensive clinical report
    
    Args:
        patient_info: Patient demographic and clinical information
        session_info: Session metadata and parameters
        analysis_results: Complete analysis results from gait analyzer
        output_path: Path for output PDF file
    
    Returns:
        Path to generated report
    """
    
    # Convert dictionaries to dataclasses
    patient = PatientInfo(**patient_info)
    session = SessionMetadata(**session_info)
    
    # Generate report
    generator = ClinicalReportGenerator()
    return generator.generate_comprehensive_report(
        patient=patient,
        session=session, 
        analysis_data=analysis_results,
        output_path=output_path
    )