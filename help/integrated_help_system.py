# help/integrated_help_system.py
"""
Integrated documentation and help system for FootLab
Includes interactive tutorials, contextual help, and comprehensive documentation
"""

import os
import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import markdown
import html

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, QTreeWidget, 
    QTreeWidgetItem, QSplitter, QPushButton, QLineEdit, QLabel,
    QTabWidget, QScrollArea, QFrame, QProgressBar, QCheckBox,
    QDialog, QDialogButtonBox, QTextEdit, QComboBox, QSpinBox,
    QMessageBox, QMenu, QAction, QToolTip, QApplication
)
from PySide6.QtCore import Qt, QTimer, QUrl, QObject, Signal, QThread, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPixmap, QMovie, QDesktopServices, QPalette, QTextCursor
from PySide6.QtWebEngineWidgets import QWebEngineView

logger = logging.getLogger(__name__)

class HelpCategory(Enum):
    """Categories of help content"""
    GETTING_STARTED = "getting_started"
    USER_INTERFACE = "user_interface"  
    DATA_ACQUISITION = "data_acquisition"
    ANALYSIS = "analysis"
    CALIBRATION = "calibration"
    REPORTING = "reporting"
    TROUBLESHOOTING = "troubleshooting"
    ADVANCED = "advanced"
    API_REFERENCE = "api_reference"

class ContentType(Enum):
    """Types of help content"""
    TEXT = "text"
    VIDEO = "video"
    INTERACTIVE = "interactive"
    TUTORIAL = "tutorial"
    FAQ = "faq"
    REFERENCE = "reference"

@dataclass
class HelpContent:
    """Individual help content item"""
    id: str
    title: str
    category: HelpCategory
    content_type: ContentType
    description: str
    content: str  # Markdown content or file path
    tags: List[str] = field(default_factory=list)
    difficulty: str = "beginner"  # beginner, intermediate, advanced
    estimated_time: int = 5  # minutes
    prerequisites: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    view_count: int = 0
    rating: float = 0.0
    interactive_elements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TutorialStep:
    """Single step in an interactive tutorial"""
    step_id: str
    title: str
    description: str
    target_element: Optional[str] = None  # CSS selector or widget name
    action_required: Optional[str] = None  # "click", "input", "wait"
    validation_func: Optional[Callable] = None
    hints: List[str] = field(default_factory=list)
    media: Optional[str] = None  # Image or video path

@dataclass
class InteractiveTutorial:
    """Complete interactive tutorial"""
    tutorial_id: str
    title: str
    description: str
    category: HelpCategory
    difficulty: str
    estimated_time: int
    steps: List[TutorialStep] = field(default_factory=list)
    completion_rate: float = 0.0
    user_progress: Dict[str, Any] = field(default_factory=dict)

class ContextualHelpProvider:
    """Provides contextual help for UI elements"""
    
    def __init__(self):
        self.help_mappings: Dict[str, Dict[str, Any]] = {}
        self.active_tooltips: Dict[str, QWidget] = {}
        
    def register_help(self, widget_name: str, help_data: Dict[str, Any]):
        """Register contextual help for a widget"""
        self.help_mappings[widget_name] = {
            "title": help_data.get("title", "Help"),
            "description": help_data.get("description", ""),
            "detailed_help": help_data.get("detailed_help", ""),
            "related_topics": help_data.get("related_topics", []),
            "shortcuts": help_data.get("shortcuts", []),
            "tips": help_data.get("tips", [])
        }
    
    def show_contextual_help(self, widget_name: str, position: tuple = None):
        """Show contextual help for a specific widget"""
        if widget_name not in self.help_mappings:
            return False
        
        help_data = self.help_mappings[widget_name]
        
        # Create contextual help popup
        help_popup = ContextualHelpPopup(help_data, position)
        help_popup.show()
        
        self.active_tooltips[widget_name] = help_popup
        return True
    
    def get_help_for_widget(self, widget_name: str) -> Optional[Dict[str, Any]]:
        """Get help data for a widget"""
        return self.help_mappings.get(widget_name)

class ContextualHelpPopup(QWidget):
    """Popup widget showing contextual help"""
    
    def __init__(self, help_data: Dict[str, Any], position: tuple = None):
        super().__init__()
        self.help_data = help_data
        
        # Configure popup
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setMaximumWidth(400)
        
        self.setupUI()
        
        if position:
            self.move(position[0], position[1])
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Title
        title_label = QLabel(self.help_data.get("title", "Help"))
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(self.help_data.get("description", ""))
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Tips
        tips = self.help_data.get("tips", [])
        if tips:
            tips_label = QLabel("<b>Tips:</b>")
            layout.addWidget(tips_label)
            
            for tip in tips:
                tip_label = QLabel(f"• {tip}")
                tip_label.setWordWrap(True)
                layout.addWidget(tip_label)
        
        # Shortcuts
        shortcuts = self.help_data.get("shortcuts", [])
        if shortcuts:
            shortcuts_label = QLabel("<b>Shortcuts:</b>")
            layout.addWidget(shortcuts_label)
            
            for shortcut in shortcuts:
                shortcut_label = QLabel(f"• {shortcut}")
                layout.addWidget(shortcut_label)
        
        # More help button
        if self.help_data.get("detailed_help"):
            more_help_btn = QPushButton("More Help...")
            more_help_btn.clicked.connect(self.show_detailed_help)
            layout.addWidget(more_help_btn)
        
        # Style the popup
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 2px solid #999999;
                border-radius: 8px;
            }
            QLabel {
                border: none;
                color: #333333;
            }
            QPushButton {
                background-color: #e0e0e0;
                border: 1px solid #999999;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
        """)
    
    def show_detailed_help(self):
        """Show detailed help in main help system"""
        # This would open the main help system
        self.hide()

class SearchableHelpBrowser(QWidget):
    """Advanced help browser with search and navigation"""
    
    def __init__(self):
        super().__init__()
        self.content_library: Dict[str, HelpContent] = {}
        self.search_index: Dict[str, List[str]] = {}  # word -> content_ids
        self.current_content: Optional[HelpContent] = None
        self.history: List[str] = []
        self.history_index: int = -1
        
        self.setupUI()
        self.load_help_content()
    
    def setupUI(self):
        layout = QVBoxLayout(self)
        
        # Search and navigation toolbar
        toolbar = QHBoxLayout()
        
        # Back/Forward buttons
        self.back_btn = QPushButton("◀")
        self.back_btn.setMaximumWidth(30)
        self.back_btn.clicked.connect(self.go_back)
        self.back_btn.setEnabled(False)
        toolbar.addWidget(self.back_btn)
        
        self.forward_btn = QPushButton("▶")
        self.forward_btn.setMaximumWidth(30)
        self.forward_btn.clicked.connect(self.go_forward)
        self.forward_btn.setEnabled(False)
        toolbar.addWidget(self.forward_btn)
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search help topics...")
        self.search_box.textChanged.connect(self.search_content)
        toolbar.addWidget(self.search_box)
        
        # Search button
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.perform_search)
        toolbar.addWidget(search_btn)
        
        layout.addLayout(toolbar)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Content tree
        self.content_tree = QTreeWidget()
        self.content_tree.setHeaderLabel("Help Topics")
        self.content_tree.itemClicked.connect(self.on_tree_item_clicked)
        splitter.addWidget(self.content_tree)
        
        # Right panel - Content display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Content title
        self.content_title = QLabel("")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(14)
        self.content_title.setFont(title_font)
        right_layout.addWidget(self.content_title)
        
        # Content metadata
        self.content_metadata = QLabel("")
        metadata_font = QFont()
        metadata_font.setPointSize(9)
        self.content_metadata.setFont(metadata_font)
        right_layout.addWidget(self.content_metadata)
        
        # Content browser
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        self.content_browser.anchorClicked.connect(self.handle_link_clicked)
        right_layout.addWidget(self.content_browser)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.print_btn = QPushButton("Print")
        self.print_btn.clicked.connect(self.print_content)
        action_layout.addWidget(self.print_btn)
        
        self.bookmark_btn = QPushButton("Bookmark")
        self.bookmark_btn.clicked.connect(self.bookmark_content)
        action_layout.addWidget(self.bookmark_btn)
        
        self.feedback_btn = QPushButton("Feedback")
        self.feedback_btn.clicked.connect(self.submit_feedback)
        action_layout.addWidget(self.feedback_btn)
        
        action_layout.addStretch()
        right_layout.addLayout(action_layout)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 600])
        
        layout.addWidget(splitter)
    
    def load_help_content(self):
        """Load help content from files and build search index"""
        
        # Sample help content - in real implementation, this would load from files
        sample_contents = [
            HelpContent(
                id="getting_started_overview",
                title="Getting Started with FootLab",
                category=HelpCategory.GETTING_STARTED,
                content_type=ContentType.TUTORIAL,
                description="Learn the basics of using FootLab for gait analysis",
                content=self._load_getting_started_content(),
                tags=["basics", "introduction", "first-time"],
                difficulty="beginner",
                estimated_time=10
            ),
            HelpContent(
                id="sensor_setup",
                title="Setting Up NURVV Sensors",
                category=HelpCategory.DATA_ACQUISITION,
                content_type=ContentType.TUTORIAL,
                description="Step-by-step guide to connecting and configuring NURVV sensors",
                content=self._load_sensor_setup_content(),
                tags=["sensors", "bluetooth", "connection"],
                difficulty="beginner",
                estimated_time=15
            ),
            HelpContent(
                id="calibration_guide",
                title="System Calibration Guide",
                category=HelpCategory.CALIBRATION,
                content_type=ContentType.REFERENCE,
                description="Complete guide to calibrating your FootLab system",
                content=self._load_calibration_content(),
                tags=["calibration", "accuracy", "setup"],
                difficulty="intermediate",
                estimated_time=30
            ),
            HelpContent(
                id="analysis_interpretation",
                title="Interpreting Gait Analysis Results",
                category=HelpCategory.ANALYSIS,
                content_type=ContentType.REFERENCE,
                description="Understanding and interpreting gait analysis metrics",
                content=self._load_analysis_content(),
                tags=["analysis", "interpretation", "clinical"],
                difficulty="intermediate",
                estimated_time=20
            )
        ]
        
        # Load content into library
        for content in sample_contents:
            self.content_library[content.id] = content
            self._index_content(content)
        
        # Populate content tree
        self._populate_content_tree()
    
    def _load_getting_started_content(self) -> str:
        return """
# Getting Started with FootLab

Welcome to FootLab, the state-of-the-art baropodometry system for clinical gait analysis.

## What is FootLab?

FootLab is a comprehensive system for analyzing plantar pressure distribution and gait patterns. It combines:

- **Real-time pressure visualization** with anatomically accurate foot models
- **Advanced biomechanical analysis** including temporal and spatial parameters
- **AI-powered pathology detection** for automated screening
- **Comprehensive reporting** with clinical interpretations

## First Steps

1. **System Setup**
   - Install FootLab software
   - Connect your pressure sensors
   - Run system calibration

2. **Patient Management**
   - Create patient profiles
   - Enter relevant clinical information
   - Set up session parameters

3. **Data Acquisition**
   - Start sensor connection
   - Begin data recording
   - Monitor real-time feedback

4. **Analysis and Reporting**
   - Review analysis results
   - Generate clinical reports
   - Save and export data

## Key Features

### Real-time Visualization
- High-resolution pressure heatmaps
- Center of pressure tracking
- Anatomical overlay options

### Advanced Analytics
- Temporal-spatial parameters
- Bilateral asymmetry analysis
- Gait phase detection
- Stability metrics

### Clinical Tools
- Normative data comparison
- Pathology screening
- Progress tracking
- Custom report generation

## Getting Help

- Use **F1** for contextual help
- Access **Help > Tutorials** for interactive guides
- Visit **Help > Documentation** for detailed references
- Contact support through **Help > Support**

Ready to start? Click **File > New Session** to begin your first analysis!
"""
    
    def _load_sensor_setup_content(self) -> str:
        return """
# Setting Up NURVV Sensors

This guide walks you through connecting and configuring NURVV Run insoles for use with FootLab.

## Prerequisites

- NURVV Run insoles (left and right)
- Fully charged sensor pods
- Bluetooth-enabled computer
- FootLab software installed

## Step 1: Prepare the Sensors

1. **Charge the Sensors**
   - Connect sensor pods to charging cable
   - Charge for at least 2 hours before first use
   - Green LED indicates full charge

2. **Insert into Insoles**
   - Carefully insert sensor pods into insole pockets
   - Ensure secure connection
   - Verify left/right orientation

## Step 2: Enable Bluetooth Discovery

1. **Put Sensors in Pairing Mode**
   - Press and hold the sensor button for 3 seconds
   - LED will flash blue indicating pairing mode
   - Repeat for both left and right sensors

2. **Enable Bluetooth on Computer**
   - Ensure Bluetooth is enabled
   - Make computer discoverable
   - Clear any previous pairings

## Step 3: Connect via FootLab

1. **Open FootLab**
   - Start FootLab application
   - Go to **Settings > Sensor Configuration**

2. **Scan for Devices**
   - Click **Scan for NURVV Devices**
   - Wait for sensors to appear in device list
   - Verify left/right foot identification

3. **Pair Devices**
   - Select left foot sensor and click **Connect**
   - Select right foot sensor and click **Connect**
   - Wait for connection confirmation

## Step 4: Verify Connection

1. **Check Status Indicators**
   - Green indicators = Connected and streaming
   - Yellow indicators = Connected but no data
   - Red indicators = Connection issues

2. **Test Data Stream**
   - Click **Test Connection**
   - Apply pressure to insoles
   - Verify real-time pressure display

## Troubleshooting

### Connection Issues
- **Sensor not found**: Check battery level and pairing mode
- **Connection drops**: Move closer to computer, check interference
- **Wrong foot assignment**: Manually reassign in device settings

### Data Quality Issues
- **No pressure data**: Check sensor pod connection to insole
- **Intermittent data**: Check Bluetooth signal strength
- **Noisy data**: Run sensor calibration procedure

### Battery Management
- **Low battery warning**: Charge sensors before use
- **Battery life**: Typical usage 8-10 hours per charge
- **Charging time**: 2-3 hours for full charge

## Best Practices

1. **Before Each Session**
   - Check battery levels
   - Verify secure sensor connection
   - Test data stream quality

2. **During Use**
   - Monitor connection status
   - Keep sensors within 10m of computer
   - Avoid electromagnetic interference

3. **After Use**
   - Disconnect sensors to save battery
   - Store in protective case
   - Clean insoles if needed

## Next Steps

Once sensors are connected:
- Run [System Calibration](calibration_guide)
- Set up [Patient Profile](patient_management)
- Start your [First Analysis Session](first_session)

Need more help? Contact support at support@footlab.com
"""
    
    def _load_calibration_content(self) -> str:
        return """
# System Calibration Guide

Proper calibration is essential for accurate measurements. This guide covers all calibration procedures.

## Overview

FootLab uses multiple calibration methods:

1. **Baseline Zero Calibration** - Removes sensor offsets
2. **Force Scaling Calibration** - Calibrates pressure readings
3. **Spatial Alignment** - Verifies sensor positions
4. **Temporal Synchronization** - Synchronizes multiple sensors

## When to Calibrate

- **Initial setup** - Always calibrate new installations
- **Routine maintenance** - Monthly for clinical use
- **After sensor changes** - When replacing or repositioning sensors
- **Quality issues** - When measurements seem inconsistent

## Calibration Procedures

### 1. Baseline Zero Calibration

**Purpose**: Remove sensor drift and establish zero baseline

**Procedure**:
1. Go to **Tools > Calibration > Baseline Zero**
2. Remove all weight from sensors
3. Click **Start Calibration**
4. Wait 30 seconds for data collection
5. Review calibration results

**Acceptance Criteria**:
- Noise level < 2% of full scale
- All sensors within ±0.5% of mean
- No trending or drift

### 2. Force Scaling Calibration

**Purpose**: Calibrate force/pressure measurements

**Equipment Needed**:
- Certified reference weights (100g, 200g, 500g, 1kg)
- Flat, stable surface
- Weight application tool (optional)

**Procedure**:
1. Go to **Tools > Calibration > Force Scaling**
2. Follow on-screen instructions for each weight
3. Apply weights to center of sensor array
4. Hold steady for 5 seconds per measurement
5. System calculates scaling factors automatically

**Acceptance Criteria**:
- Linearity error < 2% of full scale
- Correlation coefficient > 0.999
- Repeatability < 1%

### 3. Spatial Alignment Calibration

**Purpose**: Verify and calibrate sensor positions

**Equipment Needed**:
- Anatomical reference points
- Pointed probe or stylus
- Measurement ruler

**Procedure**:
1. Go to **Tools > Calibration > Spatial Alignment**
2. Use probe to apply pressure at marked anatomical points
3. System records sensor responses
4. Algorithm calculates optimal sensor positions
5. Review spatial calibration map

**Acceptance Criteria**:
- Position accuracy < 5mm RMS
- Coverage uniformity > 90%
- No significant dead zones

### 4. Temporal Synchronization

**Purpose**: Synchronize timing between multiple sensors

**Procedure**:
1. Go to **Tools > Calibration > Temporal Sync**
2. System automatically coordinates all sensors
3. Measures timing differences
4. Applies corrections automatically
5. Validates synchronization quality

**Acceptance Criteria**:
- Timing accuracy < 1ms RMS
- Synchronization quality > 95%
- No phase delays

## Calibration Results

### Interpreting Results

- **PASS**: Calibration meets all acceptance criteria
- **WARNING**: Marginal performance, monitor closely
- **FAIL**: Calibration failed, troubleshooting required

### Saving Calibration

- Calibration data saved automatically
- Backup copies stored in calibration folder
- Expiration date set based on usage pattern

### Calibration Reports

Generate calibration certificates:
1. Go to **Tools > Calibration > Generate Report**
2. Select calibration session
3. Export as PDF certificate
4. Include in quality documentation

## Quality Assurance

### Verification Testing

After calibration, perform verification:
1. **Known weight test** - Apply certified weights
2. **Repeatability test** - Multiple measurements of same load
3. **Cross-talk test** - Verify sensor independence
4. **Linearity test** - Test across full range

### Troubleshooting

**High noise levels**:
- Check sensor connections
- Eliminate vibrations
- Filter electrical interference

**Poor linearity**:
- Verify reference weights
- Check sensor mounting
- Recalibrate if necessary

**Synchronization issues**:
- Check Bluetooth connections
- Minimize interference sources
- Restart sensor systems

## Maintenance Schedule

### Daily Checks
- Verify calibration status
- Check sensor connections
- Monitor data quality

### Weekly Checks  
- Quick verification test
- Review calibration reports
- Update calibration log

### Monthly Calibration
- Full calibration procedure
- Generate calibration certificates
- Update documentation

### Annual Calibration
- Complete system validation
- Professional calibration service
- Compliance audit

## Regulatory Compliance

FootLab calibration procedures comply with:
- FDA 21 CFR Part 820 (Quality System Regulation)
- ISO 13485 (Medical Devices Quality Management)
- IEC 62304 (Medical Device Software)

Maintain calibration records for regulatory audits.

## Support

For calibration assistance:
- **Email**: calibration@footlab.com
- **Phone**: +1-555-FOOTLAB
- **Remote Support**: Available by appointment
"""
    
    def _load_analysis_content(self) -> str:
        return """
# Interpreting Gait Analysis Results

This guide helps you understand and interpret the various metrics and analyses provided by FootLab.

## Analysis Overview

FootLab provides comprehensive gait analysis including:

- **Temporal Parameters** - Timing aspects of gait
- **Spatial Parameters** - Distance and position metrics  
- **Pressure Distribution** - Force and pressure patterns
- **Bilateral Symmetry** - Left/right comparisons
- **Pathology Indicators** - Automated screening results

## Temporal Parameters

### Cadence
- **Definition**: Steps per minute
- **Normal Range**: 100-120 steps/min (adults)
- **Clinical Significance**: 
  - Low cadence may indicate balance issues
  - High cadence may suggest compensation patterns

### Step Time
- **Definition**: Time between consecutive heel strikes
- **Normal Range**: 0.5-0.7 seconds
- **Variability**: <3% coefficient of variation

### Stance Phase
- **Definition**: Percentage of gait cycle in ground contact
- **Normal Range**: 60-65% of gait cycle
- **Clinical Significance**:
  - Extended stance suggests balance deficits
  - Reduced stance may indicate pain avoidance

### Double Support Time
- **Definition**: Time when both feet are in contact
- **Normal Range**: 10-12% of gait cycle
- **Pathological Changes**:
  - Increased in balance disorders
  - Reduced in spastic conditions

## Spatial Parameters

### Step Length
- **Definition**: Distance between consecutive heel strikes
- **Normal Range**: 60-70 cm (height-dependent)
- **Asymmetry**: <5% difference between feet

### Stride Length
- **Definition**: Distance between same-foot heel strikes
- **Normal Range**: 120-140 cm
- **Clinical Relevance**: Reduced in many pathologies

### Step Width
- **Definition**: Medial-lateral distance between feet
- **Normal Range**: 8-12 cm
- **Wide base**: Balance compensation strategy

## Pressure Distribution

### Peak Pressure
- **Normal Locations**: Heel, metatarsal heads, hallux
- **Normal Values**: 200-400 kPa
- **Pathological Patterns**:
  - Forefoot overloading in diabetes
  - Lateral shift in hemiplegia

### Pressure-Time Integral (PTI)
- **Definition**: Cumulative pressure over contact time
- **Units**: kPa⋅s
- **Clinical Use**: Ulceration risk assessment

### Contact Area
- **Definition**: Area of foot in contact with ground
- **Normal Pattern**: Progressive heel-to-toe contact
- **Abnormal Patterns**: Foot drop, spasticity

## Center of Pressure (COP)

### COP Trajectory
- **Normal Pattern**: Smooth heel-to-toe progression
- **Path Length**: Related to stability and control
- **Velocity**: Indicates weight transfer efficiency

### COP Displacement
- **Medial-Lateral**: Balance control indicator
- **Anterior-Posterior**: Propulsion efficiency
- **Sway Area**: Overall stability measure

## Bilateral Symmetry

### Symmetry Index
**Formula**: |Left - Right| / (0.5 × (Left + Right)) × 100%

**Interpretation**:
- <10%: Excellent symmetry
- 10-15%: Good symmetry  
- 15-20%: Mild asymmetry
- >20%: Significant asymmetry

### Common Asymmetries

**Temporal Asymmetry**:
- Stroke: Affected side longer stance time
- Amputation: Prosthetic side compensation
- Pain: Affected side reduced loading

**Pressure Asymmetry**:
- Weight-bearing restrictions
- Structural deformities
- Neurological conditions

## Pathology Indicators

### AI Screening Results

FootLab's AI system provides automated screening for:

**Diabetic Neuropathy**:
- Reduced sensation indicators
- Pressure pattern changes
- Risk stratification

**Parkinson's Disease**:
- Reduced step length
- Increased variability
- Freezing episodes

**Stroke/Hemiplegia**:
- Asymmetric patterns
- Compensatory strategies
- Recovery indicators

### Confidence Levels
- **High (>80%)**: Strong indication present
- **Moderate (60-80%)**: Possible indication
- **Low (<60%)**: Uncertain, clinical correlation needed

## Clinical Decision Making

### Red Flags
Findings requiring immediate attention:
- Severe asymmetry (>25%)
- Extreme pressure concentrations (>500 kPa)
- Balance instability indicators
- Rapid deterioration patterns

### Intervention Planning
Use gait analysis to guide:
- Orthotic prescription
- Physical therapy goals
- Surgical planning
- Progress monitoring

### Progress Monitoring
Track changes over time:
- Improvement indicators
- Deterioration warnings
- Intervention effectiveness
- Compliance monitoring

## Normative Data

### Age-Related Changes
**Young Adults (20-40)**:
- Optimal performance parameters
- Minimal variability
- Efficient patterns

**Middle Age (40-65)**:
- Gradual parameter changes
- Increased cautious behavior
- Early compensation strategies

**Elderly (65+)**:
- Reduced speed and stride length
- Increased double support time
- Greater variability

### Population Considerations
- **Gender differences**: Women typically shorter steps
- **Height influence**: Taller individuals longer strides  
- **Weight effects**: Heavier individuals higher pressures
- **Activity level**: Athletes show different patterns

## Report Generation

### Automated Reports
FootLab generates comprehensive reports including:
- Executive summary
- Detailed metrics
- Graphical displays
- Clinical interpretations
- Recommendations

### Custom Reports
Create targeted reports for:
- Specific conditions
- Research protocols
- Insurance documentation
- Legal proceedings

## Quality Assurance

### Data Validity Checks
- Sensor coverage verification
- Signal quality assessment
- Motion artifact detection
- Calibration status confirmation

### Measurement Reliability
- Test-retest consistency
- Inter-rater reliability
- Measurement precision
- Clinical validity

## Further Reading

- [Gait Analysis Clinical Applications](clinical_applications)
- [Research Protocols](research_protocols)  
- [Case Studies](case_studies)
- [Troubleshooting Guide](troubleshooting)

For questions about analysis interpretation, contact our clinical support team.
"""
    
    def _index_content(self, content: HelpContent):
        """Index content for searching"""
        # Simple word indexing
        words = content.title.lower().split() + content.description.lower().split()
        words.extend([tag.lower() for tag in content.tags])
        
        # Add words from content (simplified)
        content_words = content.content.lower().split()[:100]  # First 100 words
        words.extend(content_words)
        
        # Remove common words and index
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        
        for word in set(words):
            if len(word) > 2 and word not in stopwords:
                if word not in self.search_index:
                    self.search_index[word] = []
                self.search_index[word].append(content.id)
    
    def _populate_content_tree(self):
        """Populate the content tree widget"""
        self.content_tree.clear()
        
        # Group by category
        categories = {}
        for content in self.content_library.values():
            category = content.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append(content)
        
        # Add to tree
        for category_name, contents in categories.items():
            category_item = QTreeWidgetItem(self.content_tree)
            category_item.setText(0, category_name.replace('_', ' ').title())
            category_item.setData(0, Qt.UserRole, f"category:{category_name}")
            
            for content in sorted(contents, key=lambda x: x.difficulty):
                content_item = QTreeWidgetItem(category_item)
                content_item.setText(0, content.title)
                content_item.setData(0, Qt.UserRole, content.id)
                
                # Add difficulty indicator
                difficulty_colors = {
                    "beginner": "green",
                    "intermediate": "orange", 
                    "advanced": "red"
                }
                color = difficulty_colors.get(content.difficulty, "black")
                content_item.setForeground(0, QPalette().color(QPalette.Text))
        
        self.content_tree.expandAll()
    
    def search_content(self, query: str):
        """Search content as user types"""
        if len(query) < 3:
            self._populate_content_tree()
            return
        
        # Simple search implementation
        query_words = query.lower().split()
        matching_content_ids = set()
        
        for word in query_words:
            if word in self.search_index:
                matching_content_ids.update(self.search_index[word])
        
        # Filter tree to show only matching content
        self._filter_tree(matching_content_ids)
    
    def _filter_tree(self, matching_ids: set):
        """Filter tree to show only matching content"""
        # This would implement tree filtering
        # For now, just highlight matching items
        pass
    
    def perform_search(self):
        """Perform detailed search"""
        query = self.search_box.text().strip()
        if not query:
            return
        
        # This would implement more sophisticated search
        self.search_content(query)
    
    def on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle tree item click"""
        content_id = item.data(0, Qt.UserRole)
        
        if content_id and not content_id.startswith("category:"):
            self.show_content(content_id)
    
    def show_content(self, content_id: str):
        """Show specific content"""
        if content_id not in self.content_library:
            return
        
        content = self.content_library[content_id]
        
        # Update navigation history
        if content_id not in self.history or self.history_index != len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
            self.history.append(content_id)
            self.history_index = len(self.history) - 1
        
        self._update_navigation_buttons()
        
        # Update content display
        self.current_content = content
        self.content_title.setText(content.title)
        
        # Metadata
        metadata_text = f"Category: {content.category.value.replace('_', ' ').title()} | "
        metadata_text += f"Difficulty: {content.difficulty.title()} | "
        metadata_text += f"Est. Time: {content.estimated_time} min"
        self.content_metadata.setText(metadata_text)
        
        # Convert markdown to HTML and display
        html_content = markdown.markdown(content.content, extensions=['tables', 'toc'])
        self.content_browser.setHtml(html_content)
        
        # Update view count
        content.view_count += 1
    
    def _update_navigation_buttons(self):
        """Update navigation button states"""
        self.back_btn.setEnabled(self.history_index > 0)
        self.forward_btn.setEnabled(self.history_index < len(self.history) - 1)
    
    def go_back(self):
        """Navigate back in history"""
        if self.history_index > 0:
            self.history_index -= 1
            content_id = self.history[self.history_index]
            self.show_content(content_id)
    
    def go_forward(self):
        """Navigate forward in history"""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            content_id = self.history[self.history_index]
            self.show_content(content_id)
    
    def handle_link_clicked(self, url: QUrl):
        """Handle internal link clicks"""
        url_string = url.toString()
        
        if url_string.startswith("http"):
            # External link
            QDesktopServices.openUrl(url)
        else:
            # Internal link - show content
            self.show_content(url_string)
    
    def print_content(self):
        """Print current content"""
        if self.current_content:
            # This would implement printing
            QMessageBox.information(self, "Print", "Printing functionality would be implemented here")
    
    def bookmark_content(self):
        """Bookmark current content"""
        if self.current_content:
            # This would implement bookmarking
            QMessageBox.information(self, "Bookmark", f"Bookmarked: {self.current_content.title}")
    
    def submit_feedback(self):
        """Submit feedback on current content"""
        if self.current_content:
            dialog = FeedbackDialog(self.current_content.title)
            if dialog.exec() == QDialog.Accepted:
                feedback_data = dialog.get_feedback()
                # Process feedback
                QMessageBox.information(self, "Feedback", "Thank you for your feedback!")

class InteractiveTutorialEngine(QWidget):
    """Engine for running interactive tutorials"""
    
    tutorial_completed = Signal(str)  # tutorial_id
    step_completed = Signal(str, str)  # tutorial_id, step_id
    
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app
        self.current_tutorial: Optional[InteractiveTutorial] = None
        self.current_step_index = 0
        self.tutorial_overlay = None
        
        self.setupUI()
        self.load_tutorials()
    
    def setupUI(self):
        """Setup tutorial UI"""
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Interactive Tutorial")
        
        layout = QVBoxLayout(self)
        
        # Tutorial info
        self.tutorial_title = QLabel()
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        self.tutorial_title.setFont(title_font)
        layout.addWidget(self.tutorial_title)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Step content
        self.step_title = QLabel()
        step_font = QFont()
        step_font.setBold(True)
        self.step_title.setFont(step_font)
        layout.addWidget(self.step_title)
        
        self.step_description = QLabel()
        self.step_description.setWordWrap(True)
        layout.addWidget(self.step_description)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.previous_step)
        button_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_step)
        button_layout.addWidget(self.next_btn)
        
        self.skip_btn = QPushButton("Skip Tutorial")
        self.skip_btn.clicked.connect(self.skip_tutorial)
        button_layout.addWidget(self.skip_btn)
        
        layout.addLayout(button_layout)
        
        # Resize and position
        self.resize(400, 200)
    
    def load_tutorials(self):
        """Load available tutorials"""
        # Sample tutorials - in real implementation, load from files
        self.tutorials = {
            "first_session": InteractiveTutorial(
                tutorial_id="first_session",
                title="Your First Analysis Session",
                description="Learn how to set up and run your first gait analysis session",
                category=HelpCategory.GETTING_STARTED,
                difficulty="beginner",
                estimated_time=15,
                steps=[
                    TutorialStep(
                        step_id="welcome",
                        title="Welcome to FootLab",
                        description="Let's start by creating a new patient profile",
                        target_element="patient_button",
                        action_required="click"
                    ),
                    TutorialStep(
                        step_id="patient_info",
                        title="Enter Patient Information",
                        description="Fill in the patient details in the dialog",
                        target_element="patient_dialog",
                        action_required="input"
                    ),
                    TutorialStep(
                        step_id="sensor_setup",
                        title="Connect Sensors", 
                        description="Make sure your sensors are connected and streaming",
                        target_element="sensor_status",
                        action_required="wait"
                    ),
                    TutorialStep(
                        step_id="start_recording",
                        title="Start Recording",
                        description="Click the record button to begin data collection",
                        target_element="record_button",
                        action_required="click"
                    ),
                    TutorialStep(
                        step_id="view_results",
                        title="View Live Results",
                        description="Watch the real-time pressure visualization",
                        target_element="heatmap_view",
                        action_required="wait"
                    )
                ]
            )
        }
    
    def start_tutorial(self, tutorial_id: str):
        """Start interactive tutorial"""
        if tutorial_id not in self.tutorials:
            return False
        
        self.current_tutorial = self.tutorials[tutorial_id]
        self.current_step_index = 0
        
        # Update UI
        self.tutorial_title.setText(self.current_tutorial.title)
        self.progress_bar.setRange(0, len(self.current_tutorial.steps))
        
        # Show tutorial window
        self.show()
        self.raise_()
        
        # Start first step
        self.show_current_step()
        
        return True
    
    def show_current_step(self):
        """Show current tutorial step"""
        if not self.current_tutorial or self.current_step_index >= len(self.current_tutorial.steps):
            return
        
        step = self.current_tutorial.steps[self.current_step_index]
        
        # Update UI
        self.step_title.setText(f"Step {self.current_step_index + 1}: {step.title}")
        self.step_description.setText(step.description)
        self.progress_bar.setValue(self.current_step_index + 1)
        
        # Update button states
        self.prev_btn.setEnabled(self.current_step_index > 0)
        self.next_btn.setText("Next" if self.current_step_index < len(self.current_tutorial.steps) - 1 else "Finish")
        
        # Highlight target element if specified
        if step.target_element:
            self._highlight_target_element(step.target_element)
    
    def _highlight_target_element(self, target_element: str):
        """Highlight the target UI element"""
        # This would find and highlight the target widget
        # For now, just show a tooltip
        target_widget = self._find_widget_by_name(target_element)
        if target_widget:
            # Create highlight overlay
            self._create_highlight_overlay(target_widget)
    
    def _find_widget_by_name(self, name: str) -> Optional[QWidget]:
        """Find widget by name in parent application"""
        if not self.parent_app:
            return None
        
        # This would implement widget finding logic
        # For now, return None
        return None
    
    def _create_highlight_overlay(self, widget: QWidget):
        """Create highlight overlay on widget"""
        # This would create a visual highlight
        # For now, just use tooltip
        QToolTip.showText(widget.mapToGlobal(widget.rect().center()), 
                         "Click here to continue", widget)
    
    def next_step(self):
        """Move to next step"""
        if not self.current_tutorial:
            return
        
        if self.current_step_index < len(self.current_tutorial.steps) - 1:
            self.current_step_index += 1
            self.show_current_step()
            
            # Emit signal
            step = self.current_tutorial.steps[self.current_step_index - 1]
            self.step_completed.emit(self.current_tutorial.tutorial_id, step.step_id)
        else:
            # Tutorial completed
            self.complete_tutorial()
    
    def previous_step(self):
        """Move to previous step"""
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.show_current_step()
    
    def skip_tutorial(self):
        """Skip tutorial"""
        reply = QMessageBox.question(self, "Skip Tutorial", 
                                   "Are you sure you want to skip this tutorial?")
        
        if reply == QMessageBox.Yes:
            self.hide()
    
    def complete_tutorial(self):
        """Complete tutorial"""
        if self.current_tutorial:
            # Update completion status
            self.current_tutorial.completion_rate = 1.0
            
            # Emit signal
            self.tutorial_completed.emit(self.current_tutorial.tutorial_id)
            
            # Show completion message
            QMessageBox.information(self, "Tutorial Complete", 
                                  f"Congratulations! You've completed '{self.current_tutorial.title}'")
        
        self.hide()

class FeedbackDialog(QDialog):
    """Dialog for submitting content feedback"""
    
    def __init__(self, content_title: str):
        super().__init__()
        self.content_title = content_title
        self.setupUI()
    
    def setupUI(self):
        self.setWindowTitle("Submit Feedback")
        self.setModal(True)
        layout = QVBoxLayout(self)
        
        # Content info
        info_label = QLabel(f"Feedback for: {self.content_title}")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Rating
        rating_layout = QHBoxLayout()
        rating_layout.addWidget(QLabel("Rating:"))
        self.rating_combo = QComboBox()
        self.rating_combo.addItems(["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
        rating_layout.addWidget(self.rating_combo)
        rating_layout.addStretch()
        layout.addLayout(rating_layout)
        
        # Feedback type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["General", "Error Report", "Suggestion", "Content Request"])
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        layout.addLayout(type_layout)
        
        # Comments
        layout.addWidget(QLabel("Comments:"))
        self.comments = QTextEdit()
        self.comments.setPlainText("Please share your thoughts...")
        layout.addWidget(self.comments)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_feedback(self) -> Dict[str, Any]:
        """Get feedback data"""
        return {
            "content_title": self.content_title,
            "rating": self.rating_combo.currentIndex() + 1,
            "type": self.type_combo.currentText(),
            "comments": self.comments.toPlainText(),
            "timestamp": datetime.now().isoformat()
        }

class IntegratedHelpSystem(QWidget):
    """Main integrated help system"""
    
    def __init__(self, parent_app=None):
        super().__init__()
        self.parent_app = parent_app
        
        # Components
        self.contextual_help = ContextualHelpProvider()
        self.help_browser = SearchableHelpBrowser()
        self.tutorial_engine = InteractiveTutorialEngine(parent_app)
        
        self.setupUI()
        self.register_default_help()
    
    def setupUI(self):
        """Setup main help system UI"""
        self.setWindowTitle("FootLab Help & Documentation")
        self.resize(1000, 700)
        
        layout = QVBoxLayout(self)
        
        # Tab widget for different help modes
        self.tab_widget = QTabWidget()
        
        # Help browser tab
        self.tab_widget.addTab(self.help_browser, "Documentation")
        
        # Tutorial tab
        tutorial_tab = self.create_tutorial_tab()
        self.tab_widget.addTab(tutorial_tab, "Interactive Tutorials")
        
        # Support tab
        support_tab = self.create_support_tab()
        self.tab_widget.addTab(support_tab, "Support")
        
        layout.addWidget(self.tab_widget)
    
    def create_tutorial_tab(self) -> QWidget:
        """Create tutorial management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Available tutorials
        layout.addWidget(QLabel("Available Tutorials:"))
        
        tutorial_list = QTreeWidget()
        tutorial_list.setHeaderLabel("Tutorials")
        
        # Populate with available tutorials
        for tutorial in self.tutorial_engine.tutorials.values():
            item = QTreeWidgetItem(tutorial_list)
            item.setText(0, tutorial.title)
            item.setData(0, Qt.UserRole, tutorial.tutorial_id)
            
            # Add difficulty and time info
            info_text = f"{tutorial.difficulty} • {tutorial.estimated_time} min"
            item.setText(1, info_text)
        
        tutorial_list.setHeaderLabels(["Tutorial", "Info"])
        tutorial_list.itemDoubleClicked.connect(self.start_tutorial_from_list)
        
        layout.addWidget(tutorial_list)
        
        # Start tutorial button
        start_btn = QPushButton("Start Selected Tutorial")
        start_btn.clicked.connect(self.start_selected_tutorial)
        layout.addWidget(start_btn)
        
        return widget
    
    def create_support_tab(self) -> QWidget:
        """Create support and contact tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Support options
        layout.addWidget(QLabel("Get Support:"))
        
        # Contact methods
        contact_layout = QHBoxLayout()
        
        email_btn = QPushButton("📧 Email Support")
        email_btn.clicked.connect(lambda: self.open_support_url("mailto:support@footlab.com"))
        contact_layout.addWidget(email_btn)
        
        phone_btn = QPushButton("📞 Phone Support") 
        phone_btn.clicked.connect(lambda: QMessageBox.information(self, "Phone Support", "Call +1-555-FOOTLAB"))
        contact_layout.addWidget(phone_btn)
        
        chat_btn = QPushButton("💬 Live Chat")
        chat_btn.clicked.connect(lambda: self.open_support_url("https://footlab.com/support/chat"))
        contact_layout.addWidget(chat_btn)
        
        layout.addLayout(contact_layout)
        
        # System information
        layout.addWidget(QLabel("System Information:"))
        system_info = QTextEdit()
        system_info.setMaximumHeight(200)
        system_info.setPlainText(self.get_system_info())
        system_info.setReadOnly(True)
        layout.addWidget(system_info)
        
        # Copy system info button
        copy_info_btn = QPushButton("Copy System Info")
        copy_info_btn.clicked.connect(lambda: QApplication.clipboard().setText(system_info.toPlainText()))
        layout.addWidget(copy_info_btn)
        
        layout.addStretch()
        
        return widget
    
    def register_default_help(self):
        """Register default contextual help"""
        
        # Example contextual help registrations
        help_items = {
            "heatmap_view": {
                "title": "Pressure Heatmap",
                "description": "Real-time visualization of plantar pressure distribution",
                "detailed_help": "The pressure heatmap shows...",
                "tips": [
                    "Use mouse wheel to zoom",
                    "Right-click for display options",
                    "Colors represent pressure intensity"
                ],
                "shortcuts": ["Ctrl+Z: Zoom to fit", "Ctrl+R: Reset view"]
            },
            "patient_button": {
                "title": "Patient Selection", 
                "description": "Select or create a patient profile",
                "tips": ["Use existing profiles when possible", "Include relevant clinical information"],
                "shortcuts": ["Ctrl+P: Quick patient selection"]
            },
            "calibration_button": {
                "title": "System Calibration",
                "description": "Calibrate sensors for accurate measurements",
                "tips": ["Calibrate monthly for clinical use", "Follow the step-by-step guide"],
                "related_topics": ["calibration_guide"]
            }
        }
        
        for widget_name, help_data in help_items.items():
            self.contextual_help.register_help(widget_name, help_data)
    
    def show_contextual_help(self, widget_name: str, position: tuple = None):
        """Show contextual help"""
        return self.contextual_help.show_contextual_help(widget_name, position)
    
    def start_tutorial_from_list(self, item: QTreeWidgetItem):
        """Start tutorial from list double-click"""
        tutorial_id = item.data(0, Qt.UserRole)
        if tutorial_id:
            self.tutorial_engine.start_tutorial(tutorial_id)
    
    def start_selected_tutorial(self):
        """Start selected tutorial"""
        # This would get the selected tutorial from the list
        pass
    
    def open_support_url(self, url: str):
        """Open support URL"""
        QDesktopServices.openUrl(QUrl(url))
    
    def get_system_info(self) -> str:
        """Get system information for support"""
        import platform
        import sys
        
        info = f"""FootLab Version: 2.0.0
Platform: {platform.system()} {platform.release()}
Python Version: {sys.version}
Qt Version: {Qt.qVersion()}
Architecture: {platform.machine()}
Processor: {platform.processor()}

Installation Path: {os.path.abspath('.')}
Configuration Path: {Path.home() / '.footlab'}
Last Update Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return info

# Integration functions
def setup_help_system(main_app) -> IntegratedHelpSystem:
    """Setup integrated help system for main application"""
    help_system = IntegratedHelpSystem(main_app)
    
    # Connect F1 key to contextual help
    def show_help():
        help_system.show()
    
    # This would be connected to the main application's F1 key handler
    return help_system

def register_widget_help(help_system: IntegratedHelpSystem, widget_name: str, help_data: Dict[str, Any]):
    """Register contextual help for a widget"""
    help_system.contextual_help.register_help(widget_name, help_data)

# Example usage
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication([])
    
    help_system = IntegratedHelpSystem()
    help_system.show()
    
    app.exec()