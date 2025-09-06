# core/enhanced_foot_shape.py
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

@dataclass
class AnatomicalLandmarks:
    """Anatomical landmarks for realistic foot shape generation"""
    # All coordinates in normalized space [0,1] where y=0 is toes, y=1 is heel
    hallux_tip: np.ndarray
    hallux_joint: np.ndarray
    mtp1: np.ndarray  # 1st metatarsophalangeal joint
    mtp2: np.ndarray  # 2nd metatarsophalangeal joint
    mtp3: np.ndarray  # 3rd metatarsophalangeal joint
    mtp4: np.ndarray  # 4th metatarsophalangeal joint
    mtp5: np.ndarray  # 5th metatarsophalangeal joint
    navicular: np.ndarray  # Medial arch highest point
    cuboid: np.ndarray     # Lateral midfoot
    heel_medial: np.ndarray
    heel_lateral: np.ndarray
    heel_posterior: np.ndarray

def create_anatomical_landmarks(left: bool = True, foot_type: str = "normal") -> AnatomicalLandmarks:
    """
    Create anatomically accurate foot landmarks based on foot type
    foot_type: "normal", "pes_planus", "pes_cavus", "wide", "narrow"
    """
    
    # Base proportions from biomechanical literature
    base_landmarks = {
        "hallux_tip": [0.38, 0.08],
        "hallux_joint": [0.36, 0.15],
        "mtp1": [0.34, 0.25],
        "mtp2": [0.42, 0.22],
        "mtp3": [0.50, 0.20],
        "mtp4": [0.58, 0.19],
        "mtp5": [0.68, 0.18],
        "navicular": [0.28, 0.45],
        "cuboid": [0.75, 0.42],
        "heel_medial": [0.42, 0.92],
        "heel_lateral": [0.58, 0.94],
        "heel_posterior": [0.50, 0.96]
    }
    
    # Adjust based on foot type
    adjustments = {
        "pes_planus": {  # Flat foot
            "navicular": [0.32, 0.45],  # Less pronounced arch
            "mtp1": [0.32, 0.26]
        },
        "pes_cavus": {  # High arch
            "navicular": [0.25, 0.45],  # More pronounced arch
            "mtp1": [0.36, 0.24]
        },
        "wide": {  # Wide foot
            "mtp5": [0.72, 0.18],
            "heel_lateral": [0.62, 0.94],
            "cuboid": [0.78, 0.42]
        },
        "narrow": {  # Narrow foot
            "mtp5": [0.64, 0.18],
            "heel_lateral": [0.54, 0.94],
            "cuboid": [0.72, 0.42]
        }
    }
    
    # Apply adjustments
    if foot_type in adjustments:
        base_landmarks.update(adjustments[foot_type])
    
    # Mirror for right foot
    if not left:
        for key, coords in base_landmarks.items():
            base_landmarks[key] = [1.0 - coords[0], coords[1]]
    
    # Convert to numpy arrays
    landmarks = {}
    for key, coords in base_landmarks.items():
        landmarks[key] = np.array(coords, dtype=float)
    
    return AnatomicalLandmarks(**landmarks)

def generate_realistic_foot_outline(landmarks: AnatomicalLandmarks, 
                                  resolution: int = 200) -> np.ndarray:
    """
    Generate a realistic foot outline using anatomical landmarks and B-spline interpolation
    """
    
    # Define the outline path using anatomical landmarks
    outline_points = [
        landmarks.hallux_tip,
        landmarks.hallux_joint,
        landmarks.mtp1,
        landmarks.mtp2,
        landmarks.mtp3,
        landmarks.mtp4,
        landmarks.mtp5,
        landmarks.cuboid + np.array([0.05, 0.08]),  # Lateral midfoot curve
        landmarks.cuboid + np.array([0.03, 0.25]),  # Lateral arch
        landmarks.heel_lateral,
        landmarks.heel_posterior,
        landmarks.heel_medial,
        landmarks.navicular + np.array([0.02, 0.20]), # Medial arch curve
        landmarks.navicular,  # Navicular prominence
        landmarks.mtp1 + np.array([-0.02, 0.05]),  # Medial forefoot
        landmarks.hallux_joint + np.array([-0.02, -0.03]),
    ]
    
    # Convert to array
    points = np.array(outline_points)
    
    # Ensure closed curve
    points = np.vstack([points, points[0]])
    
    # Use scipy's B-spline for smooth interpolation
    try:
        # Parametric spline fitting
        tck, u = splprep([points[:, 0], points[:, 1]], s=0.01, per=True)
        
        # Generate smooth outline
        u_new = np.linspace(0, 1, resolution, endpoint=False)
        smooth_outline = np.array(splev(u_new, tck)).T
        
        # Ensure closure
        smooth_outline = np.vstack([smooth_outline, smooth_outline[0]])
        
    except Exception:
        # Fallback to original points if spline fails
        smooth_outline = points
    
    # Clamp to valid range
    smooth_outline = np.clip(smooth_outline, 0.0, 1.0)
    
    return smooth_outline

def create_pressure_zones_mask(foot_outline: np.ndarray, 
                             grid_w: int, grid_h: int) -> dict:
    """
    Create anatomically relevant pressure zones within the foot outline
    """
    from scipy.spatial import distance_matrix
    
    # Convert outline to pixel coordinates
    outline_px = foot_outline.copy()
    outline_px[:, 0] *= grid_w
    outline_px[:, 1] *= grid_h
    
    # Create base mask
    mask = polygon_to_mask(outline_px, grid_w, grid_h)
    
    # Define anatomical zones in normalized coordinates
    zones = {
        "heel": {"center": [0.5, 0.85], "radius": 0.15},
        "midfoot_medial": {"center": [0.35, 0.55], "radius": 0.12},
        "midfoot_lateral": {"center": [0.65, 0.50], "radius": 0.10},
        "forefoot_medial": {"center": [0.35, 0.30], "radius": 0.08},
        "forefoot_central": {"center": [0.50, 0.25], "radius": 0.10},
        "forefoot_lateral": {"center": [0.65, 0.25], "radius": 0.08},
        "hallux": {"center": [0.38, 0.12], "radius": 0.06},
        "lesser_toes": {"center": [0.55, 0.12], "radius": 0.12}
    }
    
    # Create grid coordinates
    y_grid, x_grid = np.mgrid[0:grid_h, 0:grid_w]
    x_norm = (x_grid + 0.5) / grid_w
    y_norm = (y_grid + 0.5) / grid_h
    
    zone_masks = {}
    for zone_name, zone_info in zones.items():
        center = np.array(zone_info["center"])
        radius = zone_info["radius"]
        
        # Calculate distance from zone center
        dist = np.sqrt((x_norm - center[0])**2 + (y_norm - center[1])**2)
        zone_mask = (dist <= radius) & mask
        zone_masks[zone_name] = zone_mask
    
    return zone_masks

def polygon_to_mask(polygon_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert polygon coordinates to binary mask using scanline algorithm
    """
    from PIL import Image, ImageDraw
    
    # Create PIL image for polygon filling
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Convert polygon to list of tuples
    polygon_tuples = [(float(p[0]), float(p[1])) for p in polygon_px]
    
    # Draw filled polygon
    draw.polygon(polygon_tuples, fill=255)
    
    # Convert to numpy boolean array
    mask_array = np.array(img) > 128
    
    return mask_array

def adaptive_sensor_placement(foot_outline: np.ndarray, 
                            n_sensors: int = 16,
                            anatomical_priority: bool = True) -> np.ndarray:
    """
    Generate adaptive sensor placement based on anatomical importance and coverage
    """
    
    if anatomical_priority:
        # Anatomically important regions (higher sensor density)
        priority_regions = [
            {"center": [0.50, 0.85], "weight": 3, "n_sensors": 3},  # Heel
            {"center": [0.35, 0.30], "weight": 2, "n_sensors": 2},  # 1st MT
            {"center": [0.50, 0.25], "weight": 2, "n_sensors": 3},  # Central MT
            {"center": [0.65, 0.25], "weight": 1, "n_sensors": 2},  # Lateral MT
            {"center": [0.38, 0.12], "weight": 2, "n_sensors": 2},  # Hallux
            {"center": [0.50, 0.55], "weight": 1, "n_sensors": 2},  # Midfoot
            {"center": [0.55, 0.12], "weight": 1, "n_sensors": 2},  # Lesser toes
        ]
        
        sensors = []
        for region in priority_regions:
            center = np.array(region["center"])
            n_reg_sensors = region["n_sensors"]
            
            if len(sensors) + n_reg_sensors > n_sensors:
                n_reg_sensors = n_sensors - len(sensors)
            
            if n_reg_sensors <= 0:
                break
                
            # Generate sensors around the region center
            if n_reg_sensors == 1:
                sensors.append(center)
            else:
                angles = np.linspace(0, 2*np.pi, n_reg_sensors, endpoint=False)
                radius = 0.08  # Small radius around center
                for angle in angles:
                    sensor_pos = center + radius * np.array([np.cos(angle), np.sin(angle)])
                    sensor_pos = np.clip(sensor_pos, 0.0, 1.0)
                    sensors.append(sensor_pos)
                    
                    if len(sensors) >= n_sensors:
                        break
            
            if len(sensors) >= n_sensors:
                break
        
        # Fill remaining sensors if needed
        while len(sensors) < n_sensors:
            sensors.append([0.5, 0.5])  # Default center position
            
        sensor_array = np.array(sensors[:n_sensors])
        
    else:
        # Grid-based placement (original approach)
        sensor_array = foot_layout_24(left=True)[:n_sensors]
    
    return sensor_array

# Integration with existing foot_shape.py
def foot_outline_points_enhanced(left: bool = True, 
                               foot_type: str = "normal",
                               samples_per_seg: int = 200) -> np.ndarray:
    """
    Enhanced version of foot_outline_points with anatomical accuracy
    """
    landmarks = create_anatomical_landmarks(left=left, foot_type=foot_type)
    outline = generate_realistic_foot_outline(landmarks, resolution=samples_per_seg)
    return outline