"""
core/better_foot_shape.py
Silueta anatómicamente correcta del pie (VERSIÓN CORREGIDA)
"""

import numpy as np

def foot_outline_anatomical(left=True, samples=200):
    """
    Genera contorno anatómicamente correcto del pie.
    Coordenadas normalizadas [0,1], y=0 arriba (dedos), y=1 abajo (talón)
    """
    
    # Puntos de control para pie izquierdo (más realista)
    if left:
        control_points = np.array([
            # Dedos (arriba)
            [0.30, 0.08],  # Hallux (dedo gordo) - punta
            [0.32, 0.12],  # Entre hallux y 2do dedo
            [0.38, 0.10],  # 2do dedo
            [0.44, 0.09],  # 3er dedo
            [0.50, 0.09],  # 4to dedo
            [0.56, 0.10],  # 5to dedo
            [0.60, 0.12],  # Borde lateral de dedos
            
            # Lateral del pie (exterior)
            [0.68, 0.18],  # Metatarso 5
            [0.72, 0.28],  # Medio-lateral superior
            [0.74, 0.40],  # Medio-lateral medio
            [0.74, 0.55],  # Medio-lateral inferior
            [0.72, 0.70],  # Pre-talón lateral
            [0.68, 0.82],  # Talón lateral superior
            
            # Talón (abajo)
            [0.60, 0.92],  # Talón lateral
            [0.50, 0.95],  # Talón centro
            [0.40, 0.92],  # Talón medial
            
            # Medial del pie (interior - arco)
            [0.32, 0.82],  # Talón medial superior
            [0.28, 0.70],  # Pre-talón medial
            [0.26, 0.55],  # Arco inferior (punto más estrecho)
            [0.27, 0.40],  # Arco medio
            [0.28, 0.28],  # Arco superior
            [0.30, 0.18],  # Metatarso 1
        ])
    else:
        # Pie derecho: espejo del izquierdo
        control_points = np.array([
            [0.70, 0.08],  # Hallux derecho
            [0.68, 0.12],
            [0.62, 0.10],
            [0.56, 0.09],
            [0.50, 0.09],
            [0.44, 0.10],
            [0.40, 0.12],
            
            [0.32, 0.18],
            [0.28, 0.28],
            [0.26, 0.40],
            [0.26, 0.55],
            [0.28, 0.70],
            [0.32, 0.82],
            
            [0.40, 0.92],
            [0.50, 0.95],
            [0.60, 0.92],
            
            [0.68, 0.82],
            [0.72, 0.70],
            [0.74, 0.55],
            [0.73, 0.40],
            [0.72, 0.28],
            [0.70, 0.18],
        ])
    
    # Método simple: interpolación lineal entre puntos y suavizado
    n_points = len(control_points)
    
    # Calcular el número de puntos por segmento
    points_per_segment = max(1, samples // n_points)
    
    # Interpolar linealmente entre puntos consecutivos
    outline = []
    for i in range(n_points):
        p1 = control_points[i]
        p2 = control_points[(i + 1) % n_points]
        
        # Interpolar entre p1 y p2
        for j in range(points_per_segment):
            t = j / points_per_segment
            point = p1 * (1 - t) + p2 * t
            outline.append(point)
    
    outline = np.array(outline)
    
    # Suavizado simple con media móvil
    window_size = 5
    smooth_outline = np.zeros_like(outline)
    n = len(outline)
    
    for i in range(n):
        # Ventana circular para suavizado
        indices = [(i + j - window_size//2) % n for j in range(window_size)]
        smooth_outline[i] = np.mean(outline[indices], axis=0)
    
    # Asegurar que esté cerrado
    smooth_outline = np.vstack([smooth_outline, smooth_outline[0]])
    
    return smooth_outline


def foot_sensors_anatomical(n_sensors=24, left=True):
    """
    Distribuye sensores en posiciones anatómicamente relevantes.
    """
    if n_sensors == 16:
        # Layout NURVV típico de 16 sensores
        if left:
            sensors = np.array([
                # Talón (3 sensores)
                [0.40, 0.88],  # Talón medial
                [0.50, 0.90],  # Talón centro
                [0.60, 0.88],  # Talón lateral
                
                # Mediopié (3 sensores)
                [0.35, 0.65],  # Arco medial
                [0.50, 0.60],  # Mediopié centro
                [0.65, 0.65],  # Mediopié lateral
                
                # Antepié/Metatarsos (5 sensores)
                [0.32, 0.32],  # MTH1 (metatarso 1)
                [0.40, 0.30],  # MTH2
                [0.48, 0.28],  # MTH3
                [0.56, 0.28],  # MTH4
                [0.64, 0.30],  # MTH5
                
                # Dedos (5 sensores)
                [0.30, 0.12],  # Hallux
                [0.38, 0.10],  # 2do dedo
                [0.46, 0.10],  # 3er dedo
                [0.54, 0.10],  # 4to dedo
                [0.62, 0.12],  # 5to dedo
            ])
        else:
            # Espejo para pie derecho
            sensors = np.array([
                [0.60, 0.88], [0.50, 0.90], [0.40, 0.88],
                [0.65, 0.65], [0.50, 0.60], [0.35, 0.65],
                [0.68, 0.32], [0.60, 0.30], [0.52, 0.28], [0.44, 0.28], [0.36, 0.30],
                [0.70, 0.12], [0.62, 0.10], [0.54, 0.10], [0.46, 0.10], [0.38, 0.12],
            ])
    
    elif n_sensors == 24:
        # Layout extendido de 24 sensores - distribuidos anatómicamente
        if left:
            sensors = []
            
            # Talón (4 sensores en cruz)
            sensors.extend([
                [0.50, 0.90],  # Centro
                [0.40, 0.88],  # Medial
                [0.60, 0.88],  # Lateral
                [0.50, 0.85],  # Superior
            ])
            
            # Mediopié (6 sensores)
            sensors.extend([
                [0.35, 0.70],  # Arco medial inferior
                [0.35, 0.60],  # Arco medial superior
                [0.50, 0.65],  # Centro medio
                [0.65, 0.70],  # Lateral inferior
                [0.65, 0.60],  # Lateral superior
                [0.50, 0.55],  # Centro superior
            ])
            
            # Antepié/Metatarsos (8 sensores)
            sensors.extend([
                [0.30, 0.35],  # MTH1 posterior
                [0.32, 0.28],  # MTH1 anterior
                [0.40, 0.32],  # MTH2 posterior
                [0.42, 0.25],  # MTH2 anterior
                [0.50, 0.30],  # MTH3
                [0.58, 0.28],  # MTH4
                [0.64, 0.32],  # MTH5 posterior
                [0.66, 0.25],  # MTH5 anterior
            ])
            
            # Dedos (6 sensores)
            sensors.extend([
                [0.30, 0.12],  # Hallux
                [0.38, 0.10],  # 2do dedo
                [0.44, 0.09],  # 3er dedo
                [0.50, 0.09],  # 4to dedo
                [0.56, 0.10],  # 5to dedo pequeño
                [0.35, 0.15],  # Entre hallux y 2do
            ])
            
            sensors = np.array(sensors)
        else:
            # Espejo para derecho
            sensors = foot_sensors_anatomical(n_sensors, left=True)
            sensors[:, 0] = 1.0 - sensors[:, 0]
    
    return sensors
