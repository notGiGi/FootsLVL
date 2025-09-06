# core/foot_shape.py
import numpy as np

# ---------------- Catmull-Rom cerrado ----------------
def _catmull_rom_closed(ctrl, samples_per_seg=50):
    P = np.asarray(ctrl, float); n = len(P)
    out = []
    for i in range(n):
        p0 = P[(i-1)%n]; p1 = P[i%n]; p2 = P[(i+1)%n]; p3 = P[(i+2)%n]
        for s in np.linspace(0, 1, samples_per_seg, endpoint=False):
            s2 = s*s; s3 = s2*s
            b0 = -0.5*s3 + 1.0*s2 - 0.5*s
            b1 =  1.5*s3 - 2.5*s2 + 1.0
            b2 = -1.5*s3 + 2.0*s2 + 0.5*s
            b3 =  0.5*s3 - 0.5*s2
            out.append(b0*p0 + b1*p1 + b2*p2 + b3*p3)
    out.append(out[0])
    return np.asarray(out, float)

# ---------------- Anclajes (izq). Eje Y de imagen: 0=arriba (dedos), 1=abajo (talón) ----------------
def _anchors_left():
    # x=0 medial, x=1 lateral ; y=0 dedos (arriba), y=1 talón (abajo)
    return np.array([
        [0.62,0.12],  # 0 zona dedos laterales (superior-derecha)
        [0.72,0.18],  # 1
        [0.82,0.30],  # 2 MTH5
        [0.88,0.45],  # 3 lateral antepié
        [0.88,0.62],  # 4 lateral mediopié
        [0.84,0.78],  # 5 lateral inferior
        [0.72,0.92],  # 6 talón lateral
        [0.58,0.96],  # 7 centro talón
        [0.44,0.92],  # 8 talón medial
        [0.36,0.78],  # 9 mediopié medial bajo
        [0.32,0.62],  # 10 mediopié medial (arco)
        [0.34,0.46],  # 11 transición a MTH1
        [0.38,0.34],  # 12 MTH1
        [0.40,0.20],  # 13 hallux/2º dedo
        [0.48,0.14],  # 14 3º dedo
        [0.56,0.12],  # 15 4º dedo
    ], dtype=float)

def foot_outline_points(left=True, samples_per_seg=60):
    ctrl = _anchors_left()
    if not left:
        ctrl = ctrl.copy(); ctrl[:,0] = 1.0 - ctrl[:,0]
    return _catmull_rom_closed(ctrl, samples_per_seg=samples_per_seg)  # (M,2) en [0..1], y=0 arriba

# ---------------- Polígono → máscara de píxeles ----------------
def polygon_mask(poly01, grid_w, grid_h):
    poly = poly01.copy()
    poly[:,0] *= grid_w; poly[:,1] *= grid_h
    H, W = grid_h, grid_w
    mask = np.zeros((H, W), dtype=bool)
    x = poly[:,0]; y = poly[:,1]; n = len(poly)
    for j in range(n):
        x0,y0 = x[j],y[j]; x1,y1 = x[(j+1)%n],y[(j+1)%n]
        if y0 == y1: continue
        y_min = int(max(0, np.floor(min(y0,y1))))
        y_max = int(min(H-1, np.ceil(max(y0,y1))))
        for yy in range(y_min, y_max+1):
            t = (yy + 0.5 - y0) / (y1 - y0)
            if 0.0 <= t < 1.0:
                xx = x0 + t*(x1 - x0)
                col = int(np.clip(np.floor(xx), 0, W-1))
                mask[yy, col:] ^= True
    return mask, poly  # máscara (H,W) y contorno en píxeles (M,2)

# ---------------- Corte horizontal dentro de la máscara ----------------
def cross_section(mask, y_pix):
    row = np.asarray(mask[y_pix], bool)
    xs = np.where(row)[0]
    if xs.size == 0: return None
    return int(xs.min()), int(xs.max())
