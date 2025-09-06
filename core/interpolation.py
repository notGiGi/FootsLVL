# core/interpolation.py
import numpy as np
from .foot_shape import foot_outline_points, polygon_mask, cross_section

# === Sensores dentro de la silueta (en [0..1], y=0 arriba, y=1 abajo) ===
def foot_layout_24(left: bool = True, grid_ref_w: int = 200, grid_ref_h: int = 320) -> np.ndarray:
    """
    24 sensores distribuidos por filas dentro de la silueta:
    filas y (de dedos→talón): [0.16, 0.26, 0.38, 0.50, 0.68, 0.86]
    sensores por fila:        [2,    3,    4,    5,    5,    5]  = 24
    """
    poly01 = foot_outline_points(left=left, samples_per_seg=70)
    mask, _ = polygon_mask(poly01, grid_ref_w, grid_ref_h)

    y_levels = [0.16, 0.26, 0.38, 0.50, 0.68, 0.86]
    n_per    = [2,    3,    4,    5,    5,    5]

    coords = []
    for y01, n in zip(y_levels, n_per):
        yy = int(np.clip(round(y01 * grid_ref_h), 0, grid_ref_h-1))
        xs = cross_section(mask, yy)
        if xs is None:
            # busca cerca si justo esa fila está vacía
            found = False
            for off in range(1,8):
                for y2 in (yy-off, yy+off):
                    if 0 <= y2 < grid_ref_h:
                        xs = cross_section(mask, y2)
                        if xs is not None:
                            yy = y2; found = True; break
                if found: break
        if xs is None:  # fallback raro
            xs = (int(0.40*grid_ref_w), int(0.85*grid_ref_w))
        x0, x1 = xs
        margin = max(2, int(0.02*grid_ref_w))
        x0 += margin; x1 -= margin
        if x1 <= x0: x1 = x0 + 1
        x_positions = np.linspace(x0, x1, n)
        for x in x_positions:
            coords.append([(x+0.5)/grid_ref_w, (yy+0.5)/grid_ref_h])

    return np.asarray(coords[:24], float)

# === Grid en [0..1] (x horizontal, y vertical con 0 arriba) ===
def _grid_xy01(grid_w: int, grid_h: int):
    yy, xx = np.mgrid[0:grid_h, 0:grid_w]
    X = (xx + 0.5) / grid_w
    Y = (yy + 0.5) / grid_h
    return X, Y

# === Mapeo IDW k-NN coherente con el lienzo ===
def precompute_mapping(sensor_xy01: np.ndarray, grid_w: int, grid_h: int, k: int = 3, power: float = 2.0):
    H, W = grid_h, grid_w
    X, Y = _grid_xy01(W, H)                      # y=0 arriba
    P = np.stack([X.ravel(), Y.ravel()], axis=1) # (H*W,2)
    S = sensor_xy01[None, :, :]                  # (1,N,2)
    D = np.linalg.norm(P[:, None, :] - S, axis=2)  # (H*W,N)

    k = min(k, sensor_xy01.shape[0])
    idx = np.argpartition(D, kth=k-1, axis=1)[:, :k]
    d = np.take_along_axis(D, idx, axis=1)

    eps = 1e-9
    near = d <= 1e-12
    any_near = near.any(axis=1)
    w = np.empty_like(d, float)

    mask = ~any_near
    w[mask] = 1.0 / np.maximum(d[mask], eps) ** power
    w[mask] /= np.sum(w[mask], axis=1, keepdims=True)

    if any_near.any():
        rows = np.where(any_near)[0]
        w[any_near] = 0.0
        for r in rows:
            c = int(np.where(near[r])[0][0])
            w[r, c] = 1.0

    return idx.reshape(H, W, k).astype(np.int32), w.reshape(H, W, k).astype(np.float32)

def interpolate_to_grid(pressures, idx_map: np.ndarray, w_map: np.ndarray) -> np.ndarray:
    p = np.asarray(pressures, np.float32)
    return np.sum(p[idx_map] * w_map, axis=2)
