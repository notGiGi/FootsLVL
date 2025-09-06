# core/footprint.py
import numpy as np
from collections import deque

# ---------- morfología 3x3 (sin dependencias) ----------
def bin_dilate(m: np.ndarray, iters=1):
    m = m.astype(bool)
    for _ in range(iters):
        pad = np.pad(m, 1, mode='constant', constant_values=False)
        s = (
            pad[0:-2, 0:-2] | pad[0:-2, 1:-1] | pad[0:-2, 2:] |
            pad[1:-1, 0:-2] | pad[1:-1, 1:-1] | pad[1:-1, 2:] |
            pad[2:,   0:-2] | pad[2:,   1:-1] | pad[2:,   2:]
        )
        m = s
    return m

def bin_erode(m: np.ndarray, iters=1):
    m = m.astype(bool)
    for _ in range(iters):
        pad = np.pad(m, 1, mode='constant', constant_values=False)
        s = (
            pad[0:-2, 0:-2] & pad[0:-2, 1:-1] & pad[0:-2, 2:] &
            pad[1:-1, 0:-2] & pad[1:-1, 1:-1] & pad[1:-1, 2:] &
            pad[2:,   0:-2] & pad[2:,   1:-1] & pad[2:,   2:]
        )
        m = s
    return m

def morph_close(m: np.ndarray, iters=1):
    return bin_erode(bin_dilate(m, iters), iters)

# ---------- componentes conexas (mayor componente) ----------
def largest_component(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best_count, best_coords = 0, []
    neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    for y in range(H):
        for x in range(W):
            if mask[y,x] and not seen[y,x]:
                q = deque([(y,x)])
                seen[y,x] = True
                comp = [(y,x)]
                while q:
                    cy, cx = q.popleft()
                    for dy,dx in neigh:
                        ny, nx = cy+dy, cx+dx
                        if 0<=ny<H and 0<=nx<W and mask[ny,nx] and not seen[ny,nx]:
                            seen[ny,nx]=True; q.append((ny,nx)); comp.append((ny,nx))
                if len(comp) > best_count:
                    best_count = len(comp); best_coords = comp
    out = np.zeros_like(mask, dtype=bool)
    for (y,x) in best_coords:
        out[y,x] = True
    return out

# ---------- borde → polilínea sencilla ----------
def trace_outline(mask: np.ndarray):
    """
    Extrae borde como lista de puntos (x,y) ordenados (trazado Moore simple).
    Si falla, retorna None.
    """
    H, W = mask.shape
    # borde = mask & (~erode(mask))
    border = mask & (~bin_erode(mask, 1))
    ys, xs = np.where(border)
    if ys.size == 0:
        return None
    start = (int(ys[0]), int(xs[0]))
    # vecinos 8-conectados en orden horario (Moore)
    nbrs = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    # busca un vecino para arrancar
    path = []
    cur = start; prev_dir = 0
    for _ in range(H*W*4):
        path.append((cur[1], cur[0]))  # (x,y)
        # desde prev_dir-2 (giro a la izquierda) probar vecinos
        found = False
        for k in range(8):
            d = (prev_dir + 6 + k) % 8  # rotado
            ny, nx = cur[0] + nbrs[d][0], cur[1] + nbrs[d][1]
            if 0 <= ny < H and 0 <= nx < W and border[ny, nx]:
                cur = (ny, nx); prev_dir = d; found = True; break
        if not found:
            break
        if cur == start and len(path) > 20:
            break
    if len(path) < 10:
        return None
    # elimina duplicados consecutivos
    simp = [path[0]]
    for p in path[1:]:
        if p != simp[-1]:
            simp.append(p)
    return np.array(simp, dtype=float)  # en píxeles (x,y)
    
# ---------- huella desde datos ----------
def footprint_from_grid(grid: np.ndarray, percent=60.0, min_level=5.0, close_iters=1) -> tuple[np.ndarray, float]:
    """
    Calcula máscara de contacto (huella) desde el grid (H,W):
      - Umbral robusto: max(percentil, min_level) sobre valores >0.
      - Morfología (closing).
      - Mantiene la mayor componente.
    Retorna: (mask_bool, threshold_aplicado)
    """
    vals = grid[np.isfinite(grid)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.zeros_like(grid, dtype=bool), 0.0
    thr = max(float(np.percentile(vals, percent)), float(min_level))
    mask = grid >= thr
    mask = morph_close(mask, iters=close_iters)
    mask = largest_component(mask)
    return mask, thr
