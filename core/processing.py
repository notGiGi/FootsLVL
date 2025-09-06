import numpy as np
from dataclasses import dataclass, field

@dataclass
class FootState:
    # Estados para un pie
    last_t: int = 0
    baseline: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=float))
    pti_heel: float = 0.0
    pti_mid: float = 0.0
    pti_fore: float = 0.0
    grf_hist: list = field(default_factory=list)
    time_hist: list = field(default_factory=list)
    step_times: list = field(default_factory=list)
    in_contact: bool = False

def center_of_pressure(pressures: np.ndarray, coords_xy: np.ndarray):
    s = pressures.sum()
    if s <= 1e-9:
        return np.nan, np.nan
    xy = (coords_xy * pressures[:, None]).sum(axis=0) / s
    return float(xy[0]), float(xy[1])

def grf(pressures: np.ndarray, scale: float = 1.0):
    return float(scale * pressures.sum())

def detect_step(g: float, threshold_on: float, threshold_off: float, in_contact: bool):
    """
    Histeresis simple: entra en contacto al superar threshold_on y suelta bajo threshold_off.
    """
    if not in_contact and g > threshold_on:
        return True, True   # contact started, in_contact now True
    if in_contact and g < threshold_off:
        return True, False  # contact ended, in_contact now False
    return False, in_contact

def zones_mask(coords_xy: np.ndarray):
    """
    Divide por filas: heel (<0.33), mid (0.33-0.66), fore (>0.66) en eje y.
    """
    y = coords_xy[:, 1]
    heel = y < 0.33
    mid  = (y >= 0.33) & (y < 0.66)
    fore = y >= 0.66
    return heel, mid, fore

def update_pti(state: FootState, pressures: np.ndarray, dt: float, masks):
    heel, mid, fore = masks
    state.pti_heel += float(pressures[heel].sum() * dt)
    state.pti_mid  += float(pressures[mid].sum()  * dt)
    state.pti_fore += float(pressures[fore].sum() * dt)

def cadence_from_steps(step_times: list, window: float = 10.0):
    """
    Cadencia (pasos/min) usando últimos 'window' segundos de eventos "contact start".
    """
    if len(step_times) < 2:
        return 0.0
    # usar últimos eventos
    t_end = step_times[-1]
    recent = [t for t in step_times if t_end - t <= window]
    if len(recent) < 2:
        return 0.0
    intervals = np.diff(recent)
    if len(intervals) == 0:
        return 0.0
    mean_interval = float(np.mean(intervals))
    if mean_interval <= 1e-9:
        return 0.0
    steps_per_sec = 1.0 / mean_interval
    return 60.0 * steps_per_sec
