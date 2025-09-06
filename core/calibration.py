import numpy as np
from dataclasses import dataclass

@dataclass
class CalibParams:
    zero_L: np.ndarray
    zero_R: np.ndarray
    scale_L: float = 1.0
    scale_R: float = 1.0

def init_calibration(n_sensors: int):
    return CalibParams(zero_L=np.zeros(n_sensors), zero_R=np.zeros(n_sensors))

def apply_calibration(pressures: np.ndarray, zero: np.ndarray):
    p = pressures - zero
    p[p < 0] = 0
    return p
