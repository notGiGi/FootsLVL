# core/simulator.py
import time, threading
import numpy as np
from .foot_shape import foot_outline_points, polygon_mask, cross_section

def _gauss2(pos, mu, sigma):
    d = pos - mu[None, :]
    r2 = (d**2).sum(axis=1)
    return np.exp(-0.5 * r2 / (sigma**2))

def _anchors_from_outline(left=True, grid_w=200, grid_h=320):
    poly01 = foot_outline_points(left=left, samples_per_seg=60)
    mask, _ = polygon_mask(poly01, grid_w, grid_h)

    # util: punto medio entre bordes a cierta altura y posiciones a lo ancho
    def x_minmax(y01):
        yy = int(np.clip(round(y01*grid_h), 0, grid_h-1))
        xs = cross_section(mask, yy)
        if xs is None:
            # busca cerca
            for off in range(1,8):
                for y2 in (yy-off, yy+off):
                    if 0 <= y2 < grid_h:
                        xs = cross_section(mask, y2)
                        if xs is not None:
                            yy = y2; break
                if xs is not None: break
        if xs is None: xs = (int(0.35*grid_w), int(0.85*grid_w))
        return xs[0]/grid_w, xs[1]/grid_w, (yy+0.5)/grid_h

    # alturas anatómicas aproximadas
    x0,x1,yh = x_minmax(0.10)   # talón
    xm0,xm1,ym = x_minmax(0.45) # mediopié
    xf0,xf1,yf = x_minmax(0.80) # metatarsos
    xt0,xt1,yt = x_minmax(0.93) # dedos

    # anclas
    heel_lat = np.array([x1-0.03, yh])
    heel_med = np.array([x0+0.03, yh])
    mid_lat  = np.array([xm1-0.02, ym])
    mid_med  = np.array([xm0+0.03, ym])
    # MTH1..5 distribuidos de medial→lateral en y≈yf
    mths = []
    for r in np.linspace(0.12, 0.88, 5):
        mths.append(np.array([xf0*(1-r) + xf1*r, yf]))
    mth1,mth2,mth3,mth4,mth5 = mths
    hallux  = np.array([xt0+0.04*(xt1-xt0), yt])

    return {
        "heel_lat": heel_lat, "heel_med": heel_med,
        "mid_lat": mid_lat,   "mid_med":  mid_med,
        "mth1": mth1, "mth2": mth2, "mth3": mth3, "mth4": mth4, "mth5": mth5,
        "hallux": hallux
    }

class SimulatorSource:
    """
    Simulador clínico: patrón talón→MTH2-MTH3→hallux, anclado a la silueta.
    """
    def __init__(self, n_sensors=24, freq=120, cadence_hz=1.8, stance_ratio=0.62,
                 amp=160.0, noise=0.4, **kwargs):
        if "base_amp" in kwargs and kwargs["base_amp"] is not None:
            amp = float(kwargs["base_amp"])
        self.n = n_sensors
        self.freq = int(freq)
        self.cad = float(cadence_hz)
        self.stance_ratio = float(stance_ratio)
        self.amp = float(amp)
        self.noise = float(noise)
        self._running = False
        self._thread = None

        # Layout sintético (coincide con silueta)
        from .interpolation import foot_layout_24
        self.coordsL = foot_layout_24(left=True)
        self.coordsR = foot_layout_24(left=False)

        self.A_L = _anchors_from_outline(left=True)
        self.A_R = _anchors_from_outline(left=False)

    def start(self, on_sample):
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(on_sample,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self, cb):
        t0 = time.time()
        while self._running:
            t = time.time() - t0
            L = self._pressures(t, True)
            R = self._pressures(t + 0.5/self.cad, False)
            cb({"t_ms": int(t*1000), "left": L, "right": R})
            time.sleep(max(0.0, 1.0/self.freq))

    def _pressures(self, t, left=True):
        phi_stride = (t * self.cad) % 1.0
        if phi_stride >= self.stance_ratio:
            return (np.zeros(self.n) + np.random.normal(0.0, 0.02, self.n)).clip(min=0).tolist()
        phi = phi_stride / self.stance_ratio

        if left:
            A = self.A_L; coords = self.coordsL
        else:
            A = self.A_R; coords = self.coordsR

        # pesos por subfase (claros y visuales)
        if   phi < 0.15: w = {"heel_lat":1.0,"heel_med":0.6,"mid_lat":0.1,"mid_med":0.1,"mth2":0.0,"mth3":0.0,"mth1":0.0,"mth4":0.0,"mth5":0.0,"hallux":0.0}
        elif phi < 0.40: w = {"heel_lat":0.5,"heel_med":0.4,"mid_lat":0.6,"mid_med":0.4,"mth2":0.1,"mth3":0.1,"mth1":0.0,"mth4":0.0,"mth5":0.0,"hallux":0.0}
        elif phi < 0.70: w = {"heel_lat":0.2,"heel_med":0.2,"mid_lat":0.4,"mid_med":0.3,"mth2":0.8,"mth3":1.0,"mth1":0.3,"mth4":0.5,"mth5":0.3,"hallux":0.2}
        else:            w = {"heel_lat":0.0,"heel_med":0.1,"mid_lat":0.2,"mid_med":0.2,"mth2":0.6,"mth3":0.7,"mth1":0.7,"mth4":0.2,"mth5":0.1,"hallux":1.0}

        p = np.zeros(self.n, float)
        # spreads adaptados a la separación de sensores
        p += w["heel_lat"] * _gauss2(coords, A["heel_lat"], 0.10)
        p += w["heel_med"] * _gauss2(coords, A["heel_med"], 0.10)
        p += w["mid_lat"]  * _gauss2(coords, A["mid_lat"],  0.11)
        p += w["mid_med"]  * _gauss2(coords, A["mid_med"],  0.11)
        p += w["mth1"]     * _gauss2(coords, A["mth1"],     0.09)
        p += w["mth2"]     * _gauss2(coords, A["mth2"],     0.09)
        p += w["mth3"]     * _gauss2(coords, A["mth3"],     0.09)
        p += w["mth4"]     * _gauss2(coords, A["mth4"],     0.09)
        p += w["mth5"]     * _gauss2(coords, A["mth5"],     0.09)
        p += w["hallux"]   * _gauss2(coords, A["hallux"],   0.08)

        p *= self.amp
        if self.noise>0: p += np.random.normal(0.0, self.noise, self.n)
        p[p<0] = 0.0
        return p.tolist()
