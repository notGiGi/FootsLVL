# ui/heatmap_view.py
import os
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
from PySide6.QtCore import QRectF
from core.interpolation import foot_layout_24, precompute_mapping, interpolate_to_grid

# ========== utils ==========
def _conv2d_box3(a: np.ndarray):
    if a.ndim != 2: return a
    pad = np.pad(a, 1, mode='reflect')
    return (
        pad[:-2,:-2] + pad[:-2,1:-1] + pad[:-2,2:] +
        pad[1:-1,:-2] + pad[1:-1,1:-1] + pad[1:-1,2:] +
        pad[2:,:-2] + pad[2:,1:-1] + pad[2:,2:]
    ) / 9.0

def _catmull_rom_closed(ctrl, samples_per_seg=60):
    P = np.asarray(ctrl, float); n = len(P); out = []
    for i in range(n):
        p0 = P[(i-1)%n]; p1 = P[i%n]; p2 = P[(i+1)%n]; p3 = P[(i+2)%n]
        for s in np.linspace(0,1,samples_per_seg,endpoint=False):
            s2=s*s; s3=s2*s
            b0=-0.5*s3+1.0*s2-0.5*s
            b1= 1.5*s3-2.5*s2+1.0
            b2=-1.5*s3+2.0*s2+0.5*s
            b3= 0.5*s3-0.5*s2
            out.append(b0*p0+b1*p1+b2*p2+b3*p3)
    out.append(out[0]); return np.asarray(out,float)

def _anchors_left_template():
    # x=0 medial, x=1 lateral ; y=0 dedos (arriba), y=1 talón (abajo)
    return np.array([
        [0.58,0.10],[0.66,0.12],[0.74,0.16],   # dedos lat
        [0.82,0.30],[0.88,0.48],[0.88,0.66],   # lado lateral
        [0.82,0.82],[0.70,0.93],[0.56,0.98],   # talón
        [0.44,0.94],[0.36,0.82],[0.32,0.66],   # arco
        [0.32,0.50],[0.34,0.36],[0.38,0.24],   # hacia antepié medial
        [0.42,0.16],[0.50,0.11]                # zona hallux/2º-3º
    ], dtype=float)

def _foot_outline_points(left=True, samples_per_seg=60):
    ctrl = _anchors_left_template()
    if not left:
        ctrl = ctrl.copy(); ctrl[:,0] = 1.0 - ctrl[:,0]
    return _catmull_rom_closed(ctrl, samples_per_seg=samples_per_seg)  # (M,2) en [0..1], y=0 arriba

def _polygon_mask(poly01, grid_w, grid_h):
    poly = poly01.copy()
    poly[:,0] *= grid_w; poly[:,1] *= grid_h
    H,W = grid_h, grid_w
    mask = np.zeros((H,W), bool)
    x = poly[:,0]; y = poly[:,1]; n=len(poly)
    for j in range(n):
        x0,y0 = x[j],y[j]; x1,y1 = x[(j+1)%n],y[(j+1)%n]
        if y0 == y1: continue
        y_min = int(max(0, np.floor(min(y0,y1))))
        y_max = int(min(H-1, np.ceil(max(y0,y1))))
        for yy in range(y_min, y_max+1):
            t = (yy+0.5 - y0) / (y1 - y0)
            if 0.0 <= t < 1.0:
                xx = x0 + t*(x1-x0)
                col = int(np.clip(np.floor(xx), 0, W-1))
                mask[yy, col:] ^= True
    return mask, poly

def _mask_from_png(path, grid_w, grid_h, flip_lr=False):
    try:
        from PIL import Image
    except Exception:
        return None, None
    if not os.path.exists(path):
        return None, None
    im = Image.open(path).convert("L").resize((grid_w, grid_h), Image.BILINEAR)
    arr = np.array(im, dtype=np.uint8)
    if flip_lr:
        arr = np.fliplr(arr)
    # blanco = pie; negro = fondo → umbral 128
    mask = arr >= 128
    # contorno: borde del mask
    er = _erode(mask, 1)
    border = mask & (~er)
    ys, xs = np.where(border)
    if ys.size == 0:
        poly = None
    else:
        poly = np.stack([xs, ys], axis=1).astype(float)
    return mask, poly

def _erode(m: np.ndarray, iters=1):
    m = m.astype(bool)
    for _ in range(iters):
        pad = np.pad(m, 1, 'constant')
        m = (
            pad[:-2,:-2]&pad[:-2,1:-1]&pad[:-2,2:]&
            pad[1:-1,:-2]&pad[1:-1,1:-1]&pad[1:-1,2:]&
            pad[2:,:-2]&pad[2:,1:-1]&pad[2:,2:]
        )
    return m

# ========== Canvas de un pie ==========
class _HeatmapCanvas(QWidget):
    def __init__(self, title="Left", grid_w=64, grid_h=96, n_sensors=24, is_left=True, parent=None):
        super().__init__(parent)
        self.grid_w, self.grid_h = grid_w, grid_h
        self.n = n_sensors
        self.is_left = is_left

        # 1) Máscara desde PNG si existe, si no usar plantilla curva interna
        assets = os.path.join(os.getcwd(), "assets")
        png_name = "foot_left.png" if is_left else "foot_right.png"
        png_path = os.path.join(assets, png_name)
        mask_png, poly_png = _mask_from_png(png_path, grid_w, grid_h, flip_lr=False)
        if mask_png is not None:
            self.mask = mask_png
            self.poly_px = poly_png
        else:
            poly01 = _foot_outline_points(left=is_left, samples_per_seg=80)
            self.mask, self.poly_px = _polygon_mask(poly01, grid_w, grid_h)

        # 2) Layout sensores DENTRO de la silueta (coordenadas [0..1], y=0 arriba)
        #    Si tu main_window usa n=24, aquí se generarán 24 puntos repartidos por filas.
        self.coords01 = foot_layout_24(left=is_left)

        # 3) Mapeo IDW coherente con el lienzo
        self.idx_map, self.w_map = precompute_mapping(self.coords01, grid_w, grid_h, k=3, power=2.0)

        # Apariencia
        self._smoothing = True
        self._gain = 1.0
        self._vmin = 0.0
        self._vmax = 600.0
        self._ema_alpha = 0.15
        self._fallback = 3.0  # tinte mínimo dentro de silueta

        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
        title_lbl = QLabel(title); title_lbl.setObjectName("panelTitle"); lay.addWidget(title_lbl)

        self.glw = pg.GraphicsLayoutWidget(); self.glw.setBackground((18,20,22)); lay.addWidget(self.glw)
        self.view = self.glw.addViewBox(lockAspect=True)
        self.view.setMouseEnabled(x=False, y=False)
        self.view.invertY(True)  # y=0 arriba (dedos)

        # Heatmap
        self.image = pg.ImageItem()
        self.image.setLookupTable(pg.colormap.get("viridis").getLookupTable(nPts=256))
        self.view.addItem(self.image)

        # Contorno nítido (si PNG no tenía borde, usamos polígono de plantilla)
        if self.poly_px is not None and len(self.poly_px) >= 4:
            self.outline = pg.PlotDataItem(x=self.poly_px[:,0], y=self.poly_px[:,1],
                                           pen=pg.mkPen((240,240,245,230), width=2), connect='all')
            self.view.addItem(self.outline)

        # Colorbar
        self.cbar = pg.ColorBarItem(values=(self._vmin,self._vmax),
                                    colorMap=pg.colormap.get("viridis"),
                                    width=12, interactive=False)
        self.glw.addItem(self.cbar)

        self._set_empty()

    def _set_empty(self):
        img = np.full((self.grid_h, self.grid_w), np.nan, float)
        img[self.mask] = self._fallback
        self.image.setImage(img, autoLevels=False, levels=(self._vmin, self._vmax))
        self.image.setRect(QRectF(0,0,self.grid_w,self.grid_h))
        self.view.setRange(xRange=(0,self.grid_w), yRange=(0,self.grid_h), padding=0.02)

    # API
    def set_colormap(self, name: str):
        self.image.setLookupTable(pg.colormap.get(name).getLookupTable(nPts=256))
        self.cbar.setColorMap(pg.colormap.get(name))

    def set_intensity_scale(self, scale: float):
        self._gain = max(0.2, min(3.0, float(scale)))

    def set_smoothing(self, enabled: bool):
        self._smoothing = bool(enabled)

    def update_with_pressures(self, pressures: np.ndarray, cop_xy=None):
        if pressures is None or len(pressures) == 0: return
        grid = interpolate_to_grid(pressures, self.idx_map, self.w_map)
        if self._smoothing: grid = _conv2d_box3(grid)
        grid *= self._gain

        # Recorta 100% al pie
        out = np.full_like(grid, np.nan, float)
        out[self.mask] = np.maximum(grid[self.mask], self._fallback)

        vals = out[np.isfinite(out)]
        if vals.size > 0:
            p99 = float(np.percentile(vals, 99))
            target = max(40.0, p99)
            self._vmax = (1.0 - self._ema_alpha)*self._vmax + self._ema_alpha*target
            self.cbar.setLevels((self._vmin, self._vmax))

        self.image.setImage(out, autoLevels=False, levels=(self._vmin, self._vmax))

# ========== Panel doble ==========
class HeatmapView(QWidget):
    def __init__(self, grid_w=64, grid_h=96, n_sensors=24, title="Heatmap (Left / Right)"):
        super().__init__()
        self.setObjectName("HeatmapPanel")
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)
        head = QLabel(title); head.setObjectName("panelTitle"); lay.addWidget(head)
        row = QHBoxLayout(); row.setSpacing(10); lay.addLayout(row)

        self.left  = _HeatmapCanvas("Left Foot",  grid_w, grid_h, n_sensors, is_left=True)
        self.right = _HeatmapCanvas("Right Foot", grid_w, grid_h, n_sensors, is_left=False)

        contL = QFrame(); contL.setFrameShape(QFrame.StyledPanel)
        contR = QFrame(); contR.setFrameShape(QFrame.StyledPanel)
        gl = QGridLayout(contL); gl.setContentsMargins(6,6,6,6); gl.addWidget(self.left, 0, 0)
        gr = QGridLayout(contR); gr.setContentsMargins(6,6,6,6); gr.addWidget(self.right, 0, 0)
        row.addWidget(contL, 1); row.addWidget(contR, 1)

    def set_colormap(self, name: str):
        self.left.set_colormap(name); self.right.set_colormap(name)

    def set_intensity_scale(self, scale: float):
        self.left.set_intensity_scale(scale); self.right.set_intensity_scale(scale)

    def set_smoothing(self, enabled: bool):
        self.left.set_smoothing(enabled); self.right.set_smoothing(enabled)

    def update_with_sample(self, sample: dict, copL=None, copR=None):
        L = np.asarray(sample["left"], float); L[L<0]=0
        R = np.asarray(sample["right"], float); R[R<0]=0
        self.left.update_with_pressures(L, None)
        self.right.update_with_pressures(R, None)
