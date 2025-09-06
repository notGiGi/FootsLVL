import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

class RollingGRF(QWidget):
    def __init__(self, title="GRF (Left/Right)", maxlen=1000):
        super().__init__()
        self.maxlen = maxlen
        self.ts = np.zeros(maxlen)
        self.grfL = np.zeros(maxlen)
        self.grfR = np.zeros(maxlen)
        self.idx = 0

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(title))
        self.plot = pg.PlotWidget()
        self.curveL = self.plot.plot(pen=pg.mkPen(width=2))
        self.curveR = self.plot.plot(pen=pg.mkPen(width=2))
        layout.addWidget(self.plot)

    def push(self, t_ms: int, gL: float, gR: float):
        i = self.idx % self.maxlen
        self.ts[i] = t_ms / 1000.0
        self.grfL[i] = gL
        self.grfR[i] = gR
        self.idx += 1
        sl = slice(0, min(self.idx, self.maxlen))
        self.curveL.setData(self.ts[sl], self.grfL[sl])
        self.curveR.setData(self.ts[sl], self.grfR[sl])
