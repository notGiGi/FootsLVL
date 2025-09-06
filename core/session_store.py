import os, csv, time
from typing import Optional, IO

class SessionWriter:
    def __init__(self, base_dir="sessions"):
        os.makedirs(base_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(base_dir, f"session_{ts}.csv")
        self.f: Optional[IO] = None
        self.w: Optional[csv.writer] = None

    def open(self, n_sensors: int):
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        header = ["t_ms","grf_L","copx_L","copy_L","grf_R","copx_R","copy_R"]
        header += [f"L{i}" for i in range(n_sensors)] + [f"R{i}" for i in range(n_sensors)]
        self.w.writerow(header)

    def write(self, sample, grf_L, copL, grf_R, copR):
        if not self.w: return
        row = [
            sample["t_ms"],
            f"{grf_L:.3f}", f"{copL[0]:.4f}", f"{copL[1]:.4f}",
            f"{grf_R:.3f}", f"{copR[0]:.4f}", f"{copR[1]:.4f}",
        ]
        row += [f"{v:.3f}" for v in sample["left"]] + [f"{v:.3f}" for v in sample["right"]]
        self.w.writerow(row)

    def close(self):
        if self.f:
            self.f.close()
            self.f = None
            self.w = None
