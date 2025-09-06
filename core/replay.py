import csv, threading, time
from typing import Callable, Dict, Any

Sample = Dict[str, Any]

class ReplaySource:
    def __init__(self, path: str, n_sensors: int):
        self.path = path
        self.n = n_sensors
        self._running = False
        self._thread = None

    def start(self, on_sample: Callable[[Sample], None]):
        self._running = True
        self._thread = threading.Thread(target=self._run, args=(on_sample,), daemon=True)
        self._thread.start()

    def _run(self, cb):
        with open(self.path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            prev_t = None
            for row in r:
                if not self._running:
                    break
                t_ms = int(float(row["t_ms"]))
                if prev_t is None:
                    prev_t = t_ms
                dt = (t_ms - prev_t) / 1000.0
                if dt > 0: time.sleep(min(dt, 0.2))  # cap para no “congelar” si hay gaps largos
                prev_t = t_ms

                left = [float(row[f"L{i}"]) for i in range(self.n)]
                right = [float(row[f"R{i}"]) for i in range(self.n)]
                cb({"t_ms": t_ms, "left": left, "right": right})

    def stop(self):
        self._running = False
