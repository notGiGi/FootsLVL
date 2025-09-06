# Driver BLE a completar con UUID reales.
import asyncio, time
from typing import Callable, Dict, Any
from bleak import BleakClient, BleakScanner

Sample = Dict[str, Any]

PRESSURE_SERVICE = "0000xxxx-0000-1000-8000-00805f9b34fb"  # TODO
PRESSURE_CHAR    = "0000yyyy-0000-1000-8000-00805f9b34fb"  # TODO

class BleSource:
    def __init__(self, name_hint="nurvv", n_sensors=24):
        self.name_hint = name_hint.lower()
        self.n = n_sensors
        self._on_sample: Callable[[Sample], None] = lambda s: None
        self._running = False
        self._task = None

    def start(self, on_sample: Callable[[Sample], None]):
        self._on_sample = on_sample
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        dev = await self._find_device()
        if not dev:
            print("BLE device not found")
            return
        async with BleakClient(dev) as client:
            await client.start_notify(PRESSURE_CHAR, self._handle)
            # Mantener vivo hasta stop()
            while self._running:
                await asyncio.sleep(0.05)
            try:
                await client.stop_notify(PRESSURE_CHAR)
            except:
                pass

    async def _find_device(self):
        for _ in range(4):
            devices = await BleakScanner.discover()
            for d in devices:
                if (d.name or "").lower().find(self.name_hint) >= 0:
                    return d
        return None

    def _handle(self, handle, data: bytes):
        from .decoder import decode_packet
        # Nota: si cada plantilla (L/R) emite por separado, lanzar dos BleSource (o unir luego).
        press, _ = decode_packet(data, n_sensors=self.n)
        sample = {"t_ms": int(time.time()*1000), "left": press, "right": [0.0]*self.n}
        self._on_sample(sample)

    def stop(self):
        self._running = False
