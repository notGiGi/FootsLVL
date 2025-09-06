import struct
from typing import Tuple, List, Dict, Any

def decode_packet(data: bytes, n_sensors: int = 24) -> Tuple[List[float], Dict[str, Any]]:
    # Placeholder: interpreta todo como int16 consecutivos sin cabecera.
    count = len(data) // 2
    fmt = "<" + "h"*count
    vals = list(struct.unpack(fmt, data[:count*2]))
    # clamp a 0+
    vals = [v if v > 0 else 0 for v in vals[:n_sensors]]
    meta = {"seq": None}
    return vals, meta
