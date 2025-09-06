from typing import Protocol, Callable, Dict, Any

Sample = Dict[str, Any]  # {"t_ms": int, "left": list[float], "right": list[float]}

class DataSource(Protocol):
    def start(self, on_sample: Callable[[Sample], None]) -> None: ...
    def stop(self) -> None: ...
