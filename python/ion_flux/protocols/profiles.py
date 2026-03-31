import numpy as np
from typing import List, Any
from ion_flux.dsl.core import Condition

class ConstantCurrent:
    __slots__ = ["c_rate", "until_voltage"]
    def __init__(self, c_rate: float, until_voltage: float):
        self.c_rate = float(c_rate)
        self.until_voltage = float(until_voltage)


class CurrentProfile:
    """Enforces strictly typed, contiguous memory buffers for C-ABI safety."""
    __slots__ = ["time", "current"]
    def __init__(self, time: np.ndarray, current: np.ndarray):
        self.time = np.ascontiguousarray(time, dtype=np.float64)
        self.current = np.ascontiguousarray(current, dtype=np.float64)
        
        if self.time.ndim != 1 or self.current.ndim != 1:
            raise ValueError("Time and current profiles must be 1-dimensional arrays.")
        if self.time.shape != self.current.shape:
            raise ValueError("Time and current profiles must have identical lengths.")

# --- Multi-Mode Protocol Support ---

class ProtocolStep:
    """Base class for compiled state-machine execution steps."""
    pass

class CC(ProtocolStep):
    __slots__ = ["rate", "until", "time"]
    def __init__(self, rate: float, until: Any = None, time: float = float('inf')):
        self.rate = float(rate)
        self.until = Condition(until) if until is not None else None
        self.time = float(time)

class CV(ProtocolStep):
    __slots__ = ["voltage", "until", "time"]
    def __init__(self, voltage: float, until: Any = None, time: float = float('inf')):
        self.voltage = float(voltage)
        self.until = Condition(until) if until is not None else None
        self.time = float(time)

class Rest(ProtocolStep):
    __slots__ = ["time", "until"]
    def __init__(self, time: float):
        self.time = float(time)
        self.until = None

class Sequence:
    """Declarative state machine for hot-swapping constraints during a single solve."""
    __slots__ = ["steps"]
    def __init__(self, steps: List[ProtocolStep]):
        self.steps = steps