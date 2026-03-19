import numpy as np

class ConstantCurrent:
    """Represents a simple constant current protocol with a cutoff."""
    def __init__(self, c_rate: float, until_voltage: float):
        self.c_rate = float(c_rate)
        self.until_voltage = float(until_voltage)


class CurrentProfile:
    """
    Represents an arbitrary timeseries protocol (e.g., a drive cycle).
    Enforces C-contiguous, typed memory layout for safe FFI traversal.
    """
    def __init__(self, time: np.ndarray, current: np.ndarray):
        # Enforce strict float64 C-contiguous arrays for the Rust FFI
        self.time = np.ascontiguousarray(time, dtype=np.float64)
        self.current = np.ascontiguousarray(current, dtype=np.float64)
        
        if self.time.ndim != 1 or self.current.ndim != 1:
            raise ValueError("Time and current profiles must be 1-dimensional arrays.")
            
        if self.time.shape != self.current.shape:
            raise ValueError("Time and current profiles must have identical lengths.")
