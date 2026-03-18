import numpy as np

class ConstantCurrent:
    """Represents a simple constant current protocol with a cutoff."""
    def __init__(self, c_rate: float, until_voltage: float):
        self.c_rate = c_rate
        self.until_voltage = until_voltage

class CurrentProfile:
    """Represents an arbitrary timeseries protocol (e.g., a drive cycle)."""
    def __init__(self, time: np.ndarray, current: np.ndarray):
        self.time = time
        self.current = current