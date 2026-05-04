import numpy as np
from typing import Dict, Any, List, Optional

class Variable:
    """Wrapper mapping flat FFI arrays back into intuitive multidimensional structures."""
    __slots__ = ["data", "result", "name"]
    def __init__(self, data: np.ndarray, result: Optional[Any] = None, name: str = ""): 
        self.data = data
        self.result = result
        self.name = name
    def __repr__(self) -> str: 
        return f"<Variable: {self.name} shape={self.data.shape}>"

class SimulationResult:
    """Data object returning exact multi-dimensional trajectories outputted by the Engine."""
    __slots__ = ["_data", "parameters", "status", "engine", "trajectory"]
    
    def __init__(self, data: Dict[str, np.ndarray], parameters: Dict[str, float], status: str = "completed", engine: Optional[Any] = None, trajectory: Optional[Dict] = None):
        self._data = data
        self.parameters = parameters
        self.status = status
        self.engine = engine
        self.trajectory = trajectory

    def __getitem__(self, key: str) -> Variable:
        if key not in self._data: raise KeyError(f"Variable '{key}' not found.")
        return Variable(self._data[key], result=self, name=key)
        
    def to_dict(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Provides native JSON-serialization compatibility for Cloud workflows."""
        keys = variables or self._data.keys()
        return {k: self._data[k].tolist() for k in keys if k in self._data}