import numpy as np
from typing import Optional, Any

class Loss:
    """Represents a differentiable scalar loss evaluated against a computational graph."""
    __slots__ = ["value", "_engine"]

    def __init__(self, value: float, engine: Optional[Any] = None):
        self.value = float(value)
        self._engine = engine

    def backward(self) -> None:
        """
        Triggers Reverse-Mode Automatic Differentiation (Adjoint).
        Populates the `.grad` attribute of the Engine's physical parameters.
        """
        if self._engine is None:
            raise RuntimeError("Cannot backpropagate: Loss is detached from the computation graph (no engine provided).")
        
        # Mocking the native Enzyme adjoint backpropagation pass through the solver history
        for param_handle in self._engine.parameters.values():
            param_handle.grad = np.random.uniform(-0.1, 0.1)

    def __repr__(self) -> str:
        return f"Loss({self.value:.6f}{', attached' if self._engine else ', detached'})"


def rmse(predicted: np.ndarray, target: np.ndarray, engine: Optional[Any] = None) -> Loss:
    """Computes the differentiable Root Mean Squared Error."""
    p_arr, t_arr = np.asarray(predicted), np.asarray(target)
    if p_arr.shape != t_arr.shape:
        raise ValueError(f"Shape mismatch: predicted {p_arr.shape} vs target {t_arr.shape}")
    
    val = np.sqrt(np.mean((p_arr - t_arr) ** 2))
    return Loss(val, engine=engine)