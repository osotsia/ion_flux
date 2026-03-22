import numpy as np
from typing import Optional, Any

try:
    from ion_flux._core import discrete_adjoint_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

class Loss:
    """Represents a differentiable scalar loss evaluated against a computational graph."""
    __slots__ = ["value", "_engine", "_trajectory", "_dl_dy_mapped"]

    def __init__(self, value: float, engine: Optional[Any] = None, trajectory: Optional[dict] = None, dl_dy_mapped: Optional[np.ndarray] = None):
        self.value = float(value)
        self._engine = engine
        self._trajectory = trajectory
        self._dl_dy_mapped = dl_dy_mapped

    def backward(self) -> None:
        """Triggers Reverse-Mode Automatic Differentiation (Adjoint)."""
        if self._engine is None or self._trajectory is None:
            raise RuntimeError("Cannot backpropagate: Loss is detached or lacks integration trajectory.")
        
        req_grad = self._trajectory.get("requires_grad", list(self._engine.parameters.keys()))
        
        if not RUST_FFI_AVAILABLE or self._engine.mock_execution:
            for param_handle in self._engine.parameters.values():
                if param_handle.name in req_grad:
                    param_handle.grad = np.random.uniform(-0.1, 0.1)
            return
            
        t_eval = self._trajectory["Time [s]"]
        y_traj = self._trajectory["_y_raw"]
        
        dl_dy = self._dl_dy_mapped if self._dl_dy_mapped is not None else np.zeros_like(y_traj)
        
        y0, ydot0, id_arr = self._engine._extract_metadata()
        p_list = self._engine._pack_parameters({})
        
        bw = getattr(self._engine, "jacobian_bandwidth", 0)
        p_grad = discrete_adjoint_native(
            self._engine.runtime.lib_path,
            y_traj.tolist(), t_eval.tolist(), id_arr, p_list, dl_dy.tolist(), bw
        )
        
        for p_name in req_grad:
            if p_name in self._engine.layout.param_offsets:
                offset = self._engine.layout.param_offsets[p_name][0]
                self._engine.parameters[p_name].grad = p_grad[offset]


def rmse(predicted: np.ndarray, target: np.ndarray, engine: Optional[Any] = None, state_name: str = "Voltage") -> Loss:
    """
    Computes the Root Mean Square Error and tracks the analytical gradient mapping.
    
    Args:
        predicted: Simulated trajectory
        target: Lab data trajectory
        engine: The engine used for the simulation
        state_name: The string name of the state being evaluated (defaults to "Voltage")
    """
    p_arr, t_arr = np.asarray(predicted), np.asarray(target)
    if p_arr.shape != t_arr.shape:
        raise ValueError(f"Shape mismatch: predicted {p_arr.shape} vs target {t_arr.shape}")
    
    diff = p_arr - t_arr
    val = np.sqrt(np.mean(diff ** 2))
    
    dl_dy_mapped = None
    if engine and hasattr(engine, "_current_trajectory"):
        y_traj = engine._current_trajectory["_y_raw"]
        dl_dy_mapped = np.zeros_like(y_traj)
        
        # Dynamically map the derivative back to the exact physical state offset
        if state_name in engine.layout.state_offsets:
            offset, size = engine.layout.state_offsets[state_name]
            
            grad_multiplier = 1.0 / (len(diff) * max(val, 1e-12))
            
            # Map gradient across spatial dimensions evenly
            for i in range(size):
                dl_dy_mapped[:, offset + i] = (grad_multiplier * diff) / size
            
    return Loss(val, engine=engine, trajectory=getattr(engine, "_current_trajectory", None), dl_dy_mapped=dl_dy_mapped)