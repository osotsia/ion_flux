import numpy as np
import scipy.linalg
from typing import Dict, Any, Optional

try:
    from ion_flux._core import SolverHandle
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

class Session:
    """
    Stateful execution session for Hardware/Software-in-the-Loop co-simulation.
    Keeps native memory, sparse matrices, and integration history "hot" in hardware.
    """
    def __init__(self, engine: Any, parameters: Dict[str, float], soc: Optional[float] = None):
        self.engine = engine
        self.parameters = {**{k: v.value for k, v in engine.parameters.items()}, **parameters}
        self.time = 0.0
        self._history = {"Time [s]": [0.0]}
        
        y0, ydot0, id_arr, precond_diag = engine._extract_metadata()
        p_list = engine._pack_parameters(self.parameters)
        m_list = engine.layout.pack_mesh_data()
        self.id_arr = np.array(id_arr)
        
        if RUST_FFI_AVAILABLE and not engine.mock_execution:
            self.handle = SolverHandle(
                engine.runtime.lib_path, engine.layout.n_states, engine.jacobian_bandwidth,
                y0, ydot0, id_arr, p_list, m_list, precond_diag
            )
        else:
            self.handle = None
            self._mock_y = np.array(y0)

    def set_parameter(self, param_name: str, value: float) -> None:
        # BUG 15 FIX: Session overrides its own local dict, NOT the global Engine parameters
        self.parameters[param_name] = value 
        if self.handle:
            offset = self.engine.layout.get_param_offset(param_name)
            self.handle.set_parameter(offset, value)

    def get_array(self, variable_name: str) -> np.ndarray:
        """Returns the full underlying spatial array for localized condition triggers."""
        y = self.handle.get_state() if self.handle else self._mock_y
        if variable_name in self.engine.layout.state_offsets:
            offset, size = self.engine.layout.state_offsets[variable_name]
            return y[offset:offset+size]
        if variable_name == "Voltage" and self.handle is None:
            return np.array([max(2.5, 4.2 - (self.time * 0.0001))])
        raise KeyError(f"Variable '{variable_name}' not found in the current state.")

    def get(self, variable_name: str) -> float:
        """Provides backwards-compatible scalar averaging for telemetry and legacy logging."""
        return float(np.mean(self.get_array(variable_name)))

    def step(self, dt: float, inputs: Optional[Dict[str, float]] = None) -> None:
        if inputs:
            for k, v in inputs.items(): self.set_parameter(k, v)
        if self.handle:
            self.handle.step(dt)
        self.time += dt
        self._history["Time [s]"].append(self.time)

    def checkpoint(self) -> None:
        if self.handle:
            self._ckpt = self.handle.clone_state()
        else:
            self._ckpt = (self.time, self._mock_y.copy())
            
    def restore(self) -> None:
        if not hasattr(self, "_ckpt"): return
        if self.handle:
            self.handle.restore_state(*self._ckpt)
            self.time = self._ckpt[0]
        else:
            self.time = self._ckpt[0]
            self._mock_y = self._ckpt[1].copy()

    def triggered(self, condition: Any) -> bool:
        if condition is None or isinstance(condition, (int, float)): return False
        if isinstance(condition, str):
            from ion_flux.dsl.core import Condition
            condition = Condition(condition)
        elif hasattr(condition, "expression"):
            from ion_flux.dsl.core import Condition
            condition = Condition(condition.expression)
        return condition.evaluate(self)

    def reach_steady_state(self) -> None:
        # Level 10: Explicit Newton Root-Finding ensures proper evaluation zero-point for EIS analysis
        if self.handle: 
            self.handle.reach_steady_state()
            self.time += 1000.0
        else:
            self.time += 1000.0

    def solve_eis(self, frequencies: np.ndarray, input_var: str, output_var: str) -> Any:
        if not self.handle:
            w = np.asarray(frequencies) * 2 * np.pi
            Z = 0.05 + (0.1 / (1 + 1j * w * 0.1)) + (0.01 / np.sqrt(1j * w))
            return Z
            
        N = self.engine.layout.n_states
        J_steady = self.handle.get_jacobian(0.0)
        
        p_val = self.parameters.get(input_var, 0.0)
        eps = 1e-6
        y = self.handle.get_state()
        ydot = np.zeros_like(y)
        
        res_base = np.array(self.engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters=self.parameters))
        p_pert = self.parameters.copy()
        p_pert[input_var] = p_val + eps
        res_pert = np.array(self.engine.evaluate_residual(y.tolist(), ydot.tolist(), parameters=p_pert))
        B = -(res_pert - res_base) / eps
        
        out_offset = self.engine.layout.get_state_offset(output_var)
        C = np.zeros(N)
        C[out_offset] = 1.0 
        
        w_arr = np.asarray(frequencies) * 2 * np.pi
        Z = np.zeros_like(w_arr, dtype=np.complex128)
        M = np.diag(self.id_arr)
        
        for i, w in enumerate(w_arr):
            A = 1j * w * M + J_steady
            try:
                X = scipy.linalg.solve(A, B)
                Z[i] = np.dot(C, X)
            except scipy.linalg.LinAlgError:
                Z[i] = np.nan
                
        # Bug 4 Fix: Return a compliant SimulationResult for downstream metrics.py hooks
        data = {"Frequencies [Hz]": w_arr / (2 * np.pi), "Z_real": np.real(Z), "Z_imag": np.imag(Z)}
        trajectory = {
            "requires_grad": list(self.parameters.keys()),
            "_is_eis": True, 
            "Z": Z, 
            "w_arr": w_arr, 
            "J_steady": J_steady, 
            "B": B, 
            "C": C,
            "input_var": input_var
        }
        
        from ion_flux.runtime.engine import SimulationResult
        return SimulationResult(data, self.parameters, engine=self.engine, trajectory=trajectory)
