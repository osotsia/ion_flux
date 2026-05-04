import numpy as np
import scipy.linalg
from typing import Dict, Any, Optional

from ion_flux.runtime.eis import solve_eis
from ion_flux.runtime._2_initializers import evaluate_ic
from ion_flux.runtime._4_diagnostics import format_native_crash

try:
    from ion_flux._core import SolverHandle, SundialsHandle
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

class Session:
    """
    Stateful execution session for Hardware/Software-in-the-Loop co-simulation.
    Keeps native memory, sparse matrices, and integration history "hot" in hardware.
    """
    def __init__(self, engine: Any, parameters: Dict[str, float], soc: Optional[float] = None, debug: bool = False):
        self.engine = engine
        self.manifest = engine.manifest
        
        self.parameters = {**{k: v.value for k, v in engine.parameters.items()}, **parameters}
        self.time = 0.0
        self._history = {"Time [s]": [0.0]}
        self.debug = debug
        
        y0, ydot0 = evaluate_ic(self.manifest, self.parameters)
        p_list = self.manifest.pack_parameters(self.parameters)
        m_list = self.manifest.layout.get_mesh_data()
        
        self.id_arr = np.array(self.manifest.id_arr)
        constraints = [0.0] * self.manifest.layout.n_states
        
        if RUST_FFI_AVAILABLE and not engine.mock_execution:
            c_seeds, c_ptrs, c_rows, c_cols, c_dense = self.manifest.cpr_cache
            if getattr(engine, "solver_backend", "native") == "sundials":
                self.handle = SundialsHandle(
                    self.manifest.lib_path, self.manifest.layout.n_states,
                    y0, ydot0, self.manifest.id_arr, p_list, m_list, self.manifest.layout.n_obs,
                    c_seeds, c_ptrs, c_rows, c_cols, c_dense
                )
            else:
                self.handle = SolverHandle(
                    self.manifest.lib_path, self.manifest.layout.n_states, self.manifest.jacobian_bandwidth,
                    y0, ydot0, self.manifest.id_arr, constraints, p_list, m_list, self.manifest.spatial_diag, 
                    self.manifest.max_steps, self.manifest.layout.n_obs, self.debug,
                    c_seeds, c_ptrs, c_rows, c_cols, c_dense
                )
                try:
                    self.handle.calc_algebraic_roots()
                except RuntimeError as e:
                    raise format_native_crash(e, self.manifest) from None
        else:
            self.handle = None
            self._mock_y = np.array(y0)

    def set_parameter(self, param_name: str, value: float) -> None:
        self.parameters[param_name] = value 
        if self.handle:
            offset = self.manifest.layout.get_param_offset(param_name)
            self.handle.set_parameter(offset, value)

    def get_array(self, variable_name: str) -> np.ndarray:
        if variable_name in self.manifest.layout.state_offsets:
            y = self.handle.get_state() if self.handle else self._mock_y
            offset, size = self.manifest.layout.state_offsets[variable_name]
            return y[offset:offset+size]
        if variable_name in self.manifest.layout.obs_offsets:
            obs = self.handle.get_observables_py() if self.handle else np.zeros(self.manifest.layout.n_obs)
            offset, size = self.manifest.layout.obs_offsets[variable_name]
            return obs[offset:offset+size]
        if variable_name == "Voltage" and self.handle is None:
            return np.array([max(2.5, 4.2 - (self.time * 0.0001))])
        raise KeyError(f"Variable '{variable_name}' not found in the current state or observables.")

    def get(self, variable_name: str) -> float:
        return float(np.mean(self.get_array(variable_name)))

    def step(self, dt: float, inputs: Optional[Dict[str, float]] = None) -> None:
        if inputs:
            changed = False
            for k, v in inputs.items(): 
                current_v = self.parameters.get(k)
                if current_v is None or abs(current_v - v) > 1e-12:
                    self.set_parameter(k, v)
                    changed = True
                    
            if changed and self.handle:
                try:
                    self.handle.calc_algebraic_roots()
                except RuntimeError as e:
                    raise format_native_crash(e, self.manifest) from None
                
        if self.handle:
            if getattr(self, "record_history", False):
                if hasattr(self.handle, "step_history"):
                    try:
                        mt, my, mydot = self.handle.step_history(dt)
                    except RuntimeError as e:
                        raise format_native_crash(e, self.manifest) from None
                        
                    if len(mt) > 0:
                        self.micro_t.extend(mt.tolist())
                        self.micro_y.extend(my.tolist())
                        self.micro_ydot.extend(mydot.tolist())
                        p_list = self.manifest.pack_parameters(self.parameters)
                        self.micro_p.extend([p_list] * len(mt))
                else:
                    try:
                        self.handle.step(dt) 
                    except RuntimeError as e:
                        raise format_native_crash(e, self.manifest) from None
            else:
                try:
                    self.handle.step(dt)
                except RuntimeError as e:
                    raise format_native_crash(e, self.manifest) from None
                    
        self.time += dt
        self._history["Time [s]"].append(self.time)

    def checkpoint(self) -> None:
        if self.handle and hasattr(self.handle, "clone_state"):
            self._ckpt = self.handle.clone_state()
        else:
            self._ckpt = (self.time, self._mock_y.copy() if not self.handle else self.handle.get_state())
            
    def restore(self) -> None:
        if not hasattr(self, "_ckpt"): return
        if self.handle and hasattr(self.handle, "restore_state"):
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
        if self.handle: 
            try:
                self.handle.reach_steady_state()
            except RuntimeError as e:
                raise format_native_crash(e, self.manifest) from None
            self.time += 1000.0
        else:
            self.time += 1000.0

    def solve_eis(self, frequencies: np.ndarray, input_var: str, output_var: str) -> Any:
        return solve_eis(self, frequencies, input_var, output_var)