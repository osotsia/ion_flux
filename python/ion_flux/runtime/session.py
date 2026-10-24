import numpy as np
import scipy.linalg
from typing import Dict, Any, Optional
from contextlib import contextmanager

from ion_flux.runtime.eis import solve_eis
from ion_flux.runtime._2_initializers import evaluate_ic
from ion_flux.runtime._4_diagnostics import format_native_crash

try:
    from ion_flux._core import SolverHandle, SundialsHandle
    RUST_FFI_AVAILABLE = True
except ModuleNotFoundError:
    # Safe fallback for pure-Python AST testing
    RUST_FFI_AVAILABLE = False


class TerminalController:
    """Explicit actuator control surface mapping directly to underlying terminal multiplexers."""
    __slots__ = ["_session", "_mode_offset", "_i_offset", "_v_offset"]

    def __init__(self, session: "Session"):
        self._session = session
        meta = session.manifest.terminal_meta
        if not meta["has_terminal"]:
            raise RuntimeError("Model does not declare an fx.Terminal abstraction.")
        self._mode_offset = meta["mode_offset"]
        self._i_offset = meta["i_target_offset"]
        self._v_offset = meta["v_target_offset"]

    def set_current(self, amps: float) -> None:
        mode_changed = (self._session._current_term_mode != 1.0)
        if mode_changed:
            self._session.set_parameter_by_offset(self._mode_offset, 1.0)
            self._session._current_term_mode = 1.0
        self._session.set_parameter_by_offset(self._i_offset, float(amps))
        if mode_changed:
            self._session.reinitialize_algebraic()

    def set_voltage(self, volts: float) -> None:
        mode_changed = (self._session._current_term_mode != 0.0)
        if mode_changed:
            self._session.set_parameter_by_offset(self._mode_offset, 0.0)
            self._session._current_term_mode = 0.0
        self._session.set_parameter_by_offset(self._v_offset, float(volts))
        if mode_changed:
            self._session.reinitialize_algebraic()

    def rest(self) -> None:
        self.set_current(0.0)


class Session:
    """
    Stateful execution session for Hardware/Software-in-the-Loop co-simulation.
    Keeps native memory, sparse matrices, and integration history "hot" in hardware.
    """
    def __init__(self, engine: Any, parameters: Dict[str, float], soc: Optional[float] = None, debug: bool = False, mode: str = "scientific"):
        self.engine = engine
        self.manifest = engine.manifest
        self.mode = mode
        is_realtime = (mode == "realtime")
        
        self.parameters = {**{k: v.value for k, v in engine.parameters.items()}, **parameters}
        self.time = 0.0
        self._history = {"Time [s]": [0.0]}
        self.debug = debug
        
        y0, ydot0 = evaluate_ic(self.manifest, self.parameters)
        p_list = self.manifest.pack_parameters(self.parameters)
        m_list = self.manifest.layout.get_mesh_data()
        
        self.id_arr = np.array(self.manifest.id_arr)
        constraints = [0.0] * self.manifest.layout.n_states
        
        self._offset_to_param = {offset: name for name, (offset, _) in self.manifest.layout.param_offsets.items()}
        self._current_term_mode = self.parameters.get("_term_mode", 1.0)
        self.terminal = TerminalController(self) if self.manifest.terminal_meta["has_terminal"] else None

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
                    c_seeds, c_ptrs, c_rows, c_cols, c_dense, is_realtime
                )
                try:
                    self.handle.calc_algebraic_roots()
                except RuntimeError as e:
                    raise format_native_crash(e, self.manifest) from None
        else:
            self.handle = None
            self._mock_y = np.array(y0)

    @contextmanager
    def suspend_history(self):
        """Safely suspends trajectory recording, guaranteeing restoration via try/finally block."""
        was_recording = getattr(self, "record_history", False)
        self.record_history = False
        try:
            yield
        finally:
            self.record_history = was_recording

    def set_parameter_by_offset(self, offset: int, value: float) -> None:
        param_name = self._offset_to_param.get(offset)
        if param_name:
            self.parameters[param_name] = value
        if self.handle:
            self.handle.set_parameter(offset, value)

    def set_parameter(self, param_name: str, value: float) -> None:
        self.parameters[param_name] = value 
        if self.handle:
            offset = self.manifest.layout.get_param_offset(param_name)
            self.handle.set_parameter(offset, value)

    def reinitialize_algebraic(self) -> None:
        if self.handle:
            try:
                self.handle.calc_algebraic_roots()
            except RuntimeError as e:
                raise format_native_crash(e, self.manifest) from None

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

    def get_scalar(self, variable_name: str) -> float:
        arr = self.get_array(variable_name)
        if arr.size != 1:
            raise ValueError(
                f"State/Observable '{variable_name}' is a spatial field with size {arr.size}. "
                f"Use get_array() instead."
            )
        return float(arr[0])

    def get(self, variable_name: str) -> float:
        return self.get_scalar(variable_name)

    def read_state_into(self, buffer: np.ndarray) -> None:
        if not isinstance(buffer, np.ndarray) or buffer.dtype != np.float64:
            raise TypeError("Target buffer must be a contiguous numpy.ndarray of dtype float64.")
        if buffer.size != self.manifest.layout.n_states:
            raise ValueError(
                f"Target buffer size ({buffer.size}) must match state size ({self.manifest.layout.n_states})."
            )
        if self.handle:
            self.handle.read_state_into(buffer)
        else:
            buffer[:] = self._mock_y

    def read_observables_into(self, buffer: np.ndarray) -> None:
        if not isinstance(buffer, np.ndarray) or buffer.dtype != np.float64:
            raise TypeError("Target buffer must be a contiguous numpy.ndarray of dtype float64.")
        if buffer.size != self.manifest.layout.n_obs:
            raise ValueError(
                f"Target buffer size ({buffer.size}) must match observables size ({self.manifest.layout.n_obs})."
            )
        if self.handle:
            self.handle.read_observables_into(buffer)
        else:
            buffer.fill(0.0)

    def step(self, dt: float) -> None:
        if self.handle:
            if getattr(self, "record_history", False):
                if hasattr(self.handle, "step_history"):
                    try:
                        mt, my, mydot = self.handle.step_history(dt)
                    except RuntimeError as e:
                        raise format_native_crash(e, self.manifest) from None
                        
                    if len(mt) > 0:
                        self.micro_t.append(mt)
                        self.micro_y.append(my)
                        self.micro_ydot.append(mydot)
                        
                        p_arr = np.tile(self.manifest.pack_parameters(self.parameters), (len(mt), 1))
                        self.micro_p.append(p_arr)
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
            
            if hasattr(self.handle, "get_time"):
                self.time = self.handle.get_time()
            else:
                self.time += dt
        else:
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
            from ion_flux.compiler._1_frontend.core import Condition
            condition = Condition(condition)
        elif hasattr(condition, "expression"):
            from ion_flux.compiler._1_frontend.core import Condition
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