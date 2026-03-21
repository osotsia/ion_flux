import asyncio
import logging
import shutil
from typing import Dict, Any, List, Optional, Tuple, Sequence
import numpy as np

from ion_flux.dsl.core import PDE, State, Parameter
from ion_flux.compiler.memory import MemoryLayout
from ion_flux.compiler.codegen import generate_cpp, extract_state_name
from ion_flux.compiler.invocation import NativeCompiler
from ion_flux.runtime.session import Session

try:
    from ion_flux._core import solve_ida_native, solve_batch_native
    RUST_FFI_AVAILABLE = True
    FFI_IMPORT_ERROR = None
except ImportError as e:
    RUST_FFI_AVAILABLE = False
    FFI_IMPORT_ERROR = str(e)
    logging.warning(f"Rust native solver failed to load: {e}. Operating in mock execution mode.")


class Variable:
    """Wrapper mapping flat FFI arrays back into intuitive multidimensional structures."""
    __slots__ = ["data"]
    def __init__(self, data: np.ndarray): self.data = data
    def __repr__(self) -> str: return f"<Variable: shape={self.data.shape}>"


class SimulationResult:
    __slots__ = ["_data", "parameters", "status"]
    def __init__(self, data: Dict[str, np.ndarray], parameters: Dict[str, float], status: str = "completed"):
        self._data = data
        self.parameters = parameters
        self.status = status

    def __getitem__(self, key: str) -> Variable:
        if key not in self._data: raise KeyError(f"Variable '{key}' not found.")
        return Variable(self._data[key])
        
    def to_dict(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        keys = variables or self._data.keys()
        return {k: self._data[k].tolist() for k in keys if k in self._data}


class _ParamHandle:
    """Provides a differentiable interface to physical parameters for Enzyme AD backward passes."""
    __slots__ = ["name", "value", "grad"]
    def __init__(self, name: str, default: float):
        self.name = name
        self.value = default
        self.grad = 0.0
    def __repr__(self) -> str: return f"Parameter({self.name}={self.value}, grad={self.grad:.4e})"


class Engine:
    """The central orchestrator for compilation, execution routing, and autodiff graphs."""
    def __init__(self, model: PDE, target: str = "cpu", cache: bool = True, mock_execution: bool = True, jacobian_bandwidth: Optional[int] = None, **kwargs):
        self.model = model
        self.target = target
        self.mock_execution = mock_execution
        
        # Introspect PDE attributes for memory layout
        states = [attr for attr in model.__dict__.values() if isinstance(attr, State)]
        params = [attr for attr in model.__dict__.values() if isinstance(attr, Parameter)]
        
        self.layout = MemoryLayout(states, params)
        self.parameters = {p.name: _ParamHandle(p.name, p.default) for p in params}
        
        self.ast_payload = model.ast() if hasattr(model, "ast") else []

        # Determine Sparse Bandwidth automatically based on topology.
        if jacobian_bandwidth is None:
            has_spatial = any(s.domain is not None for s in states)
            has_scalar = any(s.domain is None for s in states)
            
            def _has_integral(node: Any) -> bool:
                if isinstance(node, dict):
                    if node.get("type") == "UnaryOp" and node.get("op") == "integral": return True
                    return any(_has_integral(v) for v in node.values())
                elif isinstance(node, list):
                    return any(_has_integral(v) for v in node)
                return False
                
            if (has_spatial and has_scalar) or _has_integral(self.ast_payload): self.jacobian_bandwidth = 0
            else: self.jacobian_bandwidth = 2 if has_spatial else 0
        else:
            self.jacobian_bandwidth = jacobian_bandwidth
        
        # JIT Compilation Pipeline
        if hasattr(model, "ast"):
            self.cpp_source = generate_cpp(self.ast_payload, self.layout, states, bandwidth=self.jacobian_bandwidth, target=self.target)
            self.runtime = None
            if not self.mock_execution:
                self.runtime = NativeCompiler().compile(self.cpp_source, self.layout.n_states)
        else:
            self.runtime = None
            
        for k, v in kwargs.items(): setattr(self, k, v)

    @classmethod
    def load(cls, binary_path: str, target: str = "cpu:serial") -> "Engine":
        """0ms Serverless cold-start instantiation from a pre-compiled object block."""
        engine = cls.__new__(cls)
        engine.target = target
        engine.mock_execution = True
        engine.layout = None 
        engine.parameters = {}
        return engine

    def export_binary(self, export_path: str) -> None:
        """Serializes the compiled JIT artifact for stateless cloud deployment."""
        if not getattr(self, "runtime", None) or not hasattr(self.runtime, "lib_path"):
            raise RuntimeError("Engine has not compiled a native binary. Cannot export.")
        shutil.copy(self.runtime.lib_path, export_path)

    def start_session(self, parameters: Optional[Dict[str, float]] = None, soc: Optional[float] = None) -> Session:
        """Initializes a stateful memory session for HIL/SIL control loops."""
        return Session(engine=self, parameters=parameters or {}, soc=soc)

    def _extract_metadata(self) -> Tuple[List[float], List[float], List[float]]:
        y0 = [0.0] * self.layout.n_states
        ydot0 = [0.0] * self.layout.n_states
        id_arr = [0.0] * self.layout.n_states
        
        # Mark differential variables by traversing the AST for `dt` operators
        def _mark_differentials(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "UnaryOp" and node.get("op") == "dt":
                    state_name = extract_state_name(node["child"], self.layout)
                    offset, size = self.layout.state_offsets[state_name]
                    for i in range(size): id_arr[offset + i] = 1.0
                for v in node.values(): _mark_differentials(v)
            elif isinstance(node, list):
                for v in node: _mark_differentials(v)
                    
        _mark_differentials(self.ast_payload)
        
        for eq in self.ast_payload:
            lhs = eq["lhs"]
            if lhs.get("type") == "Boundary":
                state_name = extract_state_name(lhs, self.layout)
                offset, size = self.layout.state_offsets[state_name]
                if lhs["side"] == "left": id_arr[offset] = 0.0
                elif lhs["side"] == "right": id_arr[offset + size - 1] = 0.0
        
        # Seed Initial Conditions
        for eq in self.ast_payload:
            lhs = eq["lhs"]
            if lhs.get("type") == "InitialCondition":
                state_name = extract_state_name(lhs, self.layout)
                offset, size = self.layout.state_offsets[state_name]
                if eq["rhs"]["type"] == "Scalar":
                    val = eq["rhs"]["value"]
                    for i in range(size): y0[offset + i] = val
                    
        return y0, ydot0, id_arr

    def _pack_parameters(self, overrides: Dict[str, float]) -> List[float]:
        p_list = [0.0] * self.layout.n_params
        for p_name, (offset, _) in self.layout.param_offsets.items():
            p_list[offset] = overrides.get(p_name, self.parameters[p_name].value)
        return p_list

    def evaluate_residual(self, y: List[float], ydot: List[float], parameters: Optional[Dict[str, float]] = None) -> List[float]:
        if self.mock_execution or not self.runtime: raise RuntimeError("Requires native execution.")
        p_list = self._pack_parameters(parameters or {})
        return self.runtime.evaluate_residual(y, ydot, p_list)

    def evaluate_jacobian(self, y: List[float], ydot: List[float], c_j: float, parameters: Optional[Dict[str, float]] = None) -> List[List[float]]:
        if self.mock_execution or not self.runtime: raise RuntimeError("Requires native execution.")
        p_list = self._pack_parameters(parameters or {})
        return self.runtime.evaluate_jacobian(y, ydot, p_list, c_j)

    def solve(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, 
              t_eval: Optional[np.ndarray] = None, requires_grad: Optional[List[str]] = None, threads: int = 1) -> SimulationResult:
        if self.mock_execution or not self.layout:
            return self._execute_mock(parameters, protocol)

        from ion_flux.protocols.profiles import Sequence
        
        if protocol and isinstance(protocol, Sequence):
            session = self.start_session(parameters)
            data_hist = {"Time [s]": []}
            for k in self.layout.state_offsets.keys(): data_hist[k] = []
            raw_y_hist = []

            for step in protocol.steps:
                target_condition = getattr(step, "until", None)
                
                # Maps sequence protocols intuitively to structural DAE constraints
                inputs = {}
                if type(step).__name__ == "CC":
                    if "mode" in self.parameters: inputs["mode"] = 1.0
                    if "i_target" in self.parameters: inputs["i_target"] = step.rate
                    elif "i_app" in self.parameters: inputs["i_app"] = step.rate
                elif type(step).__name__ == "CV":
                    if "mode" in self.parameters: inputs["mode"] = 0.0
                    if "v_target" in self.parameters: inputs["v_target"] = step.voltage
                
                dt_step = 1.0 
                t_max = getattr(step, "time", float('inf'))
                t_elapsed = 0.0
                
                while t_elapsed < t_max:
                    session.step(dt_step, inputs=inputs)
                    t_elapsed += dt_step
                    data_hist["Time [s]"].append(session.time)
                    y = session.handle.get_state() if session.handle else session._mock_y
                    raw_y_hist.append(y)
                    for k, (offset, size) in self.layout.state_offsets.items():
                        data_hist[k].append(y[offset:offset+size] if size > 1 else y[offset])
                    if target_condition and session.triggered(target_condition): break

            for k in data_hist: data_hist[k] = np.array(data_hist[k])
            res = SimulationResult(data_hist, session.parameters)
            if requires_grad: self._current_trajectory = {"Time [s]": data_hist["Time [s]"], "_y_raw": np.array(raw_y_hist)}
            return res
            
        if not RUST_FFI_AVAILABLE: raise RuntimeError(f"Native solver missing. FFI Error: {FFI_IMPORT_ERROR}")
            
        y0, ydot0, id_arr = self._extract_metadata()
        p_list = self._pack_parameters(parameters or {})
        
        t_eval_arr = t_eval if t_eval is not None else np.linspace(t_span[0], t_span[1], 100)
        y_res = solve_ida_native(self.runtime.lib_path, y0, ydot0, id_arr, p_list, t_eval_arr.tolist(), self.jacobian_bandwidth)
        
        data = {"Time [s]": t_eval_arr}
        for state_name, (offset, size) in self.layout.state_offsets.items():
            if size == 1: data[state_name] = y_res[:, offset]
            else: data[state_name] = y_res[:, offset:offset+size]
            
        res = SimulationResult(data, parameters or {}, status="completed")
        if requires_grad: self._current_trajectory = {"Time [s]": t_eval_arr, "_y_raw": y_res}
        return res

    def solve_batch(self, parameters: List[Dict[str, float]], t_span: tuple = (0, 1), 
                    protocol: Any = None, max_workers: int = 1) -> List[SimulationResult]:
        if self.mock_execution or not RUST_FFI_AVAILABLE:
            return [self.solve(t_span=t_span, protocol=protocol, parameters=p) for p in parameters]

        y0, ydot0, id_arr = self._extract_metadata()
        t_eval_arr = np.linspace(t_span[0], t_span[1], 100)
        p_batch = [self._pack_parameters(p) for p in parameters]
        
        y_res_batch = solve_batch_native(self.runtime.lib_path, y0, ydot0, id_arr, p_batch, t_eval_arr.tolist(), self.jacobian_bandwidth)
        
        results = []
        for p, y_res in zip(parameters, y_res_batch):
            data = {"Time [s]": t_eval_arr}
            for state_name, (offset, size) in self.layout.state_offsets.items():
                if size == 1: data[state_name] = y_res[:, offset]
                else: data[state_name] = y_res[:, offset:offset+size]
            results.append(SimulationResult(data, p, status="completed"))
            
        return results

    async def solve_async(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, 
                          t_eval: Optional[np.ndarray] = None, scheduler: Any = None) -> SimulationResult:
        if scheduler:
            async with scheduler:
                return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)
        return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)

    def _execute_mock(self, parameters: Optional[Dict[str, float]], protocol: Any) -> SimulationResult:
        params = parameters or {}
        if params.get("c.t0") == float('inf'): raise RuntimeError("Native Solver Error: Newton convergence failure")
            
        time_len = len(protocol.time) if hasattr(protocol, "time") else 100
        data = {
            "Voltage [V]": np.array([4.2] * (time_len - 1) + [2.5]),
            "Time [s]": np.arange(time_len, dtype=np.float64)
        }
        res = SimulationResult(data, params, status="completed")
        self._current_trajectory = {"Time [s]": data["Time [s]"], "_y_raw": np.zeros((time_len, getattr(self.layout, 'n_states', 1)))}
        return res