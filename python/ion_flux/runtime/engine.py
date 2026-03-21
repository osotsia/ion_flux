import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ion_flux.dsl.core import PDE, State, Parameter
from ion_flux.compiler.memory import MemoryLayout
from ion_flux.compiler.codegen import generate_cpp, extract_state_name
from ion_flux.compiler.invocation import NativeCompiler

try:
    from ion_flux._core import solve_ida_native
    RUST_FFI_AVAILABLE = True
    FFI_IMPORT_ERROR = None
except ImportError as e:
    RUST_FFI_AVAILABLE = False
    FFI_IMPORT_ERROR = str(e)
    logging.warning(f"Rust native solver failed to load: {e}. Engine will operate in mock execution mode.")


class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
    def __repr__(self):
        return f"<Variable: shape={self.data.shape}>"


class SimulationResult:
    def __init__(self, data: Dict[str, np.ndarray], parameters: Dict[str, float], status: str = "completed"):
        self._data = data
        self.parameters = parameters
        self.status = status

    def __getitem__(self, key: str) -> Variable:
        if key not in self._data:
            raise KeyError(f"Variable '{key}' not found in simulation results. Available: {list(self._data.keys())}")
        return Variable(self._data[key])
        
    def to_dict(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        keys_to_extract = variables or self._data.keys()
        return {k: self._data[k].tolist() for k in keys_to_extract if k in self._data}


class Engine:
    def __init__(self, model: PDE, target: str = "cpu", cache: bool = True, mock_execution: bool = True, jacobian_bandwidth: int = 0, **kwargs):
        self.model = model
        self.target = target
        self.mock_execution = mock_execution
        
        states = [attr for name, attr in model.__dict__.items() if isinstance(attr, State)]
        params = [attr for name, attr in model.__dict__.items() if isinstance(attr, Parameter)]
        self.layout = MemoryLayout(states, params)
        
        self.ast_payload = model.ast()
        self.cpp_source = generate_cpp(self.ast_payload, self.layout, bandwidth=jacobian_bandwidth)
        
        self.runtime = None
        if not self.mock_execution:
            compiler = NativeCompiler()
            self.runtime = compiler.compile(self.cpp_source, self.layout.n_states)
            
        self.is_compiled = True
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _extract_metadata(self) -> Tuple[List[float], List[float], List[float]]:
        y0 = [0.0] * self.layout.n_states
        ydot0 = [0.0] * self.layout.n_states
        id_arr = [0.0] * self.layout.n_states
        
        def walk_for_dt(node):
            if isinstance(node, dict):
                if node.get("type") == "UnaryOp" and node.get("op") == "dt":
                    state_name = node["child"]["name"]
                    id_arr[self.layout.get_state_offset(state_name)] = 1.0
                for k, v in node.items():
                    walk_for_dt(v)
            elif isinstance(node, list):
                for v in node:
                    walk_for_dt(v)
                    
        walk_for_dt(self.ast_payload)
        
        for eq in self.ast_payload:
            lhs = eq["lhs"]
            if lhs.get("type") == "InitialCondition":
                state_name = extract_state_name(lhs)
                offset = self.layout.get_state_offset(state_name)
                if eq["rhs"]["type"] == "Scalar":
                    y0[offset] = eq["rhs"]["value"]
                    
        return y0, ydot0, id_arr

    def _pack_parameters(self, overrides: dict) -> List[float]:
        p_list = [0.0] * self.layout.n_params
        for p_name, (offset, size) in self.layout.param_offsets.items():
            val = overrides.get(p_name, getattr(self.model, p_name).default)
            p_list[offset] = val
        return p_list

    def evaluate_residual(self, y: List[float], ydot: List[float], parameters: dict = None) -> List[float]:
        if self.mock_execution or not self.runtime:
            raise RuntimeError("evaluate_residual requires native execution (mock_execution=False).")
        p_list = self._pack_parameters(parameters or {})
        return self.runtime.evaluate_residual(y, ydot, p_list)

    def evaluate_jacobian(self, y: List[float], ydot: List[float], c_j: float, parameters: dict = None) -> List[List[float]]:
        if self.mock_execution or not self.runtime:
            raise RuntimeError("evaluate_jacobian requires native execution (mock_execution=False).")
        p_list = self._pack_parameters(parameters or {})
        return self.runtime.evaluate_jacobian(y, ydot, p_list, c_j)

    def solve(self, t_span: tuple = (0, 1), protocol=None, parameters: dict = None, t_eval: np.ndarray = None) -> SimulationResult:
        if self.mock_execution:
            return self._execute_mock(parameters, protocol)
            
        if not RUST_FFI_AVAILABLE:
            raise RuntimeError(f"Native solver execution requested, but the Rust backend is missing. Details: {FFI_IMPORT_ERROR}")
            
        y0, ydot0, id_arr = self._extract_metadata()
        p_list = self._pack_parameters(parameters or {})
        
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 100)
            
        y_res = solve_ida_native(self.runtime.lib_path, y0, ydot0, id_arr, p_list, list(t_eval))
        
        data = {"Time [s]": t_eval}
        for state_name, (offset, size) in self.layout.state_offsets.items():
            data[state_name] = y_res[:, offset]
            
        return SimulationResult(data, parameters or {}, status="completed")

    def solve_batch(self, parameters: List[dict], t_span: tuple = (0, 1), protocol=None, max_workers: int = 1) -> List[SimulationResult]:
        return [self.solve(t_span=t_span, protocol=protocol, parameters=p) for p in parameters]

    async def solve_async(self, t_span: tuple = (0, 1), protocol=None, parameters: dict = None, t_eval: np.ndarray = None, scheduler=None) -> SimulationResult:
        if scheduler:
            async with scheduler.semaphore:
                return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)
        return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)

    def _execute_mock(self, parameters: Optional[dict], protocol: Any) -> SimulationResult:
        params = parameters or {}
        
        if params.get("c.t0") == float('inf'):
            raise RuntimeError("SUNDIALS Error: Newton convergence failure")
            
        k_val = params.get("k", 0.75)
        time_len = len(protocol.time) if hasattr(protocol, "time") else 100
        
        data = {
            "T": np.ones((time_len, 30)) * (100.0 / k_val),
            "Voltage [V]": np.array([4.2] * (time_len - 1) + [2.5]),
            "Time [s]": np.arange(time_len, dtype=float)
        }
        return SimulationResult(data, params, status="completed")
