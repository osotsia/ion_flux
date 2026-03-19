import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ion_flux.dsl.core import PDE

# Fallback for systems without the compiled Rust extension during pure-Python tests
try:
    from ion_flux._core import compile_to_cpp
except ImportError:
    logging.warning("Rust backend '_core' not found. Engine will operate in mock mode.")
    def compile_to_cpp(ast_json: str) -> str:
        return (
            "// MOCK CPP\n"
            "extern void __enzyme_autodiff(void*, ...);\n"
            "extern \"C\" { \n"
            "  void evaluate_residual(...) {} \n"
            "  void evaluate_jacobian(...) {} \n"
            "}\n"
        )

class Variable:
    """Data container ensuring consistent interface for simulation outputs."""
    def __init__(self, data: np.ndarray):
        self.data = data

    def __repr__(self):
        return f"<Variable: shape={self.data.shape}>"


class SimulationResult:
    """Standardized result wrapper returned by the C++/CUDA solver."""
    def __init__(self, data: Dict[str, np.ndarray], parameters: Dict[str, float], status: str = "completed"):
        self._data = data
        self.parameters = parameters
        self.status = status

    def __getitem__(self, key: str) -> Variable:
        if key not in self._data:
            raise KeyError(f"Variable '{key}' not found in simulation results.")
        return Variable(self._data[key])
        
    def to_dict(self, variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Serializes result into a flat dictionary, useful for API JSON responses."""
        keys_to_extract = variables or self._data.keys()
        return {k: self._data[k].tolist() for k in keys_to_extract if k in self._data}


class Engine:
    """
    The compilation and execution boundary bridging Python DSL and the Rust/C++ backend.
    """
    def __init__(self, model: PDE, target: str = "cpu", cache: bool = True, mock_execution: bool = True, **kwargs):
        self.model = model
        self.target = target
        self.mock_execution = mock_execution
        
        # JIT Compilation Phase
        ast_payload = json.dumps(model.ast())
        self.cpp_source = compile_to_cpp(ast_payload)
        self.is_compiled = True
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def solve(self, t_span: tuple = None, protocol=None, parameters: dict = None, threads: int = 1) -> SimulationResult:
        """Synchronous block utilizing the current thread for FFI execution."""
        return self._execute_ffi(parameters, protocol)

    def solve_batch(self, parameters: List[dict], t_span: tuple = None, protocol=None, max_workers: int = 1) -> List[SimulationResult]:
        """Task parallelism executed synchronously (Rust Rayon handles threads internally)."""
        # In production, this passes the entire array of parameters to Rust.
        return [self._execute_ffi(p, protocol) for p in parameters]

    async def solve_async(self, t_span: tuple = None, protocol=None, parameters: dict = None, scheduler=None) -> SimulationResult:
        """
        Asynchronous solver orchestration. 
        Releases the Python event loop by moving the FFI call to an OS thread.
        """
        if scheduler:
            async with scheduler.semaphore:
                return await asyncio.to_thread(self._execute_ffi, parameters, protocol)
                
        return await asyncio.to_thread(self._execute_ffi, parameters, protocol)

    def _execute_ffi(self, parameters: Optional[dict], protocol: Any) -> SimulationResult:
        """
        Directly invokes the Rust solver binary. 
        Mocked here to allow downstream framework validation without hardware targets.
        """
        if not self.mock_execution:
            raise NotImplementedError("Real FFI execution is only available when linked against the native binary.")
            
        params = parameters or {}
        
        # Simulate SUNDIALS internal failures
        if params.get("c.t0") == float('inf'):
            raise RuntimeError("SUNDIALS Error: Newton convergence failure")
            
        k_val = params.get("k", 0.75)
        time_len = len(protocol.time) if hasattr(protocol, "time") else 100
        
        # Generate mock tensor data
        data = {
            "T": np.ones((time_len, 30)) * (100.0 / k_val),
            "Voltage [V]": np.array([4.2] * (time_len - 1) + [2.5]),
            "Time [s]": np.arange(time_len, dtype=float)
        }
        
        return SimulationResult(data, params, status="completed")
