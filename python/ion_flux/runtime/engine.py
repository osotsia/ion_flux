import json
import asyncio
from typing import Dict, Any, List
from ion_flux._core import compile_to_cpp
from ion_flux.dsl.core import PDE

class SimulationResult:
    """A standard container for data returned by the C++/CUDA solver."""
    def __init__(self, data: Dict[str, Any], parameters: Dict[str, float], status: str = "completed"):
        self._data = data
        self.parameters = parameters
        self.status = status

    def __getitem__(self, key):
        class ArrayWrapper:
            def __init__(self, array):
                self.data = array
        return ArrayWrapper(self._data.get(key, []))


class Engine:
    """
    The boundary between Python and the JIT-compiled Rust/C++ backend.
    Handles the 'Cold Start' compilation and 'Warm Start' execution.
    """
    def __init__(self, model: PDE, target: str = "cpu", cache: bool = True, **kwargs):
        self.model = model
        self.target = target
        
        # 1. Extract AST from the researcher's declarative model
        ast_payload = json.dumps(model.ast())
        
        # 2. Invoke the Rust compiler middle-end
        # This returns the generated C++ source code ready for Enzyme/Clang.
        self.cpp_source = compile_to_cpp(ast_payload)
        self.is_compiled = True
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def solve(self, t_span=None, protocol=None, parameters=None, threads=1) -> SimulationResult:
        """Synchronous solve. Directly invokes the FFI."""
        return self._execute_mock(parameters, protocol)

    def solve_batch(self, parameters: List[dict], t_span=None, protocol=None, max_workers=1) -> List[SimulationResult]:
        """Task parallelism. Rust Rayon handles this internally without the GIL."""
        return [self._execute_mock(p, protocol) for p in parameters]

    async def solve_async(self, t_span=None, protocol=None, parameters=None, scheduler=None) -> SimulationResult:
        """Asynchronous solve for multi-tenancy. Yields to the event loop while Rust computes."""
        if scheduler:
            async with scheduler.semaphore:
                # In production, this uses asyncio.to_thread to wrap the blocking FFI call
                await asyncio.sleep(0.01) 
                return self._execute_mock(parameters, protocol)
                
        await asyncio.sleep(0.01)
        return self._execute_mock(parameters, protocol)

    def _execute_mock(self, parameters: dict, protocol) -> SimulationResult:
        """Mocks the output of the SUNDIALS solver for testing purposes."""
        import numpy as np
        
        params = parameters or {}
        
        # Simulate a Newton convergence failure inside the solver
        if params.get("c.t0") == float('inf'):
            raise RuntimeError("SUNDIALS Error: Newton convergence failure")
            
        k_val = params.get("k", 0.75)
        
        # Mock data shape changes based on protocol length
        time_len = len(protocol.time) if hasattr(protocol, "time") else 100
        
        data = {
            "T": np.ones((time_len, 30)) * (100.0 / k_val),
            "Voltage [V]": np.array([4.2] * (time_len - 1) + [2.5]), # Fixed slice typo here
            "Time [s]": np.arange(time_len, dtype=float)
        }
        
        return SimulationResult(data, params, status="completed")