import asyncio
import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional

from ion_flux.compiler._1_frontend.core import PDE
from ion_flux.runtime.manifest import ExecutableManifest
from ion_flux.runtime.session import Session
from ion_flux.runtime.results import SimulationResult
from ion_flux.runtime.telemetry import TelemetryReport

class _ParamHandle:
    __slots__ = ["name", "value"]
    def __init__(self, name: str, default: float):
        self.name = name
        self.value = default
    def __repr__(self) -> str: return f"Parameter({self.name}={self.value})"

class Engine:
    """
    The Facade unifying the Compiler and Runtime architecture.
    Orchestrates the lifecycle of lowering Python ASTs into ExecutableManifests, 
    and delegating the FFI workflows for Stateful or Stateless solving.
    """
    def __init__(self, model: Optional[PDE] = None, target: str = "cpu:serial", solver_backend: str = "native", 
                 cache: bool = True, mock_execution: bool = False, jacobian_bandwidth: Optional[int] = None, 
                 debug: bool = False, **kwargs):
        self.model = model
        self.target = target
        self.solver_backend = solver_backend.lower()
        self.mock_execution = mock_execution
        self.debug = debug
        
        if model:
            from ion_flux.runtime._1_builder import build_manifest
            self.manifest = build_manifest(model, target, cache, jacobian_bandwidth, mock_execution)
        else:
            self.manifest = None
            
        self.parameters = {k: _ParamHandle(k, v) for k, v in self.manifest.default_parameters.items()} if self.manifest else {}
        for k, v in kwargs.items(): 
            setattr(self, k, v)

    # --- Properties ensuring backwards compatibility for Analytical Solvers (e.g., Metrics/EIS) ---
    @property
    def layout(self): 
        return self.manifest.layout if self.manifest else None
        
    @property
    def runtime(self): 
        return self.manifest.runtime if self.manifest else None

    @property
    def jacobian_bandwidth(self): 
        return self.manifest.jacobian_bandwidth if self.manifest else 0
    
    @property
    def cpp_source(self):                           
        return self.manifest.cpp_source if self.manifest else ""

    @property
    def _cpr_cache(self): 
        return self.manifest.cpr_cache if self.manifest else ([],[],[],[],[])

    def _pack_parameters(self, overrides): 
        return self.manifest.pack_parameters(overrides)

    def _extract_metadata(self):
        from ion_flux.runtime._2_initializers import evaluate_ic
        current_params = {k: v.value for k, v in self.parameters.items()}
        y0, ydot0 = evaluate_ic(self.manifest, current_params)
        return y0, ydot0, self.manifest.id_arr, self.manifest.spatial_diag, self.manifest.max_steps

    # --- Architectural Serialization Boundaries ---
    @classmethod
    def load(cls, binary_path: str, target: str = "cpu:serial", solver_backend: str = "native", debug: bool = False) -> "Engine":
        engine = cls.__new__(cls)
        engine.target = target
        engine.solver_backend = solver_backend
        engine.debug = debug
        engine.mock_execution = False
        
        engine.manifest = ExecutableManifest.load(binary_path)
        engine.parameters = {name: _ParamHandle(name, val) for name, val in engine.manifest.default_parameters.items()}
        return engine

    def export_binary(self, export_path: str) -> None:
        if not self.manifest or not self.manifest.lib_path:
            raise RuntimeError("Engine has not compiled a native binary. Cannot export.")
        self.manifest.save(export_path)

    @property
    def telemetry(self) -> Optional[TelemetryReport]: 
        return TelemetryReport(self.manifest) if self.manifest else None

    def start_session(self, parameters: Optional[Dict[str, float]] = None, soc: Optional[float] = None, mode: str = "scientific") -> Session:
        """
        Instantiates a persistent native solver session.

        Enables Hardware-in-the-Loop (HIL) and Real-Time control logic. BDF history 
        vectors, Nordsieck arrays, and sparse LU factorizations remain "hot" in hardware memory, 
        avoiding allocation overhead during continuous micro-stepping.

        Args:
            parameters (Dict[str, float], optional): Base parameter initialization for the session.
            soc (float, optional): Initial State of Charge.
            mode (str): Execution profile ("scientific" for adaptive VSVO BDF, "realtime" for deterministic bounds).

        Returns:
            Session: A stateful object holding the active FFI integration pointer.
        """
        return Session(engine=self, parameters=parameters or {}, soc=soc, debug=self.debug, mode=mode)

    # --- Primary Dispatch Hooks ---
    def evaluate_residual(self, y: List[float], ydot: List[float], parameters: Optional[Dict[str, float]] = None) -> List[float]:
        if self.mock_execution or not self.manifest.runtime: 
            raise RuntimeError("Requires native execution.")
        p_list = self.manifest.pack_parameters(parameters or {})
        m_list = self.manifest.layout.get_mesh_data()
        return self.manifest.runtime.evaluate_residual(y, ydot, p_list, m_list)

    def evaluate_jacobian(self, y: List[float], ydot: List[float], c_j: float, parameters: Optional[Dict[str, float]] = None) -> List[List[float]]:
        from ion_flux.runtime._3_dispatch import evaluate_jacobian
        return evaluate_jacobian(self.manifest, y, ydot, c_j, parameters or {})

    def solve(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, 
              t_eval: Optional[np.ndarray] = None, requires_grad: Optional[List[str]] = None, threads: int = 1, show_progress: bool = True) -> SimulationResult:
        """
        Executes a single continuous simulation trajectory.
        Canonicalizes inputs and delegates directly to solve_batch.
        """
        current_params = {**{k: v.value for k, v in self.parameters.items()}, **(parameters or {})}
        results = self.solve_batch(
            parameters=[current_params],
            t_span=t_span,
            protocols=[protocol] if protocol is not None else None,
            t_eval=t_eval,
            requires_grad=requires_grad,
            max_workers=1,
            threads=threads,
            show_progress=show_progress
        )
        return results[0]

    def solve_batch(self, parameters: List[Dict[str, float]], t_span: tuple = (0, 1), protocols: Any = None, 
                    t_eval: Optional[np.ndarray] = None, requires_grad: Optional[List[str]] = None, 
                    max_workers: int = 1, threads: int = 1, show_progress: bool = False) -> List[SimulationResult]:
        """
        Executes an array of independent models across multiple vCPUs concurrently.
        """
        from ion_flux.protocols import Sequence
        if protocols:
            if isinstance(protocols, Sequence): protocols = [protocols] * len(parameters)
            elif len(protocols) != len(parameters): raise ValueError("Batch length mismatch.")
        else:
            protocols = None

        total = len(parameters)
        spatial_threads = threads if total == 1 else 1

        if spatial_threads > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = str(spatial_threads)
        elif total > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = "1"

        if getattr(self.manifest, "runtime", None):
            self.manifest.runtime.set_spatial_threads(spatial_threads)

        if self.mock_execution or not self.manifest:
            from ion_flux.runtime._3_dispatch import execute_mock
            mock_protos = protocols or [None] * len(parameters)
            return [execute_mock(self.manifest, p, prot) for p, prot in zip(parameters, mock_protos)]

        from ion_flux.runtime._2_initializers import evaluate_ic
        from ion_flux.runtime._3_dispatch import dispatch_native

        init_params = {**{k: v.value for k, v in self.parameters.items()}, **(parameters[0] if parameters else {})}
        y0, ydot0 = evaluate_ic(self.manifest, init_params)

        t_eval_arr = t_eval if t_eval is not None else np.linspace(t_span[0], t_span[1], 100)

        def _get_v_idx():
            for k in ["V_cell", "V", "v", "Voltage"]:
                if k in self.manifest.layout.state_offsets:
                    return self.manifest.layout.state_offsets[k][0]
            return -1

        v_idx = _get_v_idx()

        dispatched = dispatch_native(
            self.manifest, y0, ydot0, parameters, t_eval_arr, protocols,
            self.solver_backend, requires_grad, max_workers, spatial_threads, show_progress, self.debug, v_idx
        )

        results = []
        for p, (t_res, y_res, obs_res, trajectory, diag_dict) in zip(parameters, dispatched):
            data = {"Time [s]": t_res}
            for state_name, (offset, size) in self.manifest.layout.state_offsets.items():
                if size == 1: data[state_name] = y_res[:, offset]
                else: data[state_name] = y_res[:, offset:offset+size]
            for obs_name, (offset, size) in self.manifest.layout.obs_offsets.items():
                if size == 1: data[obs_name] = obs_res[:, offset]
                else: data[obs_name] = obs_res[:, offset:offset+size]
            results.append(SimulationResult(data, p, status="completed", engine=self, trajectory=trajectory, diagnostics=diag_dict))

        return results

    async def solve_async(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, t_eval: Optional[np.ndarray] = None, scheduler: Any = None) -> SimulationResult:
        """
        Asynchronous wrapper for `solve`.
        """
        if scheduler:
            async with scheduler: 
                return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)
        return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)