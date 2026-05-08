import asyncio
import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional

from ion_flux.dsl.core import PDE
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
    def telemetry(self) -> TelemetryReport: 
        return TelemetryReport(self.manifest.layout.n_states, self.manifest.jacobian_bandwidth)

    def start_session(self, parameters: Optional[Dict[str, float]] = None, soc: Optional[float] = None) -> Session:
        return Session(engine=self, parameters=parameters or {}, soc=soc, debug=self.debug)

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
        if threads > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = str(threads)
            if getattr(self.manifest, "runtime", None): 
                self.manifest.runtime.set_spatial_threads(threads)
                
        if self.mock_execution or not self.manifest: 
            from ion_flux.runtime._3_dispatch import execute_mock
            return execute_mock(self.manifest, parameters, protocol)

        from ion_flux.protocols.profiles import Sequence
        if protocol and isinstance(protocol, Sequence):
            return self._orchestrate_sequence(protocol, parameters, requires_grad, show_progress)
            
        from ion_flux.runtime._2_initializers import evaluate_ic
        from ion_flux.runtime._3_dispatch import run_single
        
        current_params = {**{k: v.value for k, v in self.parameters.items()}, **(parameters or {})}
        y0, ydot0 = evaluate_ic(self.manifest, current_params)
        
        t_eval_arr = t_eval if t_eval is not None else np.linspace(t_span[0], t_span[1], 100)
        v_idx = self.manifest.layout.state_offsets.get("V_cell", (-1, 0))[0]
        
        y_res, obs_res, micro_t, micro_y, micro_ydot = run_single(
            self.manifest, y0, ydot0, current_params, t_eval_arr, 
            self.solver_backend, self.debug, show_progress, 
            requires_grad is not None, v_idx
        )
        
        data = {"Time [s]": t_eval_arr}
        for state_name, (offset, size) in self.manifest.layout.state_offsets.items():
            if size == 1: data[state_name] = y_res[:, offset]
            else: data[state_name] = y_res[:, offset:offset+size]
        for obs_name, (offset, size) in self.manifest.layout.obs_offsets.items():
            if size == 1: data[obs_name] = obs_res[:, offset]
            else: data[obs_name] = obs_res[:, offset:offset+size]
            
        trajectory = None
        if requires_grad: 
            p_list = self.manifest.pack_parameters(current_params)
            trajectory = {
                "Time [s]": t_eval_arr, "_y_raw": y_res, "_micro_t": micro_t, 
                "_micro_y": micro_y, "_micro_ydot": micro_ydot, 
                "_p_traj": [p_list]*len(micro_t), "requires_grad": requires_grad
            }
        return SimulationResult(data, current_params, status="completed", engine=self, trajectory=trajectory)

    def solve_batch(self, parameters: List[Dict[str, float]], t_span: tuple = (0, 1), protocols: Any = None, max_workers: int = 1, show_progress: bool = False) -> List[SimulationResult]:
        from ion_flux.protocols.profiles import Sequence
        if protocols:
            if isinstance(protocols, Sequence): protocols = [protocols] * len(parameters)
            elif len(protocols) != len(parameters): raise ValueError("Batch length mismatch.")
                
        if max_workers > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = "1"
            if getattr(self.manifest, "runtime", None): 
                self.manifest.runtime.set_spatial_threads(1)
            
        if self.mock_execution:
            if not protocols: protocols = [None] * len(parameters)
            return [self.solve(t_span=t_span, protocol=prot, parameters=p) for p, prot in zip(parameters, protocols)]

        from ion_flux.runtime._2_initializers import evaluate_ic
        from ion_flux.runtime._3_dispatch import run_batch
        
        # In batch mode, y0 is evaluated using the default parameters to prevent O(N) evaluation bounds overhead.
        default_params = {k: v.value for k, v in self.parameters.items()}
        y0, ydot0 = evaluate_ic(self.manifest, default_params)
        
        t_eval_arr = np.linspace(t_span[0], t_span[1], 100)
        v_idx = self.manifest.layout.state_offsets.get("V_cell", (-1, 0))[0]
        
        results = []
        y_res_batch = run_batch(
            self.manifest, y0, ydot0, parameters, t_eval_arr, protocols, 
            max_workers, self.debug, show_progress, v_idx
        )
        
        for p, (t_res, y_res, obs_res) in zip(parameters, y_res_batch):
            data = {"Time [s]": t_res}
            for state_name, (offset, size) in self.manifest.layout.state_offsets.items():
                if size == 1: data[state_name] = y_res[:, offset]
                else: data[state_name] = y_res[:, offset:offset+size]
            for obs_name, (offset, size) in self.manifest.layout.obs_offsets.items():
                if size == 1: data[obs_name] = obs_res[:, offset]
                else: data[obs_name] = obs_res[:, offset:offset+size]
            results.append(SimulationResult(data, p, status="completed", engine=self, trajectory=None))

        return results

    async def solve_async(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, t_eval: Optional[np.ndarray] = None, scheduler: Any = None) -> SimulationResult:
        if scheduler:
            async with scheduler: 
                return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)
        return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)

    def _orchestrate_sequence(self, protocol, parameters, requires_grad, show_progress) -> SimulationResult:
        """Maintains Python control flow to handle Bisection Root triggers across Sequence boundaries."""
        session = self.start_session(parameters)
        data_hist = {"Time [s]": []}
        for k in self.manifest.layout.state_offsets.keys(): data_hist[k] = []
        for k in self.manifest.layout.obs_offsets.keys(): data_hist[k] = []
        raw_y_hist, raw_p_hist = [], []

        if requires_grad:
            session.record_history = True
            
            # Start with explicitly dimensioned arrays to prevent np.vstack shape mismatches
            y0 = session.handle.get_state() if session.handle else session._mock_y
            ydot0 = np.zeros(self.manifest.layout.n_states)
            p0 = self.manifest.pack_parameters(session.parameters)
            
            session.micro_t = [np.array([0.0])]
            session.micro_y = [y0[np.newaxis, :]]
            session.micro_ydot = [ydot0[np.newaxis, :]]
            session.micro_p = [np.array(p0)[np.newaxis, :]]

        for step in protocol.steps:
            target_condition = getattr(step, "until", None)
            inputs = {}
            step_name = type(step).__name__
            
            if step_name == "CC":
                if "_term_mode" in self.parameters: inputs["_term_mode"], inputs["_term_i_target"] = 1.0, step.rate
                else: 
                    if "mode" in self.parameters: inputs["mode"] = 1.0
                    if "i_target" in self.parameters: inputs["i_target"] = step.rate
                    elif "i_app" in self.parameters: inputs["i_app"] = step.rate
            elif step_name == "CV":
                if "_term_mode" in self.parameters: inputs["_term_mode"], inputs["_term_v_target"] = 0.0, step.voltage
                else: 
                    if "mode" in self.parameters: inputs["mode"] = 0.0
                    if "v_target" in self.parameters: inputs["v_target"] = step.voltage
            elif step_name == "Rest":
                if "_term_mode" in self.parameters: inputs["_term_mode"], inputs["_term_i_target"] = 1.0, 0.0
                else: 
                    if "mode" in self.parameters: inputs["mode"] = 1.0
                    if "i_target" in self.parameters: inputs["i_target"] = 0.0
                    elif "i_app" in self.parameters: inputs["i_app"] = 0.0
            
            dt_step = 1.0
            t_max = getattr(step, "time", float('inf'))
            t_elapsed = 0.0
            
            while t_elapsed < t_max:
                session.checkpoint()
                session.step(dt_step, inputs=inputs)
                
                if target_condition and session.triggered(target_condition):
                    session.restore()
                    
                    # Prevent speculative bisection steps from recording false branches into the AD Tape
                    with session.suspend_history():
                        low, high = 0.0, dt_step
                        for _ in range(15):
                            mid = (low + high) / 2.0
                            session.step(mid, inputs=inputs)
                            if session.triggered(target_condition): high = mid
                            else: low = mid
                            session.restore()
                    
                    # Re-enable the AD tape implicitly through the context manager exit, and take the accepted step
                    session.step(low, inputs=inputs)
                    t_elapsed += low
                    self._append_to_hist(session, data_hist, raw_y_hist, raw_p_hist, requires_grad)
                    break
                
                t_elapsed += dt_step
                self._append_to_hist(session, data_hist, raw_y_hist, raw_p_hist, requires_grad)

                if show_progress:
                    try: v_str = f" | V: {session.get('V_cell'):.3f}V"
                    except KeyError: v_str = ""
                    if t_max == float('inf'): sys.stdout.write(f"\r▶ {step_name:<4} ⏳ t: {session.time:.1f}s{v_str}   ")
                    else:
                        pct = min(t_elapsed / t_max, 1.0)
                        filled = int(pct * 30)
                        bar = "█" * filled + "-" * (30 - filled)
                        sys.stdout.write(f"\r▶ {step_name:<4} [{bar}] {pct*100:.1f}% | t: {session.time:.1f}s{v_str}   ")
                    sys.stdout.flush()

            if show_progress:
                try: v_str = f" | V: {session.get('V_cell'):.3f}V"
                except KeyError: v_str = ""
                sys.stdout.write(f"\r▶ {step_name:<4} [██████████████████████████████] 100.0% | t: {session.time:.1f}s{v_str}   \n")
                sys.stdout.flush()

        for k in data_hist: data_hist[k] = np.array(data_hist[k])
        
        trajectory = None
        if requires_grad:
            trajectory = {
                "Time [s]": data_hist["Time [s]"], 
                "_y_raw": np.array(raw_y_hist), 
                "_micro_t": np.concatenate(session.micro_t), 
                "_micro_y": np.vstack(session.micro_y),
                "_micro_ydot": np.vstack(session.micro_ydot), 
                "_p_traj": np.vstack(session.micro_p), 
                "requires_grad": requires_grad
            }
        return SimulationResult(data_hist, session.parameters, engine=self, trajectory=trajectory)

    def _append_to_hist(self, session, data_hist, raw_y_hist, raw_p_hist, requires_grad):
        data_hist["Time [s]"].append(session.time)
        y = session.handle.get_state() if session.handle else session._mock_y
        obs = session.handle.get_observables_py() if session.handle else np.zeros(self.manifest.layout.n_obs)
        raw_y_hist.append(y)
        if requires_grad: 
            raw_p_hist.append(self.manifest.pack_parameters(session.parameters))
            
        for k, (offset, size) in self.manifest.layout.state_offsets.items(): 
            data_hist[k].append(y[offset:offset+size] if size > 1 else y[offset])
        for k, (offset, size) in self.manifest.layout.obs_offsets.items(): 
            data_hist[k].append(obs[offset:offset+size] if size > 1 else obs[offset])