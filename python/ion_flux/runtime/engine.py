import asyncio
import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional

from ion_flux.stage1_dsl.core import PDE
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
        """
        Instantiates a persistent native solver session.

        Enables Hardware-in-the-Loop (HIL) and Real-Time control logic. BDF history 
        vectors, Nordsieck arrays, and sparse LU factorizations remain "hot" in hardware memory, 
        avoiding allocation overhead during continuous micro-stepping.

        Args:
            parameters (Dict[str, float], optional): Base parameter initialization for the session.
            soc (float, optional): Initial State of Charge.

        Returns:
            Session: A stateful object holding the active FFI integration pointer.
        """
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
        """
        Executes a single continuous simulation trajectory.

        The primary execution boundary bridging Python to the Native Solver. Maps user 
        parameters directly into C-ABI flat arrays, dynamically evaluates Initial Conditions, 
        and drives the implicit BDF integration without Python GIL interference.

        Args:
            t_span (tuple): A tuple of (start_time, end_time) in seconds.
            protocol (Any, optional): Defines dynamic algebraic boundary constraints (e.g., Constant 
                Current, Constant Voltage sequences). Overrides `t_span` if provided.
            parameters (Dict[str, float], optional): Dictionary of parameter overrides.
            t_eval (np.ndarray, optional): Specific time points to record the solution.
            requires_grad (List[str], optional): Declares parameters for sensitivity analysis. 
                Instructs the native solver to record the forward integration trajectory, enabling 
                continuous reverse-mode Automatic Differentiation (AD).
            threads (int): Number of OpenMP threads to allocate natively if using a data-parallel target.
            show_progress (bool): Whether to display a terminal progress bar.

        Returns:
            SimulationResult: A data object containing the multidimensional trajectory arrays.
        """
        if threads > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = str(threads)
            if getattr(self.manifest, "runtime", None): 
                self.manifest.runtime.set_spatial_threads(threads)
                
        if self.mock_execution or not self.manifest: 
            from ion_flux.runtime._3_dispatch import execute_mock
            return execute_mock(self.manifest, parameters, protocol)

        from ion_flux.protocols import Sequence
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
        """
        Executes an array of independent models across multiple vCPUs concurrently.

        Resolves Python's multi-processing bottlenecks. Pushes the entire task matrix 
        down into the compiled Rust Rayon thread-pool, achieving near-linear scaling.

        Args:
            parameters (List[Dict[str, float]]): List of parameter dictionary permutations to solve.
            t_span (tuple): A tuple of (start_time, end_time) in seconds.
            protocols (Any, optional): Can be a single `Sequence` (which is automatically broadcast 
                to every parameter payload in the batch) or a `List[Sequence]` mapping exactly 1:1 
                with the `parameters` list to run unique, isolated protocols per model.
            max_workers (int): Size of the Native thread-pool. Completely bypasses the Python GIL.
            show_progress (bool): Whether to display a terminal progress bar.

        Returns:
            List[SimulationResult]: A list of result objects corresponding to the input parameters.
        """
        from ion_flux.protocols import Sequence
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
        """
        Asynchronous wrapper for `solve`.

        Prevents blocking the event loop in high-throughput environments (e.g., FastAPI, 
        WebSockets) while waiting for the native executable. 

        Args:
            t_span (tuple): A tuple of (start_time, end_time) in seconds.
            protocol (Any, optional): Defines dynamic algebraic boundary constraints.
            parameters (Dict[str, float], optional): Dictionary of parameter overrides.
            t_eval (np.ndarray, optional): Specific time points to record the solution.
            scheduler (Any, optional): A `MultiTenantScheduler` or asyncio Semaphore to limit 
                concurrent native solver invocations and prevent host OOM conditions.

        Returns:
            SimulationResult: A data object containing the multidimensional trajectory arrays.
        """
        if scheduler:
            async with scheduler: 
                return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)
        return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)

    # --- Internal Sequence Orchestration ---

    def _orchestrate_sequence(self, protocol, parameters, requires_grad, show_progress) -> SimulationResult:
        """
        Drives piece-wise protocols natively. Preserves Python control flow specifically 
        to execute exact algebraic Bisection Root triggers across Sequence boundaries, allowing 
        events like 'voltage == 4.2V' to be met exactly.
        """
        session = self.start_session(parameters)
        data_hist = {"Time [s]": []}
        for k in self.manifest.layout.state_offsets.keys(): data_hist[k] = []
        for k in self.manifest.layout.obs_offsets.keys(): data_hist[k] = []
        raw_y_hist, raw_p_hist = [], []

        if requires_grad:
            self._initialize_ad_history(session)

        for step in protocol.steps:
            target_condition = getattr(step, "until", None)
            step_name = type(step).__name__
            inputs = self._map_protocol_inputs(step, step_name)
            
            t_max = getattr(step, "time", float('inf'))
            t_elapsed = 0.0
            
            while t_elapsed < t_max:
                # FIX: Clamp the evaluation step to exactly the remaining time limit
                dt_step = min(1.0, t_max - t_elapsed)
                
                session.checkpoint()
                session.step(dt_step, inputs=inputs)
                
                if target_condition and session.triggered(target_condition):
                    low = self._find_trigger_root(session, target_condition, inputs, dt_step)
                    t_elapsed += low
                    self._append_to_hist(session, data_hist, raw_y_hist, raw_p_hist, requires_grad)
                    break
                
                t_elapsed += dt_step
                self._append_to_hist(session, data_hist, raw_y_hist, raw_p_hist, requires_grad)

                if show_progress:
                    self._print_progress(session, step_name, t_elapsed, t_max)

            if show_progress:
                self._print_progress(session, step_name, t_elapsed, t_max, is_final=True)

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

    def _map_protocol_inputs(self, step: Any, step_name: str) -> Dict[str, float]:
        """Maps sequence parameters to algebraic terminal constraints."""
        inputs = {}
        if step_name == "CC":
            if "_term_mode" in self.parameters: inputs.update({"_term_mode": 1.0, "_term_i_target": step.rate})
            else: 
                if "mode" in self.parameters: inputs["mode"] = 1.0
                if "i_target" in self.parameters: inputs["i_target"] = step.rate
                elif "i_app" in self.parameters: inputs["i_app"] = step.rate
        elif step_name == "CV":
            if "_term_mode" in self.parameters: inputs.update({"_term_mode": 0.0, "_term_v_target": step.voltage})
            else: 
                if "mode" in self.parameters: inputs["mode"] = 0.0
                if "v_target" in self.parameters: inputs["v_target"] = step.voltage
        elif step_name == "Rest":
            if "_term_mode" in self.parameters: inputs.update({"_term_mode": 1.0, "_term_i_target": 0.0})
            else: 
                if "mode" in self.parameters: inputs["mode"] = 1.0
                if "i_target" in self.parameters: inputs["i_target"] = 0.0
                elif "i_app" in self.parameters: inputs["i_app"] = 0.0
        return inputs

    def _find_trigger_root(self, session: Session, target_condition: Any, inputs: Dict[str, float], dt_step: float) -> float:
        """Executes a dense bisection to land mathematically exactly on discontinuous sequence triggers."""
        session.restore()
        
        # Prevent speculative bisection steps from writing false branches to the AD tape
        with session.suspend_history():
            low, high = 0.0, dt_step
            for _ in range(15):
                mid = (low + high) / 2.0
                session.step(mid, inputs=inputs)
                if session.triggered(target_condition): high = mid
                else: low = mid
                session.restore()
                
        # Tape automatically resumes after context exit
        session.step(low, inputs=inputs)
        return low

    def _initialize_ad_history(self, session: Session) -> None:
        """Pre-allocates strictly dimensioned arrays to prevent np.vstack shape errors."""
        session.record_history = True
        y0 = session.handle.get_state() if session.handle else session._mock_y
        ydot0 = np.zeros(self.manifest.layout.n_states)
        p0 = self.manifest.pack_parameters(session.parameters)
        
        session.micro_t = [np.array([0.0])]
        session.micro_y = [y0[np.newaxis, :]]
        session.micro_ydot = [ydot0[np.newaxis, :]]
        session.micro_p = [np.array(p0)[np.newaxis, :]]

    def _append_to_hist(self, session: Session, data_hist: Dict, raw_y_hist: List, raw_p_hist: List, requires_grad: bool) -> None:
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

    def _print_progress(self, session: Session, step_name: str, t_elapsed: float, t_max: float, is_final: bool = False) -> None:
        try: v_str = f" | V: {session.get('V_cell'):.3f}V"
        except KeyError: v_str = ""
        
        if is_final:
            sys.stdout.write(f"\r▶ {step_name:<4} [██████████████████████████████] 100.0% | t: {session.time:.1f}s{v_str}   \n")
        elif t_max == float('inf'): 
            sys.stdout.write(f"\r▶ {step_name:<4} ⏳ t: {session.time:.1f}s{v_str}   ")
        else:
            pct = min(t_elapsed / t_max, 1.0)
            filled = int(pct * 30)
            bar = "█" * filled + "-" * (30 - filled)
            sys.stdout.write(f"\r▶ {step_name:<4} [{bar}] {pct*100:.1f}% | t: {session.time:.1f}s{v_str}   ")
            
        sys.stdout.flush()