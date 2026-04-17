import asyncio
import logging
import shutil
import json
import os
import sys
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Sequence as TypingSequence
import numpy as np

from ion_flux.dsl.core import PDE, State, Parameter, Observable
from ion_flux.compiler.memory import MemoryLayout
from ion_flux.compiler.codegen import generate_cpp, extract_state_name
from ion_flux.compiler.invocation import NativeCompiler, NativeRuntime
from ion_flux.compiler.passes.verification import verify_manifold, TopologicalError
from ion_flux.runtime.session import Session

try:
    from ion_flux._core import solve_ida_native, solve_ida_sundials, solve_batch_native
    RUST_FFI_AVAILABLE = True
    FFI_IMPORT_ERROR = None
except ImportError as e:
    RUST_FFI_AVAILABLE = False
    FFI_IMPORT_ERROR = str(e)
    logging.warning(f"Rust native solver failed to load: {e}. Operating in mock execution mode.")


from .results import Variable, SimulationResult
from .telemetry import TelemetryReport

class _ParamHandle:
    """Interface to physical parameters for default values tracking."""
    __slots__ = ["name", "value"]
    def __init__(self, name: str, default: float):
        self.name = name
        self.value = default
    def __repr__(self) -> str: return f"Parameter({self.name}={self.value})"



class Engine:
    """The central orchestrator for compilation, execution routing, and autodiff graphs."""
    def __init__(self, model: PDE, target: str = "cpu", solver_backend: str = "native", cache: bool = True, mock_execution: bool = False, jacobian_bandwidth: Optional[int] = None, debug: bool = False, **kwargs):
        self.model = model
        self.target = target
        self.solver_backend = solver_backend.lower()
        self.mock_execution = mock_execution
        self.debug = debug
        
        states = model.components(State) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, State)]
        params = model.components(Parameter) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, Parameter)]
        observables = model.components(Observable) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, Observable)]
        
        self.layout = MemoryLayout(states, params, observables)
        self.parameters = {p.name: _ParamHandle(p.name, p.default) for p in params}
        self.ast_payload: Dict[str, Any] = model.ast() if hasattr(model, "ast") else {}
        
        if self.ast_payload:
            # 0. Static Manifold Verification (Halt immediately on bad topology)
            verify_manifold(self.ast_payload)
            
            # 1. Validate System Rank (No Unconstrained States)
            # Explicit Equation Targeting guarantees every eq directly maps to a state key.
            targeted_states = {eq["state"] for eq in self.ast_payload.get("equations", [])}
            for state_name in self.layout.state_offsets.keys():
                if state_name not in targeted_states:
                    raise ValueError(f"Unconstrained state detected: '{state_name}'. Rank deficiency in system.")

        if jacobian_bandwidth is None:
            self.jacobian_bandwidth = self._compute_symbolic_bandwidth(states, self.ast_payload)
        else:
            self.jacobian_bandwidth = jacobian_bandwidth
        
        if hasattr(model, "ast"):
            self.cpp_source = generate_cpp(self.ast_payload, self.layout, states, observables, bandwidth=self.jacobian_bandwidth, target=self.target)
            self.runtime = None
            if not self.mock_execution:
                try:
                    compiler = NativeCompiler() if cache else NativeCompiler(cache_dir=os.path.join(tempfile.gettempdir(), "nocache"))
                    self.runtime = compiler.compile(self.cpp_source, self.layout.n_states)
                except RuntimeError as e:
                    logging.warning(f"Compilation failed, falling back to mock execution: {e}")
                    self.mock_execution = True
        else:
            self.runtime = None
            
        for k, v in kwargs.items(): setattr(self, k, v)

    def _compute_symbolic_bandwidth(self, states, ast_payload) -> int:
        from ion_flux.compiler.codegen.ast_analysis import extract_state_names
        
        if any(getattr(s.domain, "coord_sys", "") == "unstructured" for s in states):
            return -1
            
        max_bw = 0
        
        def check_dependencies(target_state: str, node: Dict[str, Any]) -> int:
            nonlocal max_bw
            if target_state not in self.layout.state_offsets: return max_bw
            off_t, size_t = self.layout.state_offsets[target_state]
            
            if size_t > 1: max_bw = max(max_bw, 2)
                
            deps = extract_state_names(node)
            for d in deps:
                if d not in self.layout.state_offsets: continue
                off_d, _ = self.layout.state_offsets[d]
                # Off-diagonal coupling requires dense/GMRES factorization
                if abs(off_t - off_d) > 0:
                    return 0 
            return max_bw

        # Moving domains dictate the spatial stride `dx` of the entire mesh.
        # Any state evaluated over this mesh inherently depends on the boundary state,
        # creating a dense off-diagonal dependency column.
        for bc_data in ast_payload.get("boundaries", []):
            if bc_data.get("type") == "moving_domain":
                return 0

        # Trace dependencies for bulk differential/algebraic equations
        for eq_data in ast_payload.get("equations", []):
            target_state = eq_data["state"]
            if eq_data["type"] == "piecewise":
                for reg in eq_data["regions"]:
                    if check_dependencies(target_state, reg["eq"]) == 0: return 0
            else:
                if check_dependencies(target_state, eq_data["eq"]) == 0: return 0
                
        return max_bw if max_bw > 0 else 0

    @classmethod
    def load(cls, binary_path: str, target: str = "cpu:serial", solver_backend: str = "native", debug: bool = False) -> "Engine":
        meta_path = binary_path + ".meta.json"
        if not os.path.exists(meta_path): raise FileNotFoundError(f"Missing layout manifest at {meta_path}.")
            
        with open(meta_path, "r") as f: meta = json.load(f)
            
        engine = cls.__new__(cls)
        engine.target = target
        engine.solver_backend = solver_backend
        engine.debug = debug
        engine.mock_execution = False
        engine.layout = MemoryLayout.from_dict(meta["layout"])
        engine.parameters = {name: _ParamHandle(name, val) for name, val in meta["parameters"].items()}
        engine.jacobian_bandwidth = meta.get("jacobian_bandwidth", 0)
        engine._metadata_cache = (
            meta["metadata_cache"]["y0"],
            meta["metadata_cache"]["ydot0"],
            meta["metadata_cache"]["id_arr"],
            meta["metadata_cache"].get("spatial_diag", [0.0] * engine.layout.n_states),
            meta["metadata_cache"].get("max_steps", [0.0] * engine.layout.n_states)
        )
        engine.runtime = NativeRuntime(binary_path, engine.layout.n_states)
        return engine

    def export_binary(self, export_path: str) -> None:
        if not getattr(self, "runtime", None) or not hasattr(self.runtime, "lib_path"):
            raise RuntimeError("Engine has not compiled a native binary. Cannot export.")
            
        y0, ydot0, id_arr, spatial_diag, max_steps = self._extract_metadata()
        meta = {
            "layout": {
                "state_offsets": self.layout.state_offsets,
                "param_offsets": self.layout.param_offsets,
                "obs_offsets": self.layout.obs_offsets,
                "n_states": self.layout.n_states,
                "n_params": self.layout.n_params,
                "n_obs": self.layout.n_obs,
                "p_length": self.layout.p_length,
                "m_length": self.layout.m_length,
                "mesh_offsets": self.layout.mesh_offsets,
                "mesh_cache": self.layout.mesh_cache
            },
            "parameters": {name: p.value for name, p in self.parameters.items()},
            "jacobian_bandwidth": getattr(self, "jacobian_bandwidth", 0),
            "metadata_cache": {"y0": y0, "ydot0": ydot0, "id_arr": id_arr, "spatial_diag": spatial_diag, "max_steps": max_steps}
        }
        with open(export_path + ".meta.json", "w") as f:
            json.dump(meta, f)
            
        shutil.copy(self.runtime.lib_path, export_path)

    @property
    def telemetry(self) -> TelemetryReport:
        """Returns memory layout telemetry and cache-hit heuristics."""
        return TelemetryReport(self.layout.n_states, getattr(self, "jacobian_bandwidth", 0))

    def start_session(self, parameters: Optional[Dict[str, float]] = None, soc: Optional[float] = None) -> Session:
        """Initializes a stateful memory session for HIL/SIL control loops."""
        return Session(engine=self, parameters=parameters or {}, soc=soc, debug=self.debug)

    def _extract_metadata(self) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Parses the compiled AST payload to construct the initial state (y0), 
        initial derivatives (ydot0), and the Differential/Algebraic mask (id_arr).
        """
        if hasattr(self, "_metadata_cache"):
            return self._metadata_cache
            
        y0 = [0.0] * self.layout.n_states
        ydot0 = [0.0] * self.layout.n_states
        id_arr = [0.0] * self.layout.n_states
        
        # We pass an array of zeros to satisfy the Rust C-ABI FFI signature.
        # The Rust native solver handles linear algebra scaling and preconditioner 
        # diagonal shifts natively via Mass Matrices and Row Equilibration.
        spatial_diag = [0.0] * self.layout.n_states
        
        # ---------------------------------------------------------------------
        # 1. Extract the Differential/Algebraic Mask (id_arr)
        # Recursively scans explicit equations for fx.dt() nodes. 
        # 1.0 = ODE/PDE. 0.0 = Pure Algebraic DAE.
        # ---------------------------------------------------------------------
        def _mark_differentials(node: Dict[str, Any], start: int, end: int) -> None:
            if isinstance(node, dict):
                if node.get("type") == "UnaryOp" and node.get("op") == "dt":
                    for i in range(start, end): 
                        id_arr[i] = 1.0
                for v in node.values(): 
                    _mark_differentials(v, start, end)
            elif isinstance(node, list):
                for item in node:
                    _mark_differentials(item, start, end)

        for eq_data in self.ast_payload.get("equations", []):
            offset, size = self.layout.state_offsets[eq_data["state"]]
            if eq_data["type"] == "piecewise":
                for reg in eq_data["regions"]:
                    _mark_differentials(reg["eq"], offset + reg["start_idx"], offset + reg["end_idx"])
            else:
                _mark_differentials(eq_data["eq"], offset, offset + size)
                
        # Dirichlet boundaries mathematically override the bulk PDE at the boundary 
        # nodes, turning them into pure algebraic constraints (0.0).
        for bc_data in self.ast_payload.get("boundaries", []):
            if bc_data["type"] == "dirichlet":
                offset, size = self.layout.state_offsets[bc_data["state"]]
                if "left" in bc_data["bcs"]: 
                    id_arr[offset] = 0.0
                if "right" in bc_data["bcs"]: 
                    id_arr[offset + size - 1] = 0.0
                
        # ---------------------------------------------------------------------
        # 2. Evaluate Initial Conditions (y0)
        # Recursively evaluates the static scalar AST expressions at t=0.
        # ---------------------------------------------------------------------
        def _eval_ic(node: Dict[str, Any], idx: int, dx: float) -> float:
            import math
            t = node.get("type")
            if t == "Scalar": return float(node["value"])
            if t == "Parameter":
                p_name = node.get("name")
                return self.parameters[p_name].value if p_name in self.parameters else 0.0
            if t == "BinaryOp":
                l = _eval_ic(node["left"], idx, dx)
                r = _eval_ic(node["right"], idx, dx)
                op = node["op"]
                if op == "add": return l + r
                if op == "sub": return l - r
                if op == "mul": return l * r
                if op == "div": return l / r if r != 0 else 0.0
                if op == "pow": return l ** r
                if op == "max": return max(l, r)
                if op == "min": return min(l, r)
            if t == "UnaryOp":
                c = _eval_ic(node["child"], idx, dx)
                op = node["op"]
                if op == "neg": return -c
                if op == "coords": return idx * dx
                if op == "sin": return math.sin(c)
                if op == "cos": return math.cos(c)
                if op == "exp": return math.exp(c)
                if op == "log": return math.log(c) if c > 0 else 0.0
                if op == "abs": return abs(c)
            return 0.0
        
        for ic_data in self.ast_payload.get("initial_conditions", []):
            state_name = ic_data["state"]
            offset, size = self.layout.state_offsets[state_name]
            
            dx = 1.0
            state_node = next((s for s in self.model.__dict__.values() if getattr(s, "name", "") == state_name), None)
            if state_node and getattr(state_node, "domain", None):
                ds = state_node.domain.domains if hasattr(state_node.domain, "domains") else [state_node.domain]
                dx = float(ds[0].bounds[1] - ds[0].bounds[0]) / max(1, ds[0].resolution - 1)
                
            for i in range(size):
                y0[offset + i] = _eval_ic(ic_data["value"], i, dx)

        # ---------------------------------------------------------------------
        # 3. Extract Newton Step Clamps (max_steps)
        # ---------------------------------------------------------------------
        max_steps = [0.0] * self.layout.n_states
        for state_name, (offset, size) in self.layout.state_offsets.items():
            state_node = next((s for s in self.model.components(State) if getattr(s, "name", "") == state_name), None)
            if not state_node:
                state_node = next((s for s in self.model.__dict__.values() if getattr(s, "name", "") == state_name), None)
            
            if state_node and getattr(state_node, "max_newton_step", None) is not None:
                val = float(state_node.max_newton_step)
                for i in range(size):
                    max_steps[offset + i] = val
                    
        self._metadata_cache = (y0, ydot0, id_arr, spatial_diag, max_steps)
        return y0, ydot0, id_arr, spatial_diag, max_steps

    def _pack_parameters(self, overrides: Dict[str, float]) -> List[float]:
        p_list = [0.0] * self.layout.p_length
        for p_name, (offset, _) in self.layout.param_offsets.items():
            p_list[offset] = overrides.get(p_name, self.parameters[p_name].value)
        return p_list

    def _handle_native_crash(self, original_error: Exception):
        """Intercepts Rust native panics to parse the JSON diagnostic report, translating flat FFI 
        indices back into human-readable Python AST state variables."""
        import glob
        import os
        import json
        crash_files = glob.glob("ion_flux_diagnostics/crash_*.json")
        if not crash_files: raise original_error
            
        latest_crash = max(crash_files, key=os.path.getctime)
        try:
            with open(latest_crash, "r") as f: crash_data = json.load(f)
            
            idx_to_name = {}
            for name, (offset, size) in self.layout.state_offsets.items():
                for i in range(size):
                    idx_to_name[offset + i] = f"{name}[{i}]" if size > 1 else name
            
            # Inject names back into the JSON payload for better local debugging
            for off in crash_data.get("top_offenders", []):
                off["name"] = idx_to_name.get(off.get("index", -1), f"Unknown[{off.get('index', -1)}]")
                
            if "initialization_health" in crash_data:
                idx = crash_data["initialization_health"].get("t0_max_residual_index", -1)
                crash_data["initialization_health"]["t0_max_residual_name"] = idx_to_name.get(idx, f"Unknown[{idx}]")
            
            # Save the enriched JSON so users have context directly in the file
            with open(latest_crash, "w") as f:
                json.dump(crash_data, f, indent=2)
                    
            msg = f"\n{'-'*100}\n"
            msg += f"🔥 NATIVE SOLVER CRASH: {str(original_error)}\n"
            msg += f"{'-'*100}\n"
            msg += f"Reason: {crash_data.get('reason', 'Unknown')}\n"
            msg += f"Accepted Steps: {crash_data.get('accepted_steps', 0)}\n"
            
            init_health = crash_data.get("initialization_health", {})
            if init_health.get("t0_max_residual", 0.0) > 1e3:
                msg += f"\n⚠️ INITIALIZATION WARNING: Massive residual at t=0 detected!\n"
                msg += f"   Variable: {init_health.get('t0_max_residual_name')} (Residual: {init_health.get('t0_max_residual'):.3e})\n"
                msg += f"   Check your `initial_conditions` for severe algebraic imbalances.\n"
                
            jac_health = crash_data.get("jacobian_health", {})
            if jac_health.get("condition_warning", False):
                msg += f"\n⚠️ JACOBIAN CONDITION WARNING: Matrix is likely singular or badly scaled.\n"
                msg += f"   Max element: {jac_health.get('max_element'):.3e}, Min non-zero: {jac_health.get('min_nonzero_element'):.3e}\n"
                
            trace = crash_data.get("newton_thrashing_trace", [])
            if trace:
                msg += f"\nNewton Trace (Last {len(trace)} iterations):\n"
                for t in trace:
                    msg += f"   Iter {t.get('iter')}: Residual Norm = {t.get('residual_norm'):.3e}, Step Norm = {t.get('step_norm'):.3e}\n"
            
            msg += f"\nTop Offenders (Ranked by NaN presence, then Absolute Residual):\n"
            msg += f"{'State Name':<25} | {'Type':<9} | {'Residual':<10} | {'Weight':<9} | {'Step dy':<10} | {'y_val':<10}\n"
            msg += "-" * 100 + "\n"
            
            for off in crash_data.get("top_offenders", []):
                name = off.get("name", "")
                rtype = off.get("type", "")
                
                def fmt(v):
                    try: return f"{float(v):<10.3e}"
                    except: return f"{v:<10}"
                    
                res = fmt(off.get("residual", 0.0))
                yv = fmt(off.get("y_val", 0.0))
                dy = fmt(off.get("proposed_step_dy", 0.0))
                w = fmt(off.get("solver_weight", 0.0))
                
                msg += f"{name[:24]:<25} | {rtype:<9} | {res} | {w} | {dy} | {yv}\n"
                
            msg += f"{'-'*100}\n"
            raise RuntimeError(msg) from None
        except Exception:
            raise original_error from None

    def evaluate_residual(self, y: List[float], ydot: List[float], parameters: Optional[Dict[str, float]] = None) -> List[float]:
        """Provides direct Python wrapper to evaluate the analytical residual map F(y, ydot, p, m)."""
        if self.mock_execution or not self.runtime: raise RuntimeError("Requires native execution.")
        p_list = self._pack_parameters(parameters or {})
        m_list = self.layout.get_mesh_data()
        return self.runtime.evaluate_residual(y, ydot, p_list, m_list)

    def evaluate_observables(self, y: List[float], ydot: List[float], parameters: Optional[Dict[str, float]] = None) -> List[float]:
        if self.mock_execution or not self.runtime: raise RuntimeError("Requires native execution.")
        p_list = self._pack_parameters(parameters or {})
        m_list = self.layout.get_mesh_data()
        return self.runtime.evaluate_observables(y, ydot, p_list, m_list, self.layout.n_obs)

    def evaluate_jacobian(self, y: List[float], ydot: List[float], c_j: float, parameters: Optional[Dict[str, float]] = None) -> List[List[float]]:
        """Provides direct Python wrapper to evaluate the exact analytical Jacobian using Enzyme AD."""
        if self.mock_execution or not self.runtime: raise RuntimeError("Requires native execution.")
        p_list = self._pack_parameters(parameters or {})
        m_list = self.layout.get_mesh_data()
        return self.runtime.evaluate_jacobian(y, ydot, p_list, m_list, c_j)

    def solve(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, 
                t_eval: Optional[np.ndarray] = None, requires_grad: Optional[List[str]] = None, threads: int = 1, show_progress: bool = True) -> SimulationResult:
        
        if threads > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = str(threads)
            if getattr(self, "runtime", None):
                self.runtime.set_spatial_threads(threads)
            
        if self.mock_execution or not self.layout:
            return self._execute_mock(parameters, protocol)

        from ion_flux.protocols.profiles import Sequence
        
        if protocol and isinstance(protocol, Sequence):
            session = self.start_session(parameters)
            data_hist = {"Time [s]": []}
            for k in self.layout.state_offsets.keys(): data_hist[k] = []
            for k in self.layout.obs_offsets.keys(): data_hist[k] = []
            raw_y_hist = []
            raw_p_hist = []

            if requires_grad:
                session.record_history = True
                session.micro_t = [0.0]
                session.micro_y = [session.handle.get_state().tolist() if session.handle else session._mock_y.tolist()]
                session.micro_ydot = [np.zeros(self.layout.n_states).tolist()]
                session.micro_p = [self._pack_parameters(session.parameters)]

            for step in protocol.steps:
                target_condition = getattr(step, "until", None)
                inputs = {}
                step_name = type(step).__name__
                
                if step_name == "CC":
                    if "_term_mode" in self.parameters:
                        inputs["_term_mode"] = 1.0
                        inputs["_term_i_target"] = step.rate
                    else: 
                        if "mode" in self.parameters: inputs["mode"] = 1.0
                        if "i_target" in self.parameters: inputs["i_target"] = step.rate
                        elif "i_app" in self.parameters: inputs["i_app"] = step.rate
                elif step_name == "CV":
                    if "_term_mode" in self.parameters:
                        inputs["_term_mode"] = 0.0
                        inputs["_term_v_target"] = step.voltage
                    else: 
                        if "mode" in self.parameters: inputs["mode"] = 0.0
                        if "v_target" in self.parameters: inputs["v_target"] = step.voltage
                elif step_name == "Rest":
                    if "_term_mode" in self.parameters:
                        inputs["_term_mode"] = 1.0
                        inputs["_term_i_target"] = 0.0
                    else: 
                        if "mode" in self.parameters: inputs["mode"] = 1.0
                        if "i_target" in self.parameters: inputs["i_target"] = 0.0
                        elif "i_app" in self.parameters: inputs["i_app"] = 0.0
                
                dt_step = 0.5 if requires_grad else 1.0 
                t_max = getattr(step, "time", float('inf'))
                t_elapsed = 0.0
                
                while t_elapsed < t_max:
                    session.checkpoint()
                    session.step(dt_step, inputs=inputs)
                    
                    if target_condition and session.triggered(target_condition):
                        session.restore()
                        low, high = 0.0, dt_step
                        for _ in range(15):
                            mid = (low + high) / 2.0
                            session.step(mid, inputs=inputs)
                            if session.triggered(target_condition): high = mid
                            else: low = mid
                            session.restore()
                        
                        # Use the low bracket to strictly land safely before the trigger asymptote
                        session.step(low, inputs=inputs)
                        
                        t_elapsed += low
                        data_hist["Time [s]"].append(session.time)
                        y = session.handle.get_state() if session.handle else session._mock_y
                        obs = session.handle.get_observables_py() if session.handle else np.zeros(self.layout.n_obs)
                        raw_y_hist.append(y)
                        if requires_grad: raw_p_hist.append(self._pack_parameters(session.parameters))
                
                        for k, (offset, size) in self.layout.state_offsets.items():
                            data_hist[k].append(y[offset:offset+size] if size > 1 else y[offset])
                        for k, (offset, size) in self.layout.obs_offsets.items():
                            data_hist[k].append(obs[offset:offset+size] if size > 1 else obs[offset])
                        break
                    
                    t_elapsed += dt_step
                    data_hist["Time [s]"].append(session.time)
                    y = session.handle.get_state() if session.handle else session._mock_y
                    obs = session.handle.get_observables_py() if session.handle else np.zeros(self.layout.n_obs)
                    raw_y_hist.append(y)
                    if requires_grad: raw_p_hist.append(self._pack_parameters(session.parameters))
                    
                    for k, (offset, size) in self.layout.state_offsets.items():
                        data_hist[k].append(y[offset:offset+size] if size > 1 else y[offset])
                    for k, (offset, size) in self.layout.obs_offsets.items():
                        data_hist[k].append(obs[offset:offset+size] if size > 1 else obs[offset])

                    # --- PROGRESS BAR RENDERER ---
                    if show_progress:
                        try:
                            v_str = f" | V: {session.get('V_cell'):.3f}V"
                        except KeyError:
                            v_str = ""
                            
                        if t_max == float('inf'):
                            sys.stdout.write(f"\r▶ {step_name:<4} ⏳ t: {session.time:.1f}s{v_str}   ")
                        else:
                            pct = min(t_elapsed / t_max, 1.0)
                            filled = int(pct * 30)
                            bar = "█" * filled + "-" * (30 - filled)
                            sys.stdout.write(f"\r▶ {step_name:<4} [{bar}] {pct*100:.1f}% | t: {session.time:.1f}s{v_str}   ")
                        sys.stdout.flush()

                # Cap off the step with a finalized 100% bar and a clean newline
                if show_progress:
                    try:
                        v_str = f" | V: {session.get('V_cell'):.3f}V"
                    except KeyError:
                        v_str = ""
                    sys.stdout.write(f"\r▶ {step_name:<4} [██████████████████████████████] 100.0% | t: {session.time:.1f}s{v_str}   \n")
                    sys.stdout.flush()

            for k in data_hist: data_hist[k] = np.array(data_hist[k])
            
            trajectory = None
            if requires_grad:
                trajectory = {
                    "Time [s]": data_hist["Time [s]"], 
                    "_y_raw": np.array(raw_y_hist), 
                    "_micro_t": np.array(session.micro_t),
                    "_micro_y": np.array(session.micro_y),
                    "_micro_ydot": np.array(session.micro_ydot),
                    "_p_traj": session.micro_p,
                    "requires_grad": requires_grad
                }
            
            return SimulationResult(data_hist, session.parameters, engine=self, trajectory=trajectory)
            
        if not RUST_FFI_AVAILABLE: raise RuntimeError(f"Native solver missing. FFI Error: {FFI_IMPORT_ERROR}")
            
        y0, ydot0, id_arr, spatial_diag, max_steps = self._extract_metadata()
        p_list = self._pack_parameters(parameters or {})
        m_list = self.layout.get_mesh_data()
        
        t_eval_arr = t_eval if t_eval is not None else np.linspace(t_span[0], t_span[1], 100)
        record_history = requires_grad is not None
        
        v_idx = self.layout.state_offsets.get("V_cell", (-1, 0))[0]
        
        try:
            if self.solver_backend == "sundials":
                y_res, obs_res, micro_t, micro_y, micro_ydot = solve_ida_sundials(
                    self.runtime.lib_path, y0, ydot0, id_arr, p_list, m_list, t_eval_arr.tolist(), self.layout.n_obs, show_progress, v_idx
                )
            else:
                y_res, obs_res, micro_t, micro_y, micro_ydot = solve_ida_native(
                    self.runtime.lib_path, y0, ydot0, id_arr, p_list, m_list, t_eval_arr.tolist(), 
                    self.jacobian_bandwidth, spatial_diag, max_steps, self.layout.n_obs, record_history, self.debug, show_progress, v_idx
                )
        except RuntimeError as e:
            self._handle_native_crash(e)
        
        data = {"Time [s]": t_eval_arr}
        for state_name, (offset, size) in self.layout.state_offsets.items():
            if size == 1: data[state_name] = y_res[:, offset]
            else: data[state_name] = y_res[:, offset:offset+size]
        for obs_name, (offset, size) in self.layout.obs_offsets.items():
            if size == 1: data[obs_name] = obs_res[:, offset]
            else: data[obs_name] = obs_res[:, offset:offset+size]
            
        trajectory = None
        if requires_grad: 
            trajectory = {
                "Time [s]": t_eval_arr, 
                "_y_raw": y_res, 
                "_micro_t": micro_t,
                "_micro_y": micro_y,
                "_micro_ydot": micro_ydot,
                "_p_traj": [p_list]*len(micro_t), 
                "requires_grad": requires_grad
            }
        return SimulationResult(data, parameters or {}, status="completed", engine=self, trajectory=trajectory)

    def solve_batch(self, parameters: List[Dict[str, float]], t_span: tuple = (0, 1), 
                    protocols: Any = None, max_workers: int = 1, show_progress: bool = False) -> List[SimulationResult]:
        """
        Executes a massive payload of concurrent models using Rust's Rayon thread-pool,
        entirely bypassing the Python Global Interpreter Lock (GIL).

        Args:
            parameters: A list of dictionaries containing parameter overrides for each model.
            t_span: Default integration bounds if a protocol does not strictly dictate time.
            protocols: A single `Sequence` protocol (broadcasted to all models), or a list 
                       of `Sequence` protocols strictly matching the length of `parameters`.
            max_workers: The number of independent OS threads to spawn in Rust.
            show_progress: Whether to render a CLI progress bar for the batch execution.
            
        Returns:
            A list of `SimulationResult` objects mapping 1-to-1 with the `parameters` list.
        """
        from ion_flux.protocols.profiles import Sequence
        
        # 1. Broadcast or Validate Protocol Lengths
        if protocols:
            if isinstance(protocols, Sequence):
                protocols = [protocols] * len(parameters)
            elif len(protocols) != len(parameters):
                raise ValueError(
                    f"Batch length mismatch: {len(parameters)} parameter payloads provided, "
                    f"but {len(protocols)} protocols provided. Pass a single protocol to "
                    f"broadcast to all models, or an exact 1-to-1 list."
                )
                
        if max_workers > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = "1"
            if getattr(self, "runtime", None):
                self.runtime.set_spatial_threads(1)
            
        # 2. Fallback to Python Loop for Mock Execution
        if self.mock_execution or not RUST_FFI_AVAILABLE:
            if not protocols:
                protocols = [None] * len(parameters)
            return [self.solve(t_span=t_span, protocol=prot, parameters=p) for p, prot in zip(parameters, protocols)]

        y0, ydot0, id_arr, spatial_diag, max_steps = self._extract_metadata()
        t_eval_arr = np.linspace(t_span[0], t_span[1], 100)
        p_batch = [self._pack_parameters(p) for p in parameters]
        m_list = self.layout.get_mesh_data()
        
        # 3. Crush Object-Oriented Protocols into flat C-ABI Instructions
        protocol_payloads = None
        if protocols:
            protocol_payloads = []
            
            def _get_p_idx(keys):
                for k in keys:
                    if k in self.layout.param_offsets: return self.layout.param_offsets[k][0]
                return 0
                
            # Extract standard parameter indexes for multiplexer hot-swapping
            p_idx_mode = _get_p_idx(["_term_mode", "mode"])
            p_idx_i = _get_p_idx(["_term_i_target", "i_target", "i_app"])
            p_idx_v = _get_p_idx(["_term_v_target", "v_target"])
            
            for prot in protocols:
                payload = []
                for step in prot.steps:
                    step_type = 0 if type(step).__name__ == "CC" else (1 if type(step).__name__ == "CV" else 2)
                    target_val = getattr(step, "rate", getattr(step, "voltage", 0.0))
                    time_limit = getattr(step, "time", float('inf'))
                    
                    has_trig = False
                    trig_idx, trig_size, trig_is_obs, trig_op, trig_val = 0, 1, False, 0, 0.0
                    
                    cond = getattr(step, "until", None)
                    if cond:
                        has_trig = True
                        var_name, op_str, t_val = cond._compiled_logic
                        if var_name in self.layout.state_offsets:
                            trig_idx, trig_size = self.layout.state_offsets[var_name]
                        elif var_name in self.layout.obs_offsets:
                            trig_idx, trig_size = self.layout.obs_offsets[var_name]
                            trig_is_obs = True
                        else:
                            raise ValueError(f"Trigger variable '{var_name}' not found.")
                        
                        op_map = {">": 1, "<": 2, ">=": 3, "<=": 4, "==": 5, "!=": 6}
                        trig_op = op_map.get(op_str, 0)
                        trig_val = float(t_val)
                        
                    payload.append((
                        step_type, target_val, time_limit, 
                        (has_trig, trig_idx, trig_size, trig_is_obs, trig_op, trig_val), 
                        p_idx_mode, p_idx_i, p_idx_v
                    ))
                protocol_payloads.append(payload)
        
        v_idx = self.layout.state_offsets.get("V_cell", (-1, 0))[0]
        
        # 4. Native Rayon Execution
        try:
            y_res_batch = solve_batch_native(
                self.runtime.lib_path, y0, ydot0, id_arr, p_batch, m_list, 
                t_eval_arr.tolist(), self.jacobian_bandwidth, spatial_diag, max_steps, self.layout.n_obs, self.debug, 
                max_workers, show_progress, protocol_payloads, v_idx
            )
        except RuntimeError as e:
            self._handle_native_crash(e)
            
        # 5. Unpack Flat Arrays into SimulationResults
        results = []
        for p, (t_res, y_res, obs_res) in zip(parameters, y_res_batch):
            data = {"Time [s]": t_res}
            for state_name, (offset, size) in self.layout.state_offsets.items():
                if size == 1: data[state_name] = y_res[:, offset]
                else: data[state_name] = y_res[:, offset:offset+size]
            for obs_name, (offset, size) in self.layout.obs_offsets.items():
                if size == 1: data[obs_name] = obs_res[:, offset]
                else: data[obs_name] = obs_res[:, offset:offset+size]
            results.append(SimulationResult(data, p, status="completed", engine=self, trajectory=None))

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
        data = {"Time [s]": np.arange(time_len, dtype=np.float64)}
        
        if hasattr(self, "layout") and self.layout:
            for state_name, (offset, size) in self.layout.state_offsets.items():
                data[state_name] = np.zeros(time_len) if size == 1 else np.zeros((time_len, size))
            for obs_name, (offset, size) in self.layout.obs_offsets.items():
                data[obs_name] = np.zeros(time_len) if size == 1 else np.zeros((time_len, size))
                    
        data["Voltage [V]"] = np.array([4.2] * (time_len - 1) + [2.5])
        
        trajectory = {"Time [s]": data["Time [s]"], "_y_raw": np.zeros((time_len, getattr(self.layout, 'n_states', 1)))}
        return SimulationResult(data, params, status="completed", engine=self, trajectory=trajectory)