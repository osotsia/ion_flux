import asyncio
import logging
import shutil
import json
import os
import sys
import tempfile
import itertools
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
    __slots__ = ["name", "value"]
    def __init__(self, name: str, default: float):
        self.name = name
        self.value = default
    def __repr__(self) -> str: return f"Parameter({self.name}={self.value})"

class Engine:
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
            verify_manifold(self.ast_payload)
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
                compiler = NativeCompiler() if cache else NativeCompiler(cache_dir=os.path.join(tempfile.gettempdir(), "nocache"))
                self.runtime = compiler.compile(self.cpp_source, self.layout.n_states)
        else:
            self.runtime = None
            
        for k, v in kwargs.items(): setattr(self, k, v)

    def _compute_symbolic_bandwidth(self, states, ast_payload) -> int:
        from ion_flux.compiler.codegen.ast_analysis import extract_state_names
        if any(getattr(s.domain, "coord_sys", "") == "unstructured" for s in states): return -1
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
                if abs(off_t - off_d) > 0: return 0 
            return max_bw

        for bc_data in ast_payload.get("boundaries", []):
            if bc_data.get("type") == "moving_domain": return 0

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
            meta["metadata_cache"]["y0"], meta["metadata_cache"]["ydot0"], meta["metadata_cache"]["id_arr"],
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
                "state_offsets": self.layout.state_offsets, "param_offsets": self.layout.param_offsets, "obs_offsets": self.layout.obs_offsets,
                "n_states": self.layout.n_states, "n_params": self.layout.n_params, "n_obs": self.layout.n_obs,
                "p_length": self.layout.p_length, "m_length": self.layout.m_length, "mesh_offsets": self.layout.mesh_offsets, "mesh_cache": self.layout.mesh_cache
            },
            "parameters": {name: p.value for name, p in self.parameters.items()}, "jacobian_bandwidth": getattr(self, "jacobian_bandwidth", 0),
            "metadata_cache": {"y0": y0, "ydot0": ydot0, "id_arr": id_arr, "spatial_diag": spatial_diag, "max_steps": max_steps}
        }
        with open(export_path + ".meta.json", "w") as f: json.dump(meta, f)
        shutil.copy(self.runtime.lib_path, export_path)

    @property
    def telemetry(self) -> TelemetryReport: return TelemetryReport(self.layout.n_states, getattr(self, "jacobian_bandwidth", 0))

    def start_session(self, parameters: Optional[Dict[str, float]] = None, soc: Optional[float] = None) -> Session:
        return Session(engine=self, parameters=parameters or {}, soc=soc, debug=self.debug)

    def _extract_metadata(self) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        if hasattr(self, "_metadata_cache"): return self._metadata_cache
            
        y0 = [0.0] * self.layout.n_states
        ydot0 = [0.0] * self.layout.n_states
        id_arr = [0.0] * self.layout.n_states
        spatial_diag = [0.0] * self.layout.n_states
        
        from ion_flux.compiler.codegen.topology import TopologyAnalyzer
        topo = TopologyAnalyzer(self.ast_payload.get("domains", {}))
        
        all_states = self.model.components(State) if hasattr(self.model, "components") else [attr for attr in self.model.__dict__.values() if isinstance(attr, State)]
        state_dict = {getattr(s, "name", ""): s for s in all_states}

        def _mark_differentials(node: Dict[str, Any], start: int, end: int) -> None:
            if isinstance(node, dict):
                if node.get("type") == "UnaryOp" and node.get("op") == "dt":
                    for i in range(start, end): id_arr[i] = 1.0
                for v in node.values(): _mark_differentials(v, start, end)
            elif isinstance(node, list):
                for item in node: _mark_differentials(item, start, end)

        def _check_dt(node: Dict[str, Any]) -> bool:
            if isinstance(node, dict):
                if node.get("type") == "UnaryOp" and node.get("op") == "dt": return True
                for v in node.values():
                    if _check_dt(v): return True
            elif isinstance(node, list):
                for item in node:
                    if _check_dt(item): return True
            return False

        for eq_data in self.ast_payload.get("equations", []):
            state_name = eq_data["state"]
            offset, size = self.layout.state_offsets[state_name]
            
            if eq_data["type"] == "piecewise":
                state_node = state_dict.get(state_name)
                d_name = state_node.domain.name if state_node and getattr(state_node, "domain", None) else None
                
                for reg in eq_data["regions"]:
                    if not _check_dt(reg["eq"]): continue
                    if not d_name:
                        id_arr[offset] = 1.0
                        continue
                        
                    axes = topo.get_axes(d_name)
                    strides = topo.get_strides(d_name)
                    b_axis = topo.get_base_axis(reg["domain"])
                    
                    ranges = []
                    for axis in axes:
                        base = topo.get_base_axis(axis)
                        if base == b_axis: ranges.append(range(reg["start_idx"], reg["end_idx"]))
                        else: ranges.append(range(topo.domains.get(axis, {}).get("resolution", 1)))
                    
                    for indices in itertools.product(*ranges):
                        flat_idx = 0
                        for axis, idx in zip(axes, indices): flat_idx += idx * strides[axis]
                        id_arr[offset + flat_idx] = 1.0
            else:
                _mark_differentials(eq_data["eq"], offset, offset + size)
                
        for bc_data in self.ast_payload.get("boundaries", []):
            if bc_data["type"] == "dirichlet":
                state_name = bc_data["state"]
                offset, size = self.layout.state_offsets[state_name]
                
                state_node = state_dict.get(state_name)
                d_name = state_node.domain.name if state_node and getattr(state_node, "domain", None) else None
                
                if not d_name:
                    if "left" in bc_data["bcs"]: id_arr[offset] = 0.0
                    if "right" in bc_data["bcs"]: id_arr[offset + size - 1] = 0.0
                    continue
                    
                axes = topo.get_axes(d_name)
                strides = topo.get_strides(d_name)
                b_axis = axes[-1]
                coord_sys = topo.domains.get(b_axis, {}).get("coord_sys", "cartesian")
                
                if coord_sys == "unstructured":
                    surfaces = self.layout.mesh_offsets.get(b_axis, {}).get("surfaces", {})
                    for side in bc_data["bcs"]:
                        if side in surfaces:
                            mask_off = surfaces[side]
                            for i in range(size):
                                if self.layout.mesh_cache.get(mask_off + i, 0.0) > 0.5:
                                    id_arr[offset + i] = 0.0
                    continue
                
                b_res = topo.domains.get(b_axis, {}).get("resolution", 1)
                
                ranges = []
                for axis in axes:
                    if axis == b_axis: ranges.append([0]) 
                    else: ranges.append(range(topo.domains.get(axis, {}).get("resolution", 1)))
                        
                for indices in itertools.product(*ranges):
                    base_flat = 0
                    for axis, idx in zip(axes, indices): base_flat += idx * strides[axis]
                    if "left" in bc_data["bcs"]: id_arr[offset + base_flat] = 0.0
                    if "right" in bc_data["bcs"]: id_arr[offset + base_flat + (b_res - 1) * strides[b_axis]] = 0.0

        def _eval_ic(node: Dict[str, Any], flat_idx: int, d_name: str) -> float:
            import math
            t = node.get("type")
            if t == "Scalar": return float(node["value"])
            if t == "Parameter":
                p_name = node.get("name")
                return self.parameters[p_name].value if p_name in self.parameters else 0.0
            if t == "BinaryOp":
                l = _eval_ic(node["left"], flat_idx, d_name)
                r = _eval_ic(node["right"], flat_idx, d_name)
                op = node["op"]
                if op == "add": return l + r
                if op == "sub": return l - r
                if op == "mul": return l * r
                if op == "div": return l / r if r != 0 else 0.0
                if op == "pow": return l ** r
                if op == "max": return max(l, r)
                if op == "min": return min(l, r)
            if t == "UnaryOp":
                c = _eval_ic(node["child"], flat_idx, d_name)
                op = node["op"]
                if op == "neg": return -c
                if op == "coords":
                    b_axis = node.get("axis")
                    if b_axis and d_name:
                        axes = topo.get_axes(d_name)
                        strides = topo.get_strides(d_name)
                        if b_axis in axes:
                            stride = strides[b_axis]
                            res = topo.domains.get(b_axis, {}).get("resolution", 1)
                            local_idx = (flat_idx // stride) % res
                            bounds = topo.domains.get(b_axis, {}).get("bounds", (0, 1))
                            dx = float(bounds[1] - bounds[0]) / max(res - 1, 1)
                            return local_idx * dx
                    return 0.0
                if op == "sin": return math.sin(c)
                if op == "cos": return math.cos(c)
                if op == "exp": return math.exp(c)
                if op == "log": return math.log(c) if c > 0 else 0.0
                if op == "sqrt": return math.sqrt(c) if c > 0 else 0.0
                if op == "abs": return abs(c)
            return 0.0
        
        for ic_data in self.ast_payload.get("initial_conditions", []):
            state_name = ic_data["state"]
            offset, size = self.layout.state_offsets[state_name]
            state_node = state_dict.get(state_name)
            d_name = state_node.domain.name if state_node and getattr(state_node, "domain", None) else ""
                
            for i in range(size):
                y0[offset + i] = _eval_ic(ic_data["value"], i, d_name)

        max_steps = [0.0] * self.layout.n_states
        for state_name, (offset, size) in self.layout.state_offsets.items():
            state_node = state_dict.get(state_name)
            if state_node and getattr(state_node, "max_newton_step", None) is not None:
                val = float(state_node.max_newton_step)
                for i in range(size): max_steps[offset + i] = val
                    
        self._metadata_cache = (y0, ydot0, id_arr, spatial_diag, max_steps)
        return y0, ydot0, id_arr, spatial_diag, max_steps

    def _pack_parameters(self, overrides: Dict[str, float]) -> List[float]:
        p_list = [0.0] * self.layout.p_length
        for p_name, (offset, _) in self.layout.param_offsets.items():
            p_list[offset] = overrides.get(p_name, self.parameters[p_name].value)
        return p_list

    def _handle_native_crash(self, original_error: Exception):
        import glob
        crash_files = glob.glob("ion_flux_diagnostics/crash_*.json")
        if not crash_files: raise original_error
            
        latest_crash = max(crash_files, key=os.path.getctime)
        try:
            with open(latest_crash, "r") as f: crash_data = json.load(f)
            
            idx_to_name = {}
            for name, (offset, size) in self.layout.state_offsets.items():
                for i in range(size): idx_to_name[offset + i] = f"{name}[{i}]" if size > 1 else name
            
            for off in crash_data.get("top_offenders", []):
                off["name"] = idx_to_name.get(off.get("index", -1), f"Unknown[{off.get('index', -1)}]")
                
            if "initialization_health" in crash_data:
                idx = crash_data["initialization_health"].get("t0_max_residual_index", -1)
                crash_data["initialization_health"]["t0_max_residual_name"] = idx_to_name.get(idx, f"Unknown[{idx}]")
            
            with open(latest_crash, "w") as f: json.dump(crash_data, f, indent=2)
                    
            msg = f"\n{'-'*100}\n🔥 NATIVE SOLVER CRASH: {str(original_error)}\n{'-'*100}\n"
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
                
            trace = crash_data.get("newton_thrashing_trace", [])
            if trace:
                msg += f"\nNewton Trace (Last {len(trace)} iterations):\n"
                for t in trace: msg += f"   Iter {t.get('iter')}: Residual Norm = {t.get('residual_norm'):.3e}, Step Norm = {t.get('step_norm'):.3e}\n"
            
            msg += f"\nTop Offenders (Ranked by NaN presence, then Absolute Residual):\n"
            msg += f"{'State Name':<25} | {'Type':<9} | {'Residual':<10} | {'Weight':<9} | {'Step dy':<10} | {'y_val':<10}\n"
            msg += "-" * 100 + "\n"
            
            for off in crash_data.get("top_offenders", []):
                name, rtype = off.get("name", ""), off.get("type", "")
                def fmt(v): return f"{float(v):<10.3e}" if isinstance(v, (float, int)) else f"{v:<10}"
                msg += f"{name[:24]:<25} | {rtype:<9} | {fmt(off.get('residual', 0.0))} | {fmt(off.get('solver_weight', 0.0))} | {fmt(off.get('proposed_step_dy', 0.0))} | {fmt(off.get('y_val', 0.0))}\n"
                
            msg += f"{'-'*100}\n"
            raise RuntimeError(msg) from None
        except Exception:
            raise original_error from None

    def evaluate_residual(self, y: List[float], ydot: List[float], parameters: Optional[Dict[str, float]] = None) -> List[float]:
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
        """Dynamically proxies the C++ analytical Sparse representation back to dense Python matrices for Oracle tests."""
        if self.mock_execution or not self.runtime: raise RuntimeError("Requires native execution.")
        import ctypes
        
        p_list = self._pack_parameters(parameters or {})
        m_list = self.layout.get_mesh_data()
        
        N = self.layout.n_states
        rows = (ctypes.c_int * (N * 50))()
        cols = (ctypes.c_int * (N * 50))()
        vals = (ctypes.c_double * (N * 50))()
        nnz = ctypes.c_int(0)
        
        y_arr = (ctypes.c_double * N)(*y)
        ydot_arr = (ctypes.c_double * N)(*ydot)
        p_arr = (ctypes.c_double * len(p_list))(*p_list)
        m_arr = (ctypes.c_double * len(m_list))(*m_list)
        
        self.runtime.dll.evaluate_jacobian_sparse(y_arr, ydot_arr, p_arr, m_arr, ctypes.c_double(c_j), rows, cols, vals, ctypes.byref(nnz))
        
        jac_2d = [[0.0] * N for _ in range(N)]
        for i in range(nnz.value):
            jac_2d[rows[i]][cols[i]] = vals[i]
            
        return jac_2d

    def solve(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, 
                t_eval: Optional[np.ndarray] = None, requires_grad: Optional[List[str]] = None, threads: int = 1, show_progress: bool = True) -> SimulationResult:
        if threads > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = str(threads)
            if getattr(self, "runtime", None): self.runtime.set_spatial_threads(threads)
            
        if self.mock_execution or not self.layout: return self._execute_mock(parameters, protocol)

        from ion_flux.protocols.profiles import Sequence
        if protocol and isinstance(protocol, Sequence):
            session = self.start_session(parameters)
            data_hist = {"Time [s]": []}
            for k in self.layout.state_offsets.keys(): data_hist[k] = []
            for k in self.layout.obs_offsets.keys(): data_hist[k] = []
            raw_y_hist, raw_p_hist = [], []

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
                        session.step(low, inputs=inputs)
                        t_elapsed += low
                        data_hist["Time [s]"].append(session.time)
                        y = session.handle.get_state() if session.handle else session._mock_y
                        obs = session.handle.get_observables_py() if session.handle else np.zeros(self.layout.n_obs)
                        raw_y_hist.append(y)
                        if requires_grad: raw_p_hist.append(self._pack_parameters(session.parameters))
                        for k, (offset, size) in self.layout.state_offsets.items(): data_hist[k].append(y[offset:offset+size] if size > 1 else y[offset])
                        for k, (offset, size) in self.layout.obs_offsets.items(): data_hist[k].append(obs[offset:offset+size] if size > 1 else obs[offset])
                        break
                    
                    t_elapsed += dt_step
                    data_hist["Time [s]"].append(session.time)
                    y = session.handle.get_state() if session.handle else session._mock_y
                    obs = session.handle.get_observables_py() if session.handle else np.zeros(self.layout.n_obs)
                    raw_y_hist.append(y)
                    if requires_grad: raw_p_hist.append(self._pack_parameters(session.parameters))
                    for k, (offset, size) in self.layout.state_offsets.items(): data_hist[k].append(y[offset:offset+size] if size > 1 else y[offset])
                    for k, (offset, size) in self.layout.obs_offsets.items(): data_hist[k].append(obs[offset:offset+size] if size > 1 else obs[offset])

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
                    "Time [s]": data_hist["Time [s]"], "_y_raw": np.array(raw_y_hist), 
                    "_micro_t": np.array(session.micro_t), "_micro_y": np.array(session.micro_y),
                    "_micro_ydot": np.array(session.micro_ydot), "_p_traj": session.micro_p, "requires_grad": requires_grad
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
            trajectory = {"Time [s]": t_eval_arr, "_y_raw": y_res, "_micro_t": micro_t, "_micro_y": micro_y, "_micro_ydot": micro_ydot, "_p_traj": [p_list]*len(micro_t), "requires_grad": requires_grad}
        return SimulationResult(data, parameters or {}, status="completed", engine=self, trajectory=trajectory)

    def solve_batch(self, parameters: List[Dict[str, float]], t_span: tuple = (0, 1), protocols: Any = None, max_workers: int = 1, show_progress: bool = False) -> List[SimulationResult]:
        from ion_flux.protocols.profiles import Sequence
        if protocols:
            if isinstance(protocols, Sequence): protocols = [protocols] * len(parameters)
            elif len(protocols) != len(parameters): raise ValueError("Batch length mismatch.")
                
        if max_workers > 1 and "omp" in self.target:
            os.environ["OMP_NUM_THREADS"] = "1"
            if getattr(self, "runtime", None): self.runtime.set_spatial_threads(1)
            
        if self.mock_execution or not RUST_FFI_AVAILABLE:
            if not protocols: protocols = [None] * len(parameters)
            return [self.solve(t_span=t_span, protocol=prot, parameters=p) for p, prot in zip(parameters, protocols)]

        y0, ydot0, id_arr, spatial_diag, max_steps = self._extract_metadata()
        t_eval_arr = np.linspace(t_span[0], t_span[1], 100)
        p_batch = [self._pack_parameters(p) for p in parameters]
        m_list = self.layout.get_mesh_data()
        
        protocol_payloads = None
        if protocols:
            protocol_payloads = []
            def _get_p_idx(keys):
                for k in keys:
                    if k in self.layout.param_offsets: return self.layout.param_offsets[k][0]
                return 0
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
                        if var_name in self.layout.state_offsets: trig_idx, trig_size = self.layout.state_offsets[var_name]
                        elif var_name in self.layout.obs_offsets: trig_idx, trig_size, trig_is_obs = self.layout.obs_offsets[var_name][0], self.layout.obs_offsets[var_name][1], True
                        else: raise ValueError(f"Trigger variable '{var_name}' not found.")
                        trig_op = {">": 1, "<": 2, ">=": 3, "<=": 4, "==": 5, "!=": 6}.get(op_str, 0)
                        trig_val = float(t_val)
                    payload.append((step_type, target_val, time_limit, (has_trig, trig_idx, trig_size, trig_is_obs, trig_op, trig_val), p_idx_mode, p_idx_i, p_idx_v))
                protocol_payloads.append(payload)
        
        v_idx = self.layout.state_offsets.get("V_cell", (-1, 0))[0]
        
        try:
            y_res_batch = solve_batch_native(self.runtime.lib_path, y0, ydot0, id_arr, p_batch, m_list, t_eval_arr.tolist(), self.jacobian_bandwidth, spatial_diag, max_steps, self.layout.n_obs, self.debug, max_workers, show_progress, protocol_payloads, v_idx)
        except RuntimeError as e:
            self._handle_native_crash(e)
            
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

    async def solve_async(self, t_span: tuple = (0, 1), protocol: Any = None, parameters: Optional[Dict[str, float]] = None, t_eval: Optional[np.ndarray] = None, scheduler: Any = None) -> SimulationResult:
        if scheduler:
            async with scheduler: return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)
        return await asyncio.to_thread(self.solve, t_span, protocol, parameters, t_eval)

    def _execute_mock(self, parameters: Optional[Dict[str, float]], protocol: Any) -> SimulationResult:
        params = parameters or {}
        if params.get("c.t0") == float('inf'): raise RuntimeError("Native Solver Error: Newton convergence failure")
        time_len = len(protocol.time) if hasattr(protocol, "time") else 100
        data = {"Time [s]": np.arange(time_len, dtype=np.float64)}
        if hasattr(self, "layout") and self.layout:
            for state_name, (offset, size) in self.layout.state_offsets.items(): data[state_name] = np.zeros(time_len) if size == 1 else np.zeros((time_len, size))
            for obs_name, (offset, size) in self.layout.obs_offsets.items(): data[obs_name] = np.zeros(time_len) if size == 1 else np.zeros((time_len, size))
        data["Voltage [V]"] = np.array([4.2] * (time_len - 1) + [2.5])
        trajectory = {"Time [s]": data["Time [s]"], "_y_raw": np.zeros((time_len, getattr(self.layout, 'n_states', 1)))}
        return SimulationResult(data, params, status="completed", engine=self, trajectory=trajectory)