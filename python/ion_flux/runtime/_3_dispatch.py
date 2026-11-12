import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Protocol

from ion_flux.runtime.manifest import ExecutableManifest
from ion_flux.runtime._4_diagnostics import format_native_crash

try:
    from ion_flux._core import solve_native, solve_ida_sundials
    RUST_FFI_AVAILABLE = True
except ModuleNotFoundError:
    RUST_FFI_AVAILABLE = False


def _serialize_protocols(manifest: ExecutableManifest, protocols: Optional[List[Any]]) -> Optional[List[List[Any]]]:
    if not protocols:
        return None

    def _get_p_idx(keys):
        for k in keys:
            if k in manifest.layout.param_offsets:
                return manifest.layout.param_offsets[k][0]
        return 0

    p_idx_mode = _get_p_idx(["_term_mode", "mode"])
    p_idx_i = _get_p_idx(["_term_i_target", "i_target", "i_app"])
    p_idx_v = _get_p_idx(["_term_v_target", "v_target"])

    protocol_payloads = []
    for prot in protocols:
        if prot is None:
            protocol_payloads.append([])
            continue
        payload = []
        for step in prot.steps:
            step_type = {"CC": 0, "CV": 1, "Rest": 2}.get(type(step).__name__, 0)
            target_val = getattr(step, "rate", getattr(step, "voltage", 0.0))
            time_limit = getattr(step, "time", float('inf'))
            
            cond = getattr(step, "until", None)
            trig_info = (False, 0, 1, False, 0, 0.0)
            
            if cond:
                var_name, op_str, t_val = cond._compiled_logic
                is_obs = False
                if var_name in manifest.layout.state_offsets:
                    idx, size = manifest.layout.state_offsets[var_name]
                elif var_name in manifest.layout.obs_offsets:
                    idx, size = manifest.layout.obs_offsets[var_name]
                    is_obs = True
                else:
                    raise ValueError(f"Trigger variable '{var_name}' not found.")
                
                op_code = {">": 1, "<": 2, ">=": 3, "<=": 4, "==": 5, "!=": 6}.get(op_str, 0)
                trig_info = (True, idx, size, is_obs, op_code, float(t_val))
                
            payload.append((step_type, target_val, time_limit, trig_info, p_idx_mode, p_idx_i, p_idx_v))
        protocol_payloads.append(payload)
    return protocol_payloads


class SolverBackend(Protocol):
    def execute(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        pass


class SundialsBackend:
    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx
        self.manifest = ctx["manifest"]

    def execute(self) -> List[Tuple]:
        raw_results = []
        proto_list = self.ctx["protocols"] if self.ctx["protocols"] else [None] * len(self.ctx["parameters"])
        
        for p_dict, p_list, prot in zip(self.ctx["parameters"], self.ctx["p_batch"], proto_list):
            if prot is not None:
                res = self._run_sequence(p_dict, prot)
            else:
                y_res, obs_res, _, _, _ = solve_ida_sundials(
                    self.manifest.lib_path, self.ctx["y0"], self.ctx["ydot0"], self.manifest.id_arr, 
                    p_list, self.ctx["m_list"], self.ctx["t_eval"].tolist(), self.manifest.layout.n_obs,
                    *self.manifest.cpr_cache, self.ctx["show_progress"], self.ctx["v_idx"]
                )
                res = (self.ctx["t_eval"], y_res, obs_res, np.empty(0), np.empty((0, self.manifest.layout.n_states)), np.empty((0, self.manifest.layout.n_states)), {})
            raw_results.append(res)
        return raw_results

    def _run_sequence(self, p_dict: Dict[str, float], protocol: Any) -> Tuple:
        import sys
        from ion_flux._core import SundialsHandle
        from ion_flux.runtime._2_initializers import evaluate_ic

        y0, ydot0 = evaluate_ic(self.manifest, p_dict)
        handle = SundialsHandle(
            self.manifest.lib_path, self.manifest.layout.n_states, y0, ydot0, self.manifest.id_arr, 
            self.manifest.pack_parameters(p_dict), self.ctx["m_list"], self.manifest.layout.n_obs,
            *self.manifest.cpr_cache
        )

        p_mode = self._find_offset(["_term_mode", "mode"])
        p_i = self._find_offset(["_term_i_target", "i_target", "i_app"])
        p_v = self._find_offset(["_term_v_target", "v_target"])

        t_out, y_out, obs_out = [], [], []
        t_current = 0.0

        for step_idx, step in enumerate(protocol.steps):
            step_type = type(step).__name__
            target_val = getattr(step, "rate", getattr(step, "voltage", 0.0))
            time_limit = getattr(step, "time", float('inf'))
            condition = getattr(step, "until", None)

            if step_type == "CC":
                if p_mode is not None: handle.set_parameter(p_mode, 1.0)
                if p_i is not None: handle.set_parameter(p_i, target_val)
            elif step_type == "CV":
                if p_mode is not None: handle.set_parameter(p_mode, 0.0)
                if p_v is not None: handle.set_parameter(p_v, target_val)
            elif step_type == "Rest":
                if p_mode is not None: handle.set_parameter(p_mode, 1.0)
                if p_i is not None: handle.set_parameter(p_i, 0.0)

            handle.calc_algebraic_roots()
            if step_idx == 0:
                self._record_step(handle, t_current, t_out, y_out, obs_out)

            t_elapsed = 0.0
            while t_elapsed < time_limit:
                dt_step = min(1.0, time_limit - t_elapsed)
                handle.step(dt_step)
                t_current += dt_step
                t_elapsed += dt_step
                
                self._record_step(handle, t_current, t_out, y_out, obs_out)

                if condition and self._check_condition(condition, p_dict, y_out[-1], obs_out[-1]):
                    break

            if self.ctx["show_progress"]:
                v_val = y_out[-1][self.ctx["v_idx"]] if self.ctx["v_idx"] >= 0 else 0.0
                v_str = f" | V: {v_val:.3f}V" if self.ctx["v_idx"] >= 0 else ""
                sys.stdout.write(f"\r▶ {step_type:<4} [██████████████████████████████] 100.0% | t: {t_current:.1f}s{v_str}   \n")
                sys.stdout.flush()

        obs_mat = np.array(obs_out) if self.manifest.layout.n_obs > 0 else np.empty((len(t_out), 0))
        return (np.array(t_out), np.array(y_out), obs_mat, np.empty(0), np.empty((0, self.manifest.layout.n_states)), np.empty((0, self.manifest.layout.n_states)), {})

    def _find_offset(self, candidates):
        for c in candidates:
            if c in self.manifest.layout.param_offsets:
                return self.manifest.layout.param_offsets[c][0]
        return None

    def _record_step(self, handle, t, t_out, y_out, obs_out):
        t_out.append(t)
        y_out.append(handle.get_state())
        obs_out.append(handle.get_observables_py() if self.manifest.layout.n_obs > 0 else np.zeros(0))

    def _check_condition(self, condition, p_dict, y_curr, obs_curr) -> bool:
        var_name, op_str, target_thresh = condition._compiled_logic
        target_thresh = p_dict.get(target_thresh, 0.0) if isinstance(target_thresh, str) else float(target_thresh)

        if var_name in self.manifest.layout.state_offsets:
            v_val = y_curr[self.manifest.layout.state_offsets[var_name][0]]
        elif var_name in self.manifest.layout.obs_offsets:
            v_val = obs_curr[self.manifest.layout.obs_offsets[var_name][0]]
        else:
            return False

        if op_str == "<=": return v_val <= target_thresh
        elif op_str == ">=": return v_val >= target_thresh
        elif op_str == "<": return v_val < target_thresh
        elif op_str == ">": return v_val > target_thresh
        elif op_str == "==": return abs(v_val - target_thresh) < 1e-8
        return False


class NativeBackend:
    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx

    def execute(self) -> List[Tuple]:
        manifest = self.ctx["manifest"]
        return solve_native(
            manifest.lib_path, self.ctx["y0"], self.ctx["ydot0"], manifest.id_arr, 
            self.ctx["p_batch"], self.ctx["m_list"], self.ctx["t_eval"].tolist(), 
            manifest.jacobian_bandwidth, manifest.spatial_diag, manifest.max_steps, manifest.layout.n_obs,
            *manifest.cpr_cache, self.ctx["protocol_payloads"], self.ctx["record_tape"], 
            self.ctx["max_workers"], self.ctx["spatial_threads"], self.ctx["show_progress"], self.ctx["v_idx"]
        )


def dispatch_native(
    manifest: ExecutableManifest,
    y0: List[float], ydot0: List[float],
    parameters: List[Dict[str, float]],
    t_eval: np.ndarray,
    protocols: Optional[List[Any]],
    solver_backend: str,
    requires_grad: Optional[List[str]],
    max_workers: int, spatial_threads: int,
    show_progress: bool, debug: bool, v_idx: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, Any]]]]:
    """Unified dispatch gateway routing execution configurations to the correct solver backend."""
    if not RUST_FFI_AVAILABLE or not manifest.runtime:
        raise RuntimeError("Native solver missing or binary not compiled.")

    record_tape = requires_grad is not None
    if record_tape and len(parameters) > 100:
        import warnings
        warnings.warn("Batch solve requested gradient tape recording (`record_tape=True`) for large fleet.", ResourceWarning, stacklevel=2)
        
    ctx = {
        "manifest": manifest, "y0": y0, "ydot0": ydot0, "parameters": parameters,
        "p_batch": [manifest.pack_parameters(p) for p in parameters],
        "m_list": manifest.layout.get_mesh_data(), "t_eval": t_eval, "protocols": protocols,
        "protocol_payloads": _serialize_protocols(manifest, protocols),
        "record_tape": record_tape, "max_workers": max_workers, "spatial_threads": spatial_threads,
        "show_progress": show_progress, "debug": debug, "v_idx": v_idx
    }

    backend = SundialsBackend(ctx) if solver_backend == "sundials" else NativeBackend(ctx)

    try:
        raw_results = backend.execute()
    except RuntimeError as e:
        raise format_native_crash(e, manifest) from None

    dispatched = []
    for p_dict, p_list, (t_res, y_res, obs_res, micro_t, micro_y, micro_ydot, diag_dict) in zip(parameters, ctx["p_batch"], raw_results):
        trajectory = None
        if record_tape and len(micro_t) > 0:
            trajectory = {
                "Time [s]": t_res, "_y_raw": y_res,
                "_micro_t": micro_t, "_micro_y": micro_y, "_micro_ydot": micro_ydot,
                "_p_traj": np.tile(p_list, (len(micro_t), 1)), "requires_grad": requires_grad,
            }
        dispatched.append((t_res, y_res, obs_res, trajectory, diag_dict))

    return dispatched


def run_single(manifest: ExecutableManifest, y0: List[float], ydot0: List[float], parameters: Dict[str, float], 
               t_eval: np.ndarray, solver_backend: str, debug: bool, show_progress: bool, record_history: bool, v_idx: int) -> Tuple:
    req_grad = ["_tape"] if record_history else None
    res = dispatch_native(manifest, y0, ydot0, [parameters], t_eval, None, solver_backend, req_grad, 1, 1, show_progress, debug, v_idx)[0]
    return (res[1], res[2], res[3]["_micro_t"].tolist(), res[3]["_micro_y"].tolist(), res[3]["_micro_ydot"].tolist()) if res[3] else (res[1], res[2], [], [], [])


def run_batch(manifest: ExecutableManifest, y0: List[float], ydot0: List[float], parameters: List[Dict[str, float]], 
              t_eval: np.ndarray, protocols: Any, max_workers: int, debug: bool, show_progress: bool, v_idx: int) -> Any:
    protocol_list = [protocols] * len(parameters) if protocols and not isinstance(protocols, list) else protocols
    dispatched = dispatch_native(manifest, y0, ydot0, parameters, t_eval, protocol_list, "native", None, max_workers, 1, show_progress, debug, v_idx)
    return [(t, y, o) for (t, y, o, _, _) in dispatched]


def evaluate_jacobian(manifest: ExecutableManifest, y: List[float], ydot: List[float], c_j: float, parameters: Dict[str, float]) -> List[List[float]]:
    if not manifest.runtime: raise RuntimeError("Requires native execution.")
    c_seeds, c_ptrs, c_rows, c_cols, c_dense = manifest.cpr_cache
    p_list = manifest.pack_parameters(parameters)
    m_list = manifest.layout.get_mesh_data()
    N = manifest.layout.n_states
    
    J = [[0.0] * N for _ in range(N)]
    for c_idx, seed in enumerate(c_seeds):
        jvp_out = manifest.runtime.evaluate_jvp(y, ydot, p_list, m_list, c_j, seed)
        for i in range(c_ptrs[c_idx], c_ptrs[c_idx + 1]):
            J[c_rows[i]][c_cols[i]] = jvp_out[c_rows[i]]
            
    for r in c_dense:
        lam = [0.0] * N
        lam[r] = 1.0
        _, dy_out, dydot_out = manifest.runtime.evaluate_vjp(y, ydot, p_list, m_list, lam)
        for c in range(N):
            val = dy_out[c] + c_j * dydot_out[c]
            if abs(val) > 1e-16 or math.isnan(val):
                J[r][c] = val
    return J


def execute_mock(manifest: ExecutableManifest, parameters: Dict[str, float], protocol: Any) -> Any:
    if parameters and parameters.get("c.t0") == float('inf'): raise RuntimeError("Native Solver Error: Newton convergence failure")
    time_len = len(protocol.time) if hasattr(protocol, "time") else 100
    data = {"Time [s]": np.arange(time_len, dtype=np.float64)}
    
    if manifest and manifest.layout:
        for d, offs in [(data, manifest.layout.state_offsets), (data, manifest.layout.obs_offsets)]:
            for name, (_, size) in offs.items():
                d[name] = np.zeros(time_len) if size == 1 else np.zeros((time_len, size))
            
    data["Voltage [V]"] = np.array([4.2] * (time_len - 1) + [2.5])
    from ion_flux.runtime.results import SimulationResult
    return SimulationResult(data, parameters or {}, status="completed", trajectory={"Time [s]": data["Time [s]"], "_y_raw": np.zeros((time_len, getattr(manifest.layout, 'n_states', 1)))}, diagnostics={})