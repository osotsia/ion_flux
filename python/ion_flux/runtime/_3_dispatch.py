import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ion_flux.runtime.manifest import ExecutableManifest
from ion_flux.runtime._4_diagnostics import format_native_crash

try:
    from ion_flux._core import solve_ida_native, solve_ida_sundials, solve_batch_native
    RUST_FFI_AVAILABLE = True
except ImportError:
    RUST_FFI_AVAILABLE = False

def run_single(manifest: ExecutableManifest, y0: List[float], ydot0: List[float], 
               parameters: Dict[str, float], t_eval: np.ndarray, 
               solver_backend: str, debug: bool, show_progress: bool, 
               record_history: bool, v_idx: int) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[float]]:
    """Orchestrates FFI delegation for a single run, handling parameter packing and diagnostic wrapping."""
    if not RUST_FFI_AVAILABLE or not manifest.runtime: 
        raise RuntimeError("Native solver missing or binary not compiled.")
    
    p_list = manifest.pack_parameters(parameters)
    m_list = manifest.layout.get_mesh_data()
    c_seeds, c_ptrs, c_rows, c_cols, c_dense = manifest.cpr_cache
    
    try:
        if solver_backend == "sundials":
            return solve_ida_sundials(
                manifest.lib_path, y0, ydot0, manifest.id_arr, p_list, m_list, t_eval.tolist(), manifest.layout.n_obs,
                c_seeds, c_ptrs, c_rows, c_cols, c_dense, show_progress, v_idx
            )
        else:
            return solve_ida_native(
                manifest.lib_path, y0, ydot0, manifest.id_arr, p_list, m_list, t_eval.tolist(), 
                manifest.jacobian_bandwidth, manifest.spatial_diag, manifest.max_steps, manifest.layout.n_obs,
                c_seeds, c_ptrs, c_rows, c_cols, c_dense,
                record_history, debug, show_progress, v_idx
            )
    except RuntimeError as e:
        raise format_native_crash(e, manifest) from None

def run_batch(manifest: ExecutableManifest, y0: List[float], ydot0: List[float], 
              parameters: List[Dict[str, float]], t_eval: np.ndarray, protocols: Any,
              max_workers: int, debug: bool, show_progress: bool, v_idx: int) -> Any:
    """Orchestrates thread-parallel Rayon dispatch bypassing the Python GIL."""
    if not RUST_FFI_AVAILABLE or not manifest.runtime: 
        raise RuntimeError("Native solver missing or binary not compiled.")

    p_batch = [manifest.pack_parameters(p) for p in parameters]
    m_list = manifest.layout.get_mesh_data()
    c_seeds, c_ptrs, c_rows, c_cols, c_dense = manifest.cpr_cache
    
    protocol_payloads = None
    if protocols:
        protocol_payloads = []
        def _get_p_idx(keys):
            for k in keys:
                if k in manifest.layout.param_offsets: return manifest.layout.param_offsets[k][0]
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
                    if var_name in manifest.layout.state_offsets: trig_idx, trig_size = manifest.layout.state_offsets[var_name]
                    elif var_name in manifest.layout.obs_offsets: trig_idx, trig_size, trig_is_obs = manifest.layout.obs_offsets[var_name][0], manifest.layout.obs_offsets[var_name][1], True
                    else: raise ValueError(f"Trigger variable '{var_name}' not found.")
                    trig_op = {">": 1, "<": 2, ">=": 3, "<=": 4, "==": 5, "!=": 6}.get(op_str, 0)
                    trig_val = float(t_val)
                payload.append((step_type, target_val, time_limit, (has_trig, trig_idx, trig_size, trig_is_obs, trig_op, trig_val), p_idx_mode, p_idx_i, p_idx_v))
            protocol_payloads.append(payload)

    try:
        return solve_batch_native(
            manifest.lib_path, y0, ydot0, manifest.id_arr, p_batch, m_list, t_eval.tolist(), 
            manifest.jacobian_bandwidth, manifest.spatial_diag, manifest.max_steps, manifest.layout.n_obs, 
            c_seeds, c_ptrs, c_rows, c_cols, c_dense, 
            debug, max_workers, show_progress, protocol_payloads, v_idx
        )
    except RuntimeError as e:
        raise format_native_crash(e, manifest) from None

def evaluate_jacobian(manifest: ExecutableManifest, y: List[float], ydot: List[float], c_j: float, parameters: Dict[str, float]) -> List[List[float]]:
    """Evaluates the structural CPR Jacobian natively via Enzyme JVP FFI routines."""
    if not manifest.runtime: raise RuntimeError("Requires native execution.")
    c_seeds, c_ptrs, c_rows, c_cols, c_dense = manifest.cpr_cache
    p_list = manifest.pack_parameters(parameters)
    m_list = manifest.layout.get_mesh_data()
    N = manifest.layout.n_states
    
    J = [[0.0] * N for _ in range(N)]
    for c_idx, seed in enumerate(c_seeds):
        jvp_out = manifest.runtime.evaluate_jvp(y, ydot, p_list, m_list, c_j, seed)
        start, end = c_ptrs[c_idx], c_ptrs[c_idx + 1]
        for i in range(start, end):
            r, c = c_rows[i], c_cols[i]
            J[r][c] = jvp_out[r]
            
    if c_dense:
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
    """Provides CI testing and safe-fallback mock evaluation."""
    params = parameters or {}
    if params.get("c.t0") == float('inf'): raise RuntimeError("Native Solver Error: Newton convergence failure")
    time_len = len(protocol.time) if hasattr(protocol, "time") else 100
    
    data = {"Time [s]": np.arange(time_len, dtype=np.float64)}
    if manifest and manifest.layout:
        for state_name, (offset, size) in manifest.layout.state_offsets.items(): data[state_name] = np.zeros(time_len) if size == 1 else np.zeros((time_len, size))
        for obs_name, (offset, size) in manifest.layout.obs_offsets.items(): data[obs_name] = np.zeros(time_len) if size == 1 else np.zeros((time_len, size))
            
    data["Voltage [V]"] = np.array([4.2] * (time_len - 1) + [2.5])
    trajectory = {"Time [s]": data["Time [s]"], "_y_raw": np.zeros((time_len, getattr(manifest.layout, 'n_states', 1)))}
    
    from ion_flux.runtime.results import SimulationResult
    return SimulationResult(data, params, status="completed", engine=None, trajectory=trajectory)