import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ion_flux.runtime.manifest import ExecutableManifest
from ion_flux.runtime._4_diagnostics import format_native_crash

try:
    from ion_flux._core import solve_native, solve_ida_sundials
    RUST_FFI_AVAILABLE = True
except ModuleNotFoundError:
    # Safe fallback for pure-Python AST testing
    RUST_FFI_AVAILABLE = False


def _serialize_protocols(manifest: ExecutableManifest, protocols: Optional[List[Any]]) -> Optional[List[List[Any]]]:
    if not protocols:
        return None
    protocol_payloads = []

    def _get_p_idx(keys):
        for k in keys:
            if k in manifest.layout.param_offsets:
                return manifest.layout.param_offsets[k][0]
        return 0

    p_idx_mode = _get_p_idx(["_term_mode", "mode"])
    p_idx_i = _get_p_idx(["_term_i_target", "i_target", "i_app"])
    p_idx_v = _get_p_idx(["_term_v_target", "v_target"])

    for prot in protocols:
        if prot is None:
            protocol_payloads.append([])
            continue
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
                if var_name in manifest.layout.state_offsets:
                    trig_idx, trig_size = manifest.layout.state_offsets[var_name]
                elif var_name in manifest.layout.obs_offsets:
                    trig_idx, trig_size, trig_is_obs = manifest.layout.obs_offsets[var_name][0], manifest.layout.obs_offsets[var_name][1], True
                else:
                    raise ValueError(f"Trigger variable '{var_name}' not found.")
                trig_op = {">": 1, "<": 2, ">=": 3, "<=": 4, "==": 5, "!=": 6}.get(op_str, 0)
                trig_val = float(t_val)
            payload.append((step_type, target_val, time_limit, (has_trig, trig_idx, trig_size, trig_is_obs, trig_op, trig_val), p_idx_mode, p_idx_i, p_idx_v))
        protocol_payloads.append(payload)
    return protocol_payloads


def _run_sundials_sequence(
    manifest: ExecutableManifest,
    p_dict: Dict[str, float],
    protocol: Any,
    show_progress: bool,
    v_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import sys
    from ion_flux._core import SundialsHandle
    from ion_flux.runtime._2_initializers import evaluate_ic

    y0, ydot0 = evaluate_ic(manifest, p_dict)
    p_list = manifest.pack_parameters(p_dict)
    m_list = manifest.layout.get_mesh_data()
    c_seeds, c_ptrs, c_rows, c_cols, c_dense = manifest.cpr_cache

    handle = SundialsHandle(
        manifest.lib_path, manifest.layout.n_states,
        y0, ydot0, manifest.id_arr, p_list, m_list, manifest.layout.n_obs,
        c_seeds, c_ptrs, c_rows, c_cols, c_dense
    )

    def _find_offset(candidates):
        for c in candidates:
            if c in manifest.layout.param_offsets:
                return manifest.layout.param_offsets[c][0]
        return None

    p_mode = _find_offset(["_term_mode", "mode"])
    p_i = _find_offset(["_term_i_target", "i_target", "i_app"])
    p_v = _find_offset(["_term_v_target", "v_target"])

    t_out = []
    y_out = []
    obs_out = []
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
            t_out.append(t_current)
            y_out.append(handle.get_state())
            obs_out.append(handle.get_observables_py() if manifest.layout.n_obs > 0 else np.zeros(0))

        t_elapsed = 0.0
        while t_elapsed < time_limit:
            dt_step = min(1.0, time_limit - t_elapsed)
            handle.step(dt_step)
            t_current += dt_step
            t_elapsed += dt_step

            y_curr = handle.get_state()
            obs_curr = handle.get_observables_py() if manifest.layout.n_obs > 0 else np.zeros(0)

            t_out.append(t_current)
            y_out.append(y_curr)
            obs_out.append(obs_curr)

            if condition:
                var_name, op_str, target_thresh = condition._compiled_logic
                if isinstance(target_thresh, str):
                    target_thresh = p_dict.get(target_thresh, 0.0)
                else:
                    target_thresh = float(target_thresh)

                if var_name in manifest.layout.state_offsets:
                    off, sz = manifest.layout.state_offsets[var_name]
                    v_val = y_curr[off]
                elif var_name in manifest.layout.obs_offsets:
                    off, sz = manifest.layout.obs_offsets[var_name]
                    v_val = obs_curr[off]
                else:
                    v_val = 0.0

                trig = False
                if op_str == "<=": trig = bool(v_val <= target_thresh)
                elif op_str == ">=": trig = bool(v_val >= target_thresh)
                elif op_str == "<": trig = bool(v_val < target_thresh)
                elif op_str == ">": trig = bool(v_val > target_thresh)
                elif op_str == "==": trig = bool(abs(v_val - target_thresh) < 1e-8)

                if trig:
                    break

        if show_progress:
            v_val_disp = y_out[-1][v_idx] if v_idx >= 0 and len(y_out) > 0 else 0.0
            v_str = f" | V: {v_val_disp:.3f}V" if v_idx >= 0 else ""
            sys.stdout.write(f"\r▶ {step_type:<4} [██████████████████████████████] 100.0% | t: {t_current:.1f}s{v_str}   \n")
            sys.stdout.flush()

    y_mat = np.array(y_out)
    obs_mat = np.array(obs_out) if manifest.layout.n_obs > 0 else np.empty((len(t_out), 0))
    t_arr = np.array(t_out)
    empty_micro = np.empty(0)
    empty_y = np.empty((0, manifest.layout.n_states))
    return (t_arr, y_mat, obs_mat, empty_micro, empty_y, empty_y)


def dispatch_native(
    manifest: ExecutableManifest,
    y0: List[float],
    ydot0: List[float],
    parameters: List[Dict[str, float]],
    t_eval: np.ndarray,
    protocols: Optional[List[Any]],
    solver_backend: str,
    requires_grad: Optional[List[str]],
    max_workers: int,
    spatial_threads: int,
    show_progress: bool,
    debug: bool,
    v_idx: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, Any]]]]:
    """Unified dispatch gateway routing single and batch executions through solve_native."""
    if not RUST_FFI_AVAILABLE or not manifest.runtime:
        raise RuntimeError("Native solver missing or binary not compiled.")

    p_batch = [manifest.pack_parameters(p) for p in parameters]
    m_list = manifest.layout.get_mesh_data()
    c_seeds, c_ptrs, c_rows, c_cols, c_dense = manifest.cpr_cache
    record_tape = requires_grad is not None
    if record_tape and len(parameters) > 100:
        import warnings
        warnings.warn(
            f"Batch solve requested gradient tape recording (`record_tape=True`) for {len(parameters)} models. "
            "High memory usage may occur if each model records thousands of BDF steps.",
            ResourceWarning,
            stacklevel=2,
        )
    protocol_payloads = _serialize_protocols(manifest, protocols)

    try:
        if solver_backend == "sundials":
            raw_results = []
            proto_list = protocols if protocols else [None] * len(parameters)
            for p_dict, p_list, prot in zip(parameters, p_batch, proto_list):
                if prot is not None:
                    res_tuple = _run_sundials_sequence(manifest, p_dict, prot, show_progress, v_idx)
                else:
                    y_res, obs_res, _, _, _ = solve_ida_sundials(
                        manifest.lib_path, y0, ydot0, manifest.id_arr, p_list, m_list,
                        t_eval.tolist(), manifest.layout.n_obs,
                        c_seeds, c_ptrs, c_rows, c_cols, c_dense,
                        show_progress, v_idx
                    )
                    res_tuple = (t_eval, y_res, obs_res, np.empty(0), np.empty((0, manifest.layout.n_states)), np.empty((0, manifest.layout.n_states)))
                raw_results.append(res_tuple)
        else:
            raw_results = solve_native(
                manifest.lib_path, y0, ydot0, manifest.id_arr, p_batch, m_list,
                t_eval.tolist(), manifest.jacobian_bandwidth, manifest.spatial_diag,
                manifest.max_steps, manifest.layout.n_obs,
                c_seeds, c_ptrs, c_rows, c_cols, c_dense,
                protocol_payloads, record_tape, max_workers, spatial_threads, show_progress, v_idx
            )
    except RuntimeError as e:
        raise format_native_crash(e, manifest) from None

    dispatched = []
    for p_dict, p_list, (t_res, y_res, obs_res, micro_t, micro_y, micro_ydot) in zip(parameters, p_batch, raw_results):
        trajectory = None
        if record_tape and len(micro_t) > 0:
            trajectory = {
                "Time [s]": t_res,
                "_y_raw": y_res,
                "_micro_t": micro_t,
                "_micro_y": micro_y,
                "_micro_ydot": micro_ydot,
                "_p_traj": np.tile(p_list, (len(micro_t), 1)),
                "requires_grad": requires_grad,
            }
        dispatched.append((t_res, y_res, obs_res, trajectory))

    return dispatched


def run_single(manifest: ExecutableManifest, y0: List[float], ydot0: List[float], 
               parameters: Dict[str, float], t_eval: np.ndarray, 
               solver_backend: str, debug: bool, show_progress: bool, 
               record_history: bool, v_idx: int) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[float]]:
    """Backward-compatible shim delegating to dispatch_native."""
    req_grad = ["_tape"] if record_history else None
    res = dispatch_native(
        manifest, y0, ydot0, [parameters], t_eval, None,
        solver_backend, req_grad, 1, 1, show_progress, debug, v_idx
    )[0]
    t_res, y_res, obs_res, traj = res
    if traj is not None:
        return y_res, obs_res, traj["_micro_t"].tolist(), traj["_micro_y"].tolist(), traj["_micro_ydot"].tolist()
    else:
        return y_res, obs_res, [], [], []


def run_batch(manifest: ExecutableManifest, y0: List[float], ydot0: List[float], 
              parameters: List[Dict[str, float]], t_eval: np.ndarray, protocols: Any,
              max_workers: int, debug: bool, show_progress: bool, v_idx: int) -> Any:
    """Backward-compatible shim delegating to dispatch_native."""
    protocol_list = protocols if protocols is not None else None
    if protocol_list and not isinstance(protocol_list, list):
        protocol_list = [protocol_list] * len(parameters)
    dispatched = dispatch_native(
        manifest, y0, ydot0, parameters, t_eval, protocol_list,
        "native", None, max_workers, 1, show_progress, debug, v_idx
    )
    return [(t_res, y_res, obs_res) for (t_res, y_res, obs_res, _) in dispatched]

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