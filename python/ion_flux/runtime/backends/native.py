from typing import List, Dict, Any, Optional, Tuple

from ion_flux.runtime.manifest import ExecutableManifest

try:
    from ion_flux._core import solve_native
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
