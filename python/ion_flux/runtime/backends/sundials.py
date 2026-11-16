import numpy as np
from typing import List, Dict, Any, Tuple

try:
    from ion_flux._core import solve_ida_sundials, solve_ida_sequence
    RUST_FFI_AVAILABLE = True
except ModuleNotFoundError:
    RUST_FFI_AVAILABLE = False


class SundialsBackend:
    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx
        self.manifest = ctx["manifest"]

    def execute(self) -> List[Tuple]:
        raw_results = []
        n_states = self.manifest.layout.n_states
        proto_payloads = self.ctx["protocol_payloads"] or [[] for _ in self.ctx["p_batch"]]

        for p_list, raw_steps in zip(self.ctx["p_batch"], proto_payloads):
            if raw_steps:
                t_res, y_res, obs_res = solve_ida_sequence(
                    self.manifest.lib_path, self.ctx["y0"], self.ctx["ydot0"], self.manifest.id_arr,
                    p_list, self.ctx["m_list"], self.manifest.layout.n_obs,
                    *self.manifest.cpr_cache, raw_steps, self.ctx["show_progress"], self.ctx["v_idx"]
                )
                res = (t_res, y_res, obs_res, np.empty(0), np.empty((0, n_states)), np.empty((0, n_states)), {})
            else:
                y_res, obs_res, _, _, _ = solve_ida_sundials(
                    self.manifest.lib_path, self.ctx["y0"], self.ctx["ydot0"], self.manifest.id_arr,
                    p_list, self.ctx["m_list"], self.ctx["t_eval"].tolist(), self.manifest.layout.n_obs,
                    *self.manifest.cpr_cache, self.ctx["show_progress"], self.ctx["v_idx"]
                )
                res = (self.ctx["t_eval"], y_res, obs_res, np.empty(0), np.empty((0, n_states)), np.empty((0, n_states)), {})
            raw_results.append(res)
        return raw_results
