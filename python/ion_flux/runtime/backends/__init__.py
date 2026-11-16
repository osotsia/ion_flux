import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Protocol

from ion_flux.runtime.manifest import ExecutableManifest
from ion_flux.runtime.diagnostics import format_native_crash
from ion_flux.runtime.backends.native import NativeBackend, RUST_FFI_AVAILABLE, _serialize_protocols
from ion_flux.runtime.backends.sundials import SundialsBackend
from ion_flux.runtime.backends.mock import execute_mock


class SolverBackend(Protocol):
    def execute(self) -> List[Tuple]:
        pass


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
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, Any]], Dict[str, Any]]]:
    """Unified dispatch gateway routing execution configurations to the correct solver backend."""
    if not RUST_FFI_AVAILABLE or not manifest.native_lib:
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


__all__ = [
    "SolverBackend",
    "NativeBackend",
    "SundialsBackend",
    "dispatch_native",
    "execute_mock",
    "RUST_FFI_AVAILABLE",
]
