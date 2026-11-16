import numpy as np
from typing import Dict, Any

from ion_flux.runtime.manifest import ExecutableManifest


def execute_mock(manifest: ExecutableManifest, parameters: Dict[str, float], protocol: Any) -> Any:
    if parameters and parameters.get("c.t0") == float('inf'):
        raise RuntimeError("Native Solver Error: Newton convergence failure")
    time_len = len(protocol.time) if hasattr(protocol, "time") else 100
    data = {"Time [s]": np.arange(time_len, dtype=np.float64)}

    if manifest and manifest.layout:
        for d, offs in [(data, manifest.layout.state_offsets), (data, manifest.layout.obs_offsets)]:
            for name, (_, size) in offs.items():
                d[name] = np.zeros(time_len) if size == 1 else np.zeros((time_len, size))

    data["Voltage [V]"] = np.array([4.2] * (time_len - 1) + [2.5])
    from ion_flux.runtime.results import SimulationResult
    return SimulationResult(
        data,
        parameters or {},
        status="completed",
        trajectory={"Time [s]": data["Time [s]"], "_y_raw": np.zeros((time_len, getattr(manifest.layout, 'n_states', 1)))},
        diagnostics={},
    )


def mock_eis_impedance(w_arr: np.ndarray) -> np.ndarray:
    """Mock transfer function for AST-only test/CI environments."""
    return 0.05 + (0.1 / (1 + 1j * w_arr * 0.1)) + (0.01 / np.sqrt(1j * w_arr))
