import json
import shutil
import os
import ctypes
from typing import Dict, Any, List, Tuple, Optional
from ion_flux.compiler._2_middle_end.memory_layout import MemoryLayout
from ion_flux.runtime.ffi_runtime import NativeRuntime

class ExecutableManifest:
    """
    The immutable artifact emitted by the Compiler.
    Stores the memory layouts, differential-algebraic topological identifiers, 
    CPR coloring graphs, and the native Rust/C++ executable target.
    """
    def __init__(self, 
                 lib_path: str,
                 layout: MemoryLayout,
                 default_parameters: Dict[str, float],
                 ast_payload: Dict[str, Any],
                 jacobian_bandwidth: int,
                 cpr_cache: Tuple[List, List, List, List, List],
                 id_arr: List[float],
                 spatial_diag: List[float],
                 max_steps: List[float],
                 state_domain_map: Dict[str, str],
                 cpp_source: str = ""):
        self.lib_path = lib_path
        self.layout = layout
        self.default_parameters = default_parameters
        self.ast_payload = ast_payload
        self.jacobian_bandwidth = jacobian_bandwidth
        self.cpr_cache = cpr_cache
        self.id_arr = id_arr
        self.spatial_diag = spatial_diag
        self.max_steps = max_steps
        self.state_domain_map = state_domain_map
        self.cpp_source = cpp_source
        
        self.terminal_meta = {
            "has_terminal": "_term_mode" in self.layout.param_offsets,
            "mode_offset": self.layout.param_offsets.get("_term_mode", (None, 0))[0],
            "i_target_offset": self.layout.param_offsets.get("_term_i_target", (None, 0))[0],
            "v_target_offset": self.layout.param_offsets.get("_term_v_target", (None, 0))[0],
        }

        self.runtime = NativeRuntime(lib_path, layout.n_states) if lib_path else None

    def pack_parameters(self, overrides: Dict[str, float]) -> List[float]:
        """Safely maps a high-level dictionary to the flat C-ABI array."""
        p_list = [0.0] * self.layout.p_length
        for p_name, (offset, _) in self.layout.param_offsets.items():
            p_list[offset] = overrides.get(p_name, self.default_parameters.get(p_name, 0.0))
        return p_list

    def save(self, export_path: str) -> None:
        """Serializes the standalone compiled binary for single-artifact deployments."""
        if self.lib_path and os.path.exists(self.lib_path):
            shutil.copy(self.lib_path, export_path)

    @classmethod
    def load(cls, binary_path: str) -> "ExecutableManifest":
        """Deserializes a compiled model with strictly 0ms Clang parsing overhead."""
        meta = None
        try:
            dll = ctypes.CDLL(binary_path)
            if hasattr(dll, "get_ion_flux_metadata"):
                dll.get_ion_flux_metadata.restype = ctypes.c_char_p
                raw = dll.get_ion_flux_metadata()
                if raw:
                    meta = json.loads(raw.decode("utf-8"))
        except Exception:
            meta = None

        if meta is None:
            meta_path = binary_path + ".meta.json"
            if not os.path.exists(meta_path): 
                raise FileNotFoundError(f"Missing embedded metadata or layout manifest at {meta_path}.")
            with open(meta_path, "r") as f: 
                meta = json.load(f)
        
        layout = MemoryLayout.from_dict(meta["layout"])
        default_parameters = meta["parameters"]
        jacobian_bandwidth = meta.get("jacobian_bandwidth", 0)
        cpr_cache = meta.get("cpr_cache", ([], [], [], [], []))
        id_arr = meta["metadata_cache"]["id_arr"]
        spatial_diag = meta["metadata_cache"].get("spatial_diag", [0.0] * layout.n_states)
        max_steps = meta["metadata_cache"].get("max_steps", [0.0] * layout.n_states)
        ast_payload = meta.get("ast_payload", {})
        state_domain_map = meta.get("state_domain_map", {})
        cpp_source = meta.get("cpp_source", "")
        
        return cls(binary_path, layout, default_parameters, ast_payload, jacobian_bandwidth, 
                   cpr_cache, id_arr, spatial_diag, max_steps, state_domain_map, cpp_source)