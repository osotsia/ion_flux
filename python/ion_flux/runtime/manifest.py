import json
import shutil
import os
from typing import Dict, Any, List, Tuple, Optional
from ion_flux.stage2_compiler._1_analysis.memory_layout import MemoryLayout
from ion_flux.stage3_backend.ffi_runtime import NativeRuntime

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
        
        self.runtime = NativeRuntime(lib_path, layout.n_states) if lib_path else None

    def pack_parameters(self, overrides: Dict[str, float]) -> List[float]:
        """Safely maps a high-level dictionary to the flat C-ABI array."""
        p_list = [0.0] * self.layout.p_length
        for p_name, (offset, _) in self.layout.param_offsets.items():
            p_list[offset] = overrides.get(p_name, self.default_parameters.get(p_name, 0.0))
        return p_list

    def save(self, export_path: str) -> None:
        """Serializes the execution topology for serverless deployments."""
        meta = {
            "layout": {
                "state_offsets": self.layout.state_offsets, "param_offsets": self.layout.param_offsets, "obs_offsets": self.layout.obs_offsets,
                "n_states": self.layout.n_states, "n_params": self.layout.n_params, "n_obs": self.layout.n_obs,
                "p_length": self.layout.p_length, "m_length": self.layout.m_length, "mesh_offsets": self.layout.mesh_offsets, "mesh_cache": self.layout.mesh_cache
            },
            "parameters": self.default_parameters,
            "jacobian_bandwidth": self.jacobian_bandwidth,
            "metadata_cache": {
                "id_arr": self.id_arr, "spatial_diag": self.spatial_diag, "max_steps": self.max_steps
            },
            "cpr_cache": self.cpr_cache,
            "ast_payload": self.ast_payload,
            "state_domain_map": self.state_domain_map,
            "cpp_source": self.cpp_source
        }
        with open(export_path + ".meta.json", "w") as f: 
            json.dump(meta, f)
        if self.lib_path and os.path.exists(self.lib_path):
            shutil.copy(self.lib_path, export_path)

    @classmethod
    def load(cls, binary_path: str) -> "ExecutableManifest":
        """Deserializes a compiled model with strictly 0ms Clang parsing overhead."""
        meta_path = binary_path + ".meta.json"
        if not os.path.exists(meta_path): 
            raise FileNotFoundError(f"Missing layout manifest at {meta_path}.")
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