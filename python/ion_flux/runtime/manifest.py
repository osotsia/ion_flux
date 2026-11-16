import json
import shutil
import os
import ctypes
import tempfile
import itertools
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ion_flux.compiler._1_frontend.core import PDE, State, Parameter, Observable
from ion_flux.compiler._1_frontend.spatial import Domain, CompositeDomain
from ion_flux.compiler._2_middle_end.memory_layout import MemoryLayout
from ion_flux.compiler._2_middle_end.topology import TopologyAnalyzer
from ion_flux.compiler.pipeline import Compiler
from ion_flux.compiler._4_codegen.clang_invoker import NativeCompiler

try:
    from ion_flux._core import NativeLibrary
    RUST_FFI_AVAILABLE = True
except ModuleNotFoundError:
    NativeLibrary = None
    RUST_FFI_AVAILABLE = False


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

        self.native_lib = (
            NativeLibrary(lib_path, layout.n_states, layout.p_length, layout.n_obs)
            if (lib_path and RUST_FFI_AVAILABLE) else None
        )

    def evaluate_ic(self, overrides: Optional[Dict[str, float]] = None) -> Tuple[List[float], List[float]]:
        if not self.native_lib:
            return [0.0] * self.layout.n_states, [0.0] * self.layout.n_states
        p_list = self.pack_parameters(overrides or {})
        m_list = self.layout.get_mesh_data()
        return self.native_lib.evaluate_ic(p_list, m_list)

    def evaluate_residual(self, y: List[float], ydot: List[float], overrides: Optional[Dict[str, float]] = None) -> List[float]:
        if not self.native_lib:
            raise RuntimeError("Requires native execution.")
        p_list = self.pack_parameters(overrides or {})
        m_list = self.layout.get_mesh_data()
        return self.native_lib.evaluate_residual(y, ydot, p_list, m_list)

    def evaluate_jacobian(self, y: List[float], ydot: List[float], c_j: float, overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
        if not self.native_lib:
            raise RuntimeError("Requires native execution.")
        p_list = self.pack_parameters(overrides or {})
        m_list = self.layout.get_mesh_data()
        c_seeds, c_ptrs, c_rows, c_cols, c_dense = self.cpr_cache
        return self.native_lib.evaluate_jacobian(y, ydot, p_list, m_list, c_j, c_seeds, c_ptrs, c_rows, c_cols, c_dense)

    def evaluate_vjp(self, y: List[float], ydot: List[float], overrides: Optional[Dict[str, float]], lambda_vec: List[float]) -> Tuple[List[float], List[float], List[float]]:
        if not self.native_lib:
            raise RuntimeError("Requires native execution.")
        p_list = self.pack_parameters(overrides or {})
        m_list = self.layout.get_mesh_data()
        return self.native_lib.evaluate_vjp(y, ydot, p_list, m_list, lambda_vec)

    def extract_eis_system(self, y: List[float], ydot: List[float], overrides: Optional[Dict[str, float]], input_var: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.native_lib:
            raise RuntimeError("Requires native execution.")
        p_list = self.pack_parameters(overrides or {})
        m_list = self.layout.get_mesh_data()
        input_param_idx = self.layout.get_param_offset(input_var)
        c_seeds, c_ptrs, c_rows, c_cols, c_dense = self.cpr_cache
        return self.native_lib.extract_eis_system(
            y, ydot, p_list, m_list, input_param_idx, c_seeds, c_ptrs, c_rows, c_cols, c_dense
        )

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


def build_manifest(model: PDE, target: str = "cpu:serial", cache: bool = True, jacobian_bandwidth: Optional[int] = None, mock_execution: bool = False) -> ExecutableManifest:
    """Orchestrates the Compiler pipeline to emit a frozen execution target."""
    states = model.components(State) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, State)]
    params = model.components(Parameter) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, Parameter)]
    observables = model.components(Observable) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, Observable)]
    domains = model.components(Domain) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, Domain)]
    comp_domains = model.components(CompositeDomain) if hasattr(model, "components") else [attr for attr in model.__dict__.values() if isinstance(attr, CompositeDomain)]
    all_domains = domains + comp_domains

    layout = MemoryLayout(states, params, observables, all_domains)
    default_parameters = {p.name: p.default for p in params}
    ast_payload = model.ast() if hasattr(model, "ast") else {}

    state_domain_map = {s.name: getattr(s.domain, "name", "") for s in states}
    state_max_steps = {s.name: s.max_newton_step for s in states}

    topo = None
    lib_path = ""
    cpp_source = ""
    cpr_cache = ([], [], [], [], [])

    if ast_payload:
        compiler_result = Compiler.compile(ast_payload, layout, states, observables, target, jacobian_bandwidth)

        cpp_source = compiler_result["cpp_source"]
        cpr_cache = compiler_result["cpr_cache"]
        ast_payload = compiler_result["ast_payload"]
        topo = compiler_result["topo"]
        jacobian_bandwidth = compiler_result["jacobian_bandwidth"]

        id_arr, spatial_diag, max_steps = _compute_static_metadata(layout, ast_payload, topo, state_domain_map, state_max_steps)

        meta_dict = {
            "layout": {
                "state_offsets": layout.state_offsets, "param_offsets": layout.param_offsets, "obs_offsets": layout.obs_offsets,
                "n_states": layout.n_states, "n_params": layout.n_params, "n_obs": layout.n_obs,
                "p_length": layout.p_length, "m_length": layout.m_length,
                "domain_info": layout.domain_info,
                "mesh_offsets": layout.mesh_offsets, "mesh_cache": layout.mesh_cache
            },
            "parameters": default_parameters,
            "jacobian_bandwidth": jacobian_bandwidth or 0,
            "metadata_cache": {
                "id_arr": id_arr, "spatial_diag": spatial_diag, "max_steps": max_steps
            },
            "cpr_cache": cpr_cache,
            "state_domain_map": state_domain_map,
        }
        meta_json_str = json.dumps(meta_dict, sort_keys=True)
        cpp_source += f'\nextern "C" const char* get_ion_flux_metadata() {{\n    return R"ION_FLUX_META({meta_json_str})ION_FLUX_META";\n}}\n'

        if not mock_execution:
            compiler = NativeCompiler() if cache else NativeCompiler(cache_dir=os.path.join(tempfile.gettempdir(), "nocache"))
            lib_path = compiler.compile(cpp_source, layout.n_states)
    else:
        id_arr, spatial_diag, max_steps = _compute_static_metadata(layout, ast_payload, topo, state_domain_map, state_max_steps)

    return ExecutableManifest(
        lib_path=lib_path,
        layout=layout,
        default_parameters=default_parameters,
        ast_payload=ast_payload,
        jacobian_bandwidth=jacobian_bandwidth or 0,
        cpr_cache=cpr_cache,
        id_arr=id_arr,
        spatial_diag=spatial_diag,
        max_steps=max_steps,
        state_domain_map=state_domain_map,
        cpp_source=cpp_source
    )


def _compute_static_metadata(layout: MemoryLayout, ast_payload: Dict[str, Any], topo: Optional[TopologyAnalyzer], state_domain_map: Dict[str, str], state_max_steps: Dict[str, Any]):
    """Extracts immutable topological masks (e.g. IDA differential array, Spatial Root clamping)."""
    id_arr = [0.0] * layout.n_states
    spatial_diag = [0.0] * layout.n_states
    max_steps = [0.0] * layout.n_states

    if not ast_payload or not topo:
        return id_arr, spatial_diag, max_steps

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

    # 1. Map Time Derivatives (Differential variables == 1.0)
    for eq_data in ast_payload.get("equations", []):
        state_name = eq_data["state"]
        offset, size = layout.state_offsets[state_name]

        if eq_data["type"] == "piecewise":
            d_name = state_domain_map.get(state_name, "")
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

    # 2. Mask Dirichlet Boundary Algebraic Constraints (0.0)
    for bc_data in ast_payload.get("boundaries", []):
        if bc_data["type"] == "dirichlet":
            state_name = bc_data["state"]
            offset, size = layout.state_offsets[state_name]
            d_name = state_domain_map.get(state_name, "")

            if not d_name:
                if "left" in bc_data["bcs"]: id_arr[offset] = 0.0
                if "right" in bc_data["bcs"]: id_arr[offset + size - 1] = 0.0
                continue

            axes = topo.get_axes(d_name)
            strides = topo.get_strides(d_name)
            b_axis = axes[-1]
            coord_sys = topo.domains.get(b_axis, {}).get("coord_sys", "cartesian")

            if coord_sys == "unstructured":
                surfaces = layout.mesh_offsets.get(b_axis, {}).get("surfaces", {})
                for side in bc_data["bcs"]:
                    if side in surfaces:
                        mask_off = surfaces[side]
                        for i in range(size):
                            if layout.mesh_cache.get(mask_off + i, 0.0) > 0.5:
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

    # 3. Apply Max Newton Steps Limiters
    for state_name, (offset, size) in layout.state_offsets.items():
        if state_max_steps.get(state_name) is not None:
            val = float(state_max_steps[state_name])
            for i in range(size):
                max_steps[offset + i] = val

    return id_arr, spatial_diag, max_steps