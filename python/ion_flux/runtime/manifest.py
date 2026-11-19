import json
import shutil
import os
import ctypes
import hashlib
import tempfile
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from ion_flux.compiler._1_frontend.core import PDE, State, Parameter, Observable
from ion_flux.compiler._1_frontend.spatial import Domain, CompositeDomain
from ion_flux.compiler._2_middle_end.memory_layout import MemoryLayout
from ion_flux.compiler.pipeline import Compiler
from ion_flux.compiler._4_codegen.clang_invoker import NativeCompiler

try:
    from ion_flux._core import NativeLibrary, SolverHandle
    RUST_FFI_AVAILABLE = True
except ModuleNotFoundError:
    NativeLibrary = None
    SolverHandle = None
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
                 cpp_source: str = "",
                 static_audit: Optional[Dict[str, Any]] = None,
                 math_ir: str = "",
                 compute_ir: str = ""):
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
        self.static_audit = static_audit
        self.math_ir = math_ir
        self.compute_ir = compute_ir
        
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

    def evaluate_jvp(self, y: List[float], ydot: List[float], c_j: float, v: List[float], overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
        if not self.native_lib:
            raise RuntimeError("Requires native execution.")
        p_list = self.pack_parameters(overrides or {})
        m_list = self.layout.get_mesh_data()
        return self.native_lib.evaluate_jvp(y, ydot, p_list, m_list, c_j, v)

    def evaluate_jacobian_triplets(self, y: List[float], ydot: List[float], c_j: float, overrides: Optional[Dict[str, float]] = None) -> Tuple[List[int], List[int], List[float]]:
        if not self.native_lib:
            raise RuntimeError("Requires native execution.")
        p_list = self.pack_parameters(overrides or {})
        m_list = self.layout.get_mesh_data()
        c_seeds, c_ptrs, c_rows, c_cols, c_dense = self.cpr_cache
        return self.native_lib.evaluate_jacobian_triplets(y, ydot, p_list, m_list, c_j, c_seeds, c_ptrs, c_rows, c_cols, c_dense)

    def evaluate_audit_observables(self, y: List[float], ydot: List[float], overrides: Optional[Dict[str, float]] = None) -> List[float]:
        if not self.native_lib:
            raise RuntimeError("Requires native execution.")
        n_audit = len((self.static_audit or {}).get("clamp_meta", [])) * 2
        if n_audit == 0:
            return []
        p_list = self.pack_parameters(overrides or {})
        m_list = self.layout.get_mesh_data()
        return self.native_lib.evaluate_audit_observables(y, ydot, p_list, m_list, n_audit)

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
        static_audit = meta.get("static_audit")
        
        return cls(binary_path, layout, default_parameters, ast_payload, jacobian_bandwidth, 
                   cpr_cache, id_arr, spatial_diag, max_steps, state_domain_map, cpp_source, static_audit)


COMPILER_CACHE_VERSION = "2026.04.v7"


def _compute_ast_fingerprint(ast_payload: Dict[str, Any], layout: MemoryLayout, target: str, jacobian_bandwidth: Optional[int], audit_compiler: bool = False) -> str:
    key_dict = {
        "v": COMPILER_CACHE_VERSION,
        "ast": ast_payload,
        "offsets": layout.state_offsets,
        "params": layout.param_offsets,
        "obs": layout.obs_offsets,
        "target": target,
        "bw": jacobian_bandwidth,
        "audit_compiler": audit_compiler,
    }
    raw = json.dumps(key_dict, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _encode_meta_cpp(meta_json_str: str) -> str:
    raw_bytes = meta_json_str.encode("utf-8") + b"\x00"
    hex_rows = [
        ", ".join(f"0x{b:02x}" for b in raw_bytes[i:i + 24])
        for i in range(0, len(raw_bytes), 24)
    ]
    array_body = ",\n    ".join(hex_rows)
    return (
        '#ifdef __cplusplus\nextern "C" {\n#endif\n'
        'int get_ion_flux_abi_version() { return 2; }\n'
        f'static const unsigned char ION_FLUX_META_BYTES[] = {{\n    {array_body}\n}};\n'
        'const char* get_ion_flux_metadata() {\n'
        '    return (const char*)ION_FLUX_META_BYTES;\n'
        '}\n'
        '#ifdef __cplusplus\n}\n#endif\n'
    )


def run_native_static_audit(
    manifest: ExecutableManifest,
    default_parameters: Dict[str, float],
    cpr_cache: Tuple[List, List, List, List, List],
    spatial_diag: List[float],
    max_steps: List[float],
) -> Dict[str, Any]:
    """Runs O(1) randomized CPR-vs-Enzyme JVP verification and t=0 IC/Jacobian checks."""
    n = manifest.layout.n_states
    y0, ydot0 = manifest.evaluate_ic(default_parameters)
    rng = np.random.default_rng(42)

    y_probe = np.asarray(y0, dtype=np.float64) + rng.uniform(0.01, 0.05, size=n)
    ydot_probe = np.asarray(ydot0, dtype=np.float64)
    rows, cols, vals = manifest.evaluate_jacobian_triplets(y_probe.tolist(), ydot_probe.tolist(), 1.0, default_parameters)

    r_arr = np.asarray(rows, dtype=np.intp)
    c_arr = np.asarray(cols, dtype=np.intp)
    v_arr = np.asarray(vals, dtype=np.float64)

    max_jvp_abs_diff = 0.0
    max_jvp_rel_diff = 0.0
    for seed_idx in range(3):
        vec = rng.standard_normal(n)
        jvp_enzyme = manifest.evaluate_jvp(y_probe.tolist(), ydot_probe.tolist(), 1.0, vec.tolist(), default_parameters)
        j_cpr_vec = np.bincount(r_arr, weights=v_arr * vec[c_arr], minlength=n) if len(v_arr) > 0 else np.zeros(n)
        abs_diff = float(np.max(np.abs(jvp_enzyme - j_cpr_vec)))
        jvp_scale = max(1.0, float(np.max(np.abs(jvp_enzyme))))
        rel_diff = abs_diff / jvp_scale
        if abs_diff > max_jvp_abs_diff:
            max_jvp_abs_diff = abs_diff
        if rel_diff > max_jvp_rel_diff:
            max_jvp_rel_diff = rel_diff

    res_pre = np.asarray(manifest.evaluate_residual(y0, ydot0, default_parameters), dtype=np.float64)
    id_np = np.asarray(manifest.id_arr, dtype=np.float64)
    alg_mask = id_np < 0.5
    pre_snap_alg_max_res = float(np.max(np.abs(res_pre[alg_mask]))) if np.any(alg_mask) else 0.0

    ic_info: Dict[str, Any] = {
        "pre_snap_alg_max_res": pre_snap_alg_max_res,
        "ic_converged": True,
        "ic_final_max_res": pre_snap_alg_max_res,
        "ic_iters": 0,
        "max_snap_delta": 0.0,
        "max_snap_state": "",
        "init_jac_max": 0.0,
        "init_jac_min": 0.0,
    }

    if SolverHandle is not None:
        p_list = manifest.pack_parameters(default_parameters)
        m_list = manifest.layout.get_mesh_data()
        constraints = [0.0] * n
        c_seeds, c_ptrs, c_rows, c_cols, c_dense = cpr_cache
        try:
            handle = SolverHandle(
                manifest.lib_path, n, manifest.jacobian_bandwidth,
                y0, ydot0, manifest.id_arr, constraints, p_list, m_list,
                spatial_diag, max_steps, manifest.layout.n_obs,
                c_seeds, c_ptrs, c_rows, c_cols, c_dense, False,
                None, None, None, False
            )
            diag = handle.get_diagnostics()
            y_snapped = np.asarray(handle.get_state(), dtype=np.float64)
            delta = np.abs(y_snapped - np.asarray(y0, dtype=np.float64))
            max_idx = int(np.argmax(delta)) if n > 0 else 0
            max_state_name = ""
            for s_name, (off, sz) in manifest.layout.state_offsets.items():
                if off <= max_idx < off + sz:
                    max_state_name = f"{s_name}[{max_idx - off}]" if sz > 1 else s_name
                    break
            ic_info.update({
                "ic_converged": bool(diag.get("ic_converged", True)),
                "ic_final_max_res": float(diag.get("ic_final_max_res", 0.0)),
                "ic_iters": int(diag.get("ic_iters", 0)),
                "max_snap_delta": float(delta[max_idx]) if n > 0 else 0.0,
                "max_snap_state": max_state_name,
                "init_jac_max": float(diag.get("init_jac_max", 0.0)),
                "init_jac_min": float(diag.get("init_jac_min", 0.0)),
            })
        except Exception as e:
            ic_info["ic_converged"] = False
            ic_info["ic_error"] = str(e)

    return {
        "cpr_jvp_probe_max_diff": max_jvp_rel_diff,
        "cpr_jvp_probe_abs_diff": max_jvp_abs_diff,
        "cpr_jvp_probe_passed": bool(np.isfinite(max_jvp_rel_diff) and max_jvp_rel_diff < 1e-8),
        "ic_health": ic_info,
    }


def build_manifest(model: PDE, target: str = "cpu:serial", cache: bool = True, jacobian_bandwidth: Optional[int] = None, mock_execution: bool = False, audit_compiler: bool = False) -> ExecutableManifest:
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
    state_max_steps = {s.name: s.max_newton_step for s in states}

    if cache and not mock_execution and ast_payload:
        cache_dir = os.path.expanduser("~/.cache/ion_flux/jit_cache")
        ast_hash = _compute_ast_fingerprint(ast_payload, layout, target, jacobian_bandwidth, audit_compiler=audit_compiler)
        idx_path = os.path.join(cache_dir, f"idx_{ast_hash}.txt")

        if os.path.exists(idx_path):
            try:
                with open(idx_path, "r") as fp:
                    cached_lib = fp.read().strip()
                if cached_lib and os.path.exists(cached_lib):
                    manifest = ExecutableManifest.load(cached_lib)
                    manifest.ast_payload = ast_payload
                    try:
                        os.utime(cached_lib, None)
                    except OSError:
                        pass
                    if not manifest.cpp_source:
                        cpp_candidate = os.path.splitext(cached_lib)[0].replace("lib_res_", "res_") + ".cpp"
                        if os.path.exists(cpp_candidate):
                            try:
                                with open(cpp_candidate, "r") as cfp:
                                    manifest.cpp_source = cfp.read()
                            except OSError:
                                pass
                    return manifest
            except Exception:
                pass

    state_domain_map = {s.name: getattr(s.domain, "name", "") for s in states}
    lib_path = ""
    cpp_source = ""
    math_ir_str = ""
    compute_ir_str = ""
    id_arr = [0.0] * layout.n_states
    cpr_cache = ([], [], [], [], [])
    static_audit: Optional[Dict[str, Any]] = None

    if ast_payload:
        compiler_result = Compiler.compile(
            ast_payload, layout, states, observables, target, jacobian_bandwidth, audit_compiler=audit_compiler
        )

        cpp_source = compiler_result["cpp_source"]
        math_ir_str = compiler_result["math_pretty"]
        compute_ir_str = compiler_result["compute_pretty"]
        cpr_cache = compiler_result["cpr_cache"]
        id_arr = compiler_result["id_arr"]
        ast_payload = compiler_result["ast_payload"]
        jacobian_bandwidth = compiler_result["jacobian_bandwidth"]

        spatial_diag, max_steps = _compute_static_metadata(layout, state_max_steps)

        if audit_compiler:
            static_audit = {
                "audit_compiler": True,
                "boundary_ledger": compiler_result["boundary_ledger"],
                "clamp_meta": compiler_result["clamp_meta"],
                "math_pretty": compiler_result["math_pretty"],
                "compute_pretty": compiler_result["compute_pretty"],
                "cpp_sha256": hashlib.sha256(cpp_source.encode("utf-8")).hexdigest()[:16],
            }

        meta_dict: Dict[str, Any] = {
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

        if not mock_execution:
            compiler = NativeCompiler() if cache else NativeCompiler(cache_dir=os.path.join(tempfile.gettempdir(), "nocache"))
            if audit_compiler and static_audit is not None:
                static_audit["toolchain"] = compiler.get_toolchain_info()
                meta_dict["static_audit"] = static_audit

            meta_cpp = _encode_meta_cpp(json.dumps(meta_dict, sort_keys=True))
            lib_path = compiler.compile(cpp_source, layout.n_states, meta_source=meta_cpp)
            if cache and ast_payload:
                try:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(idx_path, "w") as fp:
                        fp.write(lib_path)
                except OSError:
                    pass
    else:
        id_arr = [0.0] * layout.n_states
        spatial_diag, max_steps = _compute_static_metadata(layout, state_max_steps)

    manifest = ExecutableManifest(
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
        cpp_source=cpp_source,
        static_audit=static_audit,
        math_ir=math_ir_str,
        compute_ir=compute_ir_str,
    )
    if audit_compiler and static_audit is not None and not mock_execution and RUST_FFI_AVAILABLE and manifest.native_lib is not None:
        static_audit.update(run_native_static_audit(manifest, default_parameters, cpr_cache, spatial_diag, max_steps))
    return manifest


def _compute_static_metadata(layout: MemoryLayout, state_max_steps: Dict[str, Any]):
    """Extracts per-state solver step limiters and spatial diagnostic masks."""
    spatial_diag = [0.0] * layout.n_states
    max_steps = [0.0] * layout.n_states

    for state_name, (offset, size) in layout.state_offsets.items():
        if state_max_steps.get(state_name) is not None:
            val = float(state_max_steps[state_name])
            for i in range(size):
                max_steps[offset + i] = val

    return spatial_diag, max_steps